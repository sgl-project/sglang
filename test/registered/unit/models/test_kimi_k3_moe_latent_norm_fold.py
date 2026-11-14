"""Kimi-K3 latent MoE norm fold (SGLANG_OPT_HIP_K3_FOLD_MOE_LATENT_NORM): the
post-load fold and the fused-front tail that finishes the norm."""

import unittest
from contextlib import nullcontext
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight), None


class _TopK(nn.Module):
    def forward(self, x, logits):
        return logits


class _Experts(nn.Module):
    def forward(self, x, logits):
        return x * logits.sigmoid()


class _Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)
        self.act_fn = nn.Identity()


def _fused_front_moe():
    """A 4-wide KimiK3MoE on the fused front with a 2-wide latent: only the
    latent norm and up_proj are the real modules under test."""
    moe = kimi_k3.KimiK3MoE.__new__(kimi_k3.KimiK3MoE)
    nn.Module.__init__(moe)
    moe.use_latent_moe = True
    moe.moe_hidden_size = 2
    moe.tp_size = 1
    moe.alt_stream = None
    moe.fuse_ar_norm = False
    moe._eligible_for_fused_front = True
    moe._front_fp32 = False
    moe._front_block_w = None
    moe._moe_front_needs_dense_bf16 = True
    moe._route_quant_fuse_eligible = False
    moe._latent_norm_folded = False
    moe._derived_weight_cache_error = None
    moe.gate = nn.Linear(4, 1, bias=False)
    moe.routed_expert_down_proj = _Linear(4, 2, bias=False)
    moe.routed_expert_up_proj = _Linear(2, 4, bias=False)
    moe.routed_expert_up_proj.quant_method = UnquantizedLinearMethod()
    moe.routed_expert_norm = RMSNorm(2, eps=1e-5)
    moe.routed_expert_norm.weight.data = torch.tensor([0.5, 3.0])
    # CustomOp would dispatch on the host GPU platform; the math is the contract.
    moe.routed_expert_norm.forward = moe.routed_expert_norm.forward_native
    moe.shared_experts = _Shared()
    moe.experts = _Experts()
    moe.topk = _TopK()
    moe._front_sizes = [4, 1, 2]
    moe._front_w = torch.cat(
        [
            moe.shared_experts.gate_up_proj.weight,
            moe.gate.weight,
            moe.routed_expert_down_proj.weight,
        ]
    ).detach()
    return moe


def _rstd_scale_add3_reference(a, x, b, c=None, *, eps):
    y = a * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) + b
    return y if c is None else y + c


class TestLatentNormFold(CustomTestCase):
    @torch.inference_mode()
    def test_folded_tail_keeps_the_fused_front_output(self):
        """After the fold, the fused front must hand the pre-norm latent, the norm
        eps and the shared and prefix rows to the rstd tail, a repeated
        post-load fold must not scale the weight twice, and online weight updates,
        which would overwrite one half of the product, must be rejected."""
        with (
            get_context().override_server_args(tp_size=1),
            get_parallel().override(tp_group=None),
            mock.patch.object(
                kimi_k3,
                "use_symmetric_memory",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            mock.patch.object(
                kimi_k3, "tensor_model_parallel_all_reduce", side_effect=lambda x: x
            ),
            mock.patch.object(
                kimi_k3, "rstd_scale_add3", side_effect=_rstd_scale_add3_reference
            ) as tail,
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_OPT_HIP_K3_FOLD_MOE_LATENT_NORM.override(True),
        ):
            moe = _fused_front_moe()
            x, prefix = torch.randn(3, 4), torch.randn(3, 4)
            expected = moe._forward_fused(x, prefix_sum=prefix)
            # post_load_weights can run twice after one load.
            for _ in range(2):
                moe._fold_latent_norm()
            self.assertTrue(moe._latent_norm_folded)
            self.assertIsNotNone(moe._derived_weight_cache_error)
            for prefix_sum in (prefix, None):
                with self.subTest(prefix=prefix_sum is not None):
                    actual = moe._forward_fused(x, prefix_sum=prefix_sum)
                    torch.testing.assert_close(
                        actual,
                        expected if prefix_sum is not None else expected - prefix,
                    )
            self.assertEqual(tail.call_count, 2)

    def test_fold_leaves_the_weights_unless_it_can_run(self):
        """Multiplying the norm into a quantized (packed) up weight would corrupt
        it, the monokernel keeps checkpoint copies of both weights, and the fold
        is opt-in."""
        cases = (
            dict(quantized=True, enabled=True, monokernel=False),
            dict(quantized=False, enabled=True, monokernel=True),
            dict(quantized=False, enabled=False, monokernel=False),
        )
        for case in cases:
            with (
                self.subTest(**case),
                get_context().override_server_args(
                    enable_amd_monokernel=case["monokernel"]
                ),
                mock.patch.object(kimi_k3, "_is_hip", True),
                envs.SGLANG_OPT_HIP_K3_FOLD_MOE_LATENT_NORM.override(case["enabled"]),
            ):
                moe = _fused_front_moe()
                if case["quantized"]:
                    moe.routed_expert_up_proj.quant_method = mock.Mock()
                up = moe.routed_expert_up_proj.weight.detach().clone()
                moe._fold_latent_norm()
                self.assertFalse(moe._latent_norm_folded)
                self.assertIsNone(moe._derived_weight_cache_error)
                self.assertTrue(torch.equal(moe.routed_expert_up_proj.weight, up))
                self.assertEqual(moe.routed_expert_norm.weight.tolist(), [0.5, 3.0])


if __name__ == "__main__":
    unittest.main()
