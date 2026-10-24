"""Latent MXFP4 selection, reload lifecycle, and fused/unfused MoE plumbing."""

import unittest
from contextlib import nullcontext
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3_rocm_mxfp4 import LatentMXFP4Projections
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    def forward(self, x):
        return self.down_proj(self.gate_up_proj(x))


def _moe(*, prepared: bool | None):
    moe = kimi_k3.KimiK3MoE.__new__(kimi_k3.KimiK3MoE)
    nn.Module.__init__(moe)
    moe.use_latent_moe = True
    moe.moe_hidden_size = 2
    moe.tp_size = 1
    moe.alt_stream = None
    moe.fuse_ar_norm = False
    moe._sbo_shared_overlap = False
    moe._shared_experts_tp_comm = False
    moe._use_mega_moe = False
    moe._ep_front_eligible = False
    moe._routed_needs_reduce = False
    moe._front_fp32 = False
    moe._moe_front_needs_dense_bf16 = True
    moe._route_quant_fuse_eligible = False
    moe.gate = nn.Linear(4, 1, bias=False)
    moe.routed_expert_down_proj = _Linear(4, 2, bias=False)
    moe.routed_expert_up_proj = _Linear(2, 4, bias=False)
    moe.routed_expert_norm = nn.Identity()
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
    if prepared is None:
        moe._latent_mxfp4 = None
        return moe
    moe._latent_mxfp4 = LatentMXFP4Projections(min_tokens=2)
    if prepared:
        moe._latent_mxfp4.down_weight = moe.routed_expert_down_proj.weight.detach() * 2
        moe._latent_mxfp4.up_weight = moe.routed_expert_up_proj.weight.detach() * 3
        moe._latent_mxfp4.down_scale = torch.ones(1)
        moe._latent_mxfp4.up_scale = torch.ones(1)
    return moe


class TestLatentMXFP4Runtime(CustomTestCase):
    @torch.inference_mode()
    def test_fused_and_unfused_preserve_routing_shared_and_threshold(self):
        # Only the external GEMM and distributed boundaries are replaced.
        with (
            get_context().override_server_args(tp_size=1),
            get_parallel().override(tp_group=None),
            mock.patch.object(
                latent_mxfp4_aiter_hip,
                "run",
                side_effect=lambda *, x, weight, scale: F.linear(x, weight),
            ),
            mock.patch.object(
                kimi_k3,
                "use_symmetric_memory",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            mock.patch.object(
                kimi_k3, "tensor_model_parallel_all_reduce", side_effect=lambda x: x
            ),
        ):
            for prepared in (None, False, True):
                moe = _moe(prepared=prepared)
                for tokens in (1, 2, 3):
                    x = torch.randn(tokens, 4)
                    use_mxfp4 = prepared and tokens >= 2
                    down = moe.routed_expert_down_proj.weight
                    up = moe.routed_expert_up_proj.weight
                    if use_mxfp4:
                        down = moe._latent_mxfp4.down_weight
                        up = moe._latent_mxfp4.up_weight
                    routed = F.linear(x, down) * moe.gate(x).sigmoid()
                    expected = F.linear(routed, up) + moe.shared_experts(x)
                    for method in (moe._forward_fused, moe._forward_unfused):
                        with self.subTest(
                            prepared=prepared, tokens=tokens, method=method.__name__
                        ):
                            actual = method(x, prefix_sum=None)
                            torch.testing.assert_close(actual, expected)

    def test_prepare_and_reload_invalidate_stale_copies(self):
        projections = LatentMXFP4Projections(min_tokens=2048)
        self.assertFalse(projections.can_run(num_tokens=2048))
        down = torch.empty((3584, 7168), dtype=torch.bfloat16, device="meta")
        up = torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta")
        with (
            mock.patch.object(latent_mxfp4_aiter_hip, "supported", return_value=True),
            mock.patch.object(
                latent_mxfp4_aiter_hip,
                "pack",
                side_effect=[
                    (torch.ones(1), torch.ones(1)),
                    (torch.ones(1), torch.ones(1)),
                ],
            ),
        ):
            projections.prepare(down_weight=down, up_weight=up)
        self.assertFalse(projections.can_run(num_tokens=2047))
        self.assertTrue(projections.can_run(num_tokens=2048))
        self.assertTrue(projections.can_run(num_tokens=2049))
        self.assertEqual(dict(projections.state_dict()), {})
        with self.assertLogs("sglang.srt.models.kimi_k3_rocm_mxfp4", level="WARNING"):
            projections.prepare(down_weight=down.to(torch.float16), up_weight=up)
        self.assertFalse(projections.can_run(num_tokens=2048))
        self.assertIsNone(projections.up_weight)
        with self.assertRaisesRegex(RuntimeError, "has not been prepared"):
            projections.up(x=torch.ones(1, 3584))

    def test_rejects_nonpositive_token_floor(self):
        for min_tokens in (0, -1):
            with self.subTest(min_tokens=min_tokens):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    LatentMXFP4Projections(min_tokens=min_tokens)


if __name__ == "__main__":
    unittest.main()
