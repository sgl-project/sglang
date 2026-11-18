"""Latent MXFP4 selection, reload lifecycle, and fused/unfused MoE plumbing."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.srt.environ import envs
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3_amd import latent_mxfp4
from sglang.srt.models.kimi_k3_amd import moe as kimi_k3_moe_hip
from sglang.srt.models.kimi_k3_amd.latent_mxfp4 import (
    LatentMXFP4Projections,
    build_latent_mxfp4_projections,
)
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
    # gfx950: AITER's fused_moe, not the A8W4 grouped GEMMs, runs the toy experts
    moe._a8w4_moe_min_tokens = None
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
    moe._latent_mxfp4 = LatentMXFP4Projections()
    if prepared:
        moe._latent_mxfp4.down_weight = moe.routed_expert_down_proj.weight.detach() * 2
        moe._latent_mxfp4.up_weight = moe.routed_expert_up_proj.weight.detach() * 3
        moe._latent_mxfp4.down_scale = torch.ones(1)
        moe._latent_mxfp4.up_scale = torch.ones(1)
    return moe


def _toy_runtime() -> ExitStack:
    """Replace only the external GEMM and distributed boundaries, and scale the
    2048-token MXFP4 floor down to toy batches of 2 tokens."""
    stack = ExitStack()
    stack.enter_context(get_context().override_server_args(tp_size=1))
    stack.enter_context(get_parallel().override(tp_group=None))
    for target, name, value in (
        (
            latent_mxfp4_aiter_hip,
            "run",
            lambda *, x, weight, scale: F.linear(x, weight),
        ),
        (kimi_k3, "use_symmetric_memory", lambda *args, **kwargs: nullcontext()),
        (kimi_k3, "tensor_model_parallel_all_reduce", lambda x: x),
    ):
        stack.enter_context(mock.patch.object(target, name, side_effect=value))
    stack.enter_context(mock.patch.object(latent_mxfp4, "_MIN_TOKENS", 2))
    return stack


def _expected(moe, x, *, use_mxfp4: bool) -> torch.Tensor:
    down = moe.routed_expert_down_proj.weight
    up = moe.routed_expert_up_proj.weight
    if use_mxfp4:
        down = moe._latent_mxfp4.down_weight
        up = moe._latent_mxfp4.up_weight
    routed = F.linear(x, down) * moe.gate(x).sigmoid()
    return F.linear(routed, up) + moe.shared_experts(x)


class _Gfx950Tail(Exception):
    """Raised where a batch enters the BF16 gfx950 fused path."""


class TestLatentMXFP4Runtime(CustomTestCase):
    @torch.inference_mode()
    def test_fused_and_unfused_preserve_routing_shared_and_threshold(self):
        with _toy_runtime():
            for prepared in (None, False, True):
                moe = _moe(prepared=prepared)
                for tokens in (1, 2, 3):
                    x = torch.randn(tokens, 4)
                    expected = _expected(
                        moe, x, use_mxfp4=bool(prepared) and tokens >= 2
                    )
                    for method in (moe._forward_fused, moe._forward_unfused):
                        with self.subTest(
                            prepared=prepared, tokens=tokens, method=method.__name__
                        ):
                            actual = method(x, prefix_sum=None)
                            torch.testing.assert_close(actual, expected)

    @torch.inference_mode()
    def test_gfx950_forward_runs_mxfp4_batches_off_the_hip_tails(self):
        """gfx950's _forward_fused_hip and its row-sharded tail run the BF16
        latent projections, so taking them for an MXFP4-eligible batch would
        silently ignore --enable-kimi-k3-latent-mxfp4. Such batches must run
        the MXFP4 pair on every row; smaller batches keep the gfx950 path."""
        with (
            _toy_runtime(),
            mock.patch.object(kimi_k3, "_is_gfx95", True),
            mock.patch.object(kimi_k3_moe_hip, "_is_gfx95", True),
            # None leaves each GEMM to torch; CPU hosts never import the HIP one
            mock.patch.object(
                kimi_k3, "k3_bf16_gemm_hip", return_value=None, create=True
            ),
        ):
            moe = _moe(prepared=True)
            moe._eligible_for_fused_front = True
            # The BF16 gfx950 body starts by choosing its tail; stop it there.
            with mock.patch.object(
                kimi_k3_moe_hip.KimiK3MoEHipMixin,
                "_row_sharded_tail_eligible",
                side_effect=_Gfx950Tail,
            ):
                for tokens in (1, 2, 3):
                    with self.subTest(tokens=tokens):
                        x = torch.randn(tokens, 4)
                        if tokens < 2:
                            with self.assertRaises(_Gfx950Tail):
                                moe.forward(x)
                        else:
                            torch.testing.assert_close(
                                moe.forward(x), _expected(moe, x, use_mxfp4=True)
                            )
            # nor may the decoder shard the residual by rows for that tail
            moe.tp_size = 8
            with (
                mock.patch.object(kimi_k3_moe_hip, "_ROW_SHARDED_TAIL_MIN_TOKENS", 2),
                mock.patch.object(
                    kimi_k3_moe_hip.k3_ar_fusion, "enabled", return_value=False
                ),
            ):
                self.assertFalse(moe.takes_row_sharded_tail(2))
                moe._latent_mxfp4 = None
                self.assertTrue(moe.takes_row_sharded_tail(2))

    @torch.inference_mode()
    def test_mxfp4_front_routes_on_the_same_logits(self):
        """MXFP4 replaces only the routed input. The router logits must keep the
        merged front's dtype (fp32 on gfx950 + AITER), or the top-k picks can
        differ from a run without the flag."""
        moe = _moe(prepared=True)
        moe._front_w = moe._front_w.bfloat16()
        moe._latent_mxfp4.down_weight = moe._latent_mxfp4.down_weight.bfloat16()
        x = torch.randn(3, 4, dtype=torch.bfloat16)

        def gemm(x, weight, out=None, out_dtype=None):
            return F.linear(x.float(), weight.float()).to(out_dtype or x.dtype)

        with (
            _toy_runtime(),
            # torch has no CPU bf16 -> fp32 mm; the boundary GEMM keeps the dtype contract
            mock.patch.object(kimi_k3, "_k3_bf16_gemm", side_effect=gemm),
        ):
            for front_fp32 in (False, True):
                moe._front_fp32 = front_fp32
                with self.subTest(front_fp32=front_fp32):
                    _, merged_logits, _ = moe._fused_front(x)
                    _, mxfp4_logits, _ = moe._fused_front(x, latent_mxfp4=True)
                    self.assertEqual(mxfp4_logits.dtype, merged_logits.dtype)
                    torch.testing.assert_close(
                        mxfp4_logits, merged_logits, rtol=0, atol=0
                    )

    def test_prepare_and_reload_refill_the_copies_in_place(self):
        projections = LatentMXFP4Projections()
        self.assertFalse(projections.can_run(num_tokens=2048))
        down = torch.empty((3584, 7168), dtype=torch.bfloat16, device="meta")
        up = torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta")
        with mock.patch.object(
            latent_mxfp4_aiter_hip,
            "pack",
            side_effect=[
                (torch.ones(1), torch.ones(1)),
                (torch.ones(1), torch.ones(1)),
                (torch.full((1,), 2.0), torch.ones(1)),
                (torch.full((1,), 3.0), torch.ones(1)),
            ],
        ):
            projections.prepare(down_weight=down, up_weight=up)
            # A reload refills the copies captured CUDA graphs already point at.
            buffers = (projections.down_weight, projections.up_weight)
            projections.prepare(down_weight=down, up_weight=up)
        self.assertIs(projections.down_weight, buffers[0])
        self.assertIs(projections.up_weight, buffers[1])
        self.assertEqual(projections.down_weight.item(), 2.0)
        self.assertEqual(projections.up_weight.item(), 3.0)
        self.assertFalse(projections.can_run(num_tokens=2047))
        self.assertTrue(projections.can_run(num_tokens=2048))
        self.assertEqual(dict(projections.state_dict()), {})

    def test_prepare_raises_on_weights_it_cannot_pack(self):
        """The user asked for MXFP4, so a checkpoint whose latent weights are not
        the BF16 (3584, 7168) / (7168, 3584) pair fails at load time instead of
        silently serving BF16."""
        down = torch.empty((3584, 7168), dtype=torch.bfloat16, device="meta")
        up = torch.empty((7168, 3584), dtype=torch.bfloat16, device="meta")
        projections = LatentMXFP4Projections()
        for name, bad_down, bad_up in (
            ("dtype", down.to(torch.float16), up),
            ("shape", down, up.view(3584, 7168)),
        ):
            with (
                self.subTest(name),
                mock.patch.object(latent_mxfp4_aiter_hip, "pack") as pack,
                self.assertRaisesRegex(ValueError, "--enable-kimi-k3-latent-mxfp4"),
            ):
                projections.prepare(down_weight=bad_down, up_weight=bad_up)
            pack.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "has not been prepared"):
            projections.up(x=torch.ones(1, 3584))

    def test_builder_raises_where_the_flag_cannot_apply(self):
        """SGLANG_USE_AITER=0, quantized latent projections, an MoE a2a backend and
        deterministic inference (the token-count switch to lossy MXFP4 would make
        outputs batch dependent) each fail the explicit opt-in loudly."""
        ok = dict(
            use_aiter=True, latent_quantized=False, moe_a2a=False, deterministic=False
        )
        with (
            mock.patch.object(latent_mxfp4_aiter_hip, "available", return_value=True),
            mock.patch.object(latent_mxfp4, "is_hip", return_value=True),
        ):
            for broken in (None, *ok):
                case = dict(ok)
                if broken is not None:
                    case[broken] = not case[broken]
                use_aiter = case.pop("use_aiter")
                with (
                    self.subTest(broken=broken),
                    envs.SGLANG_USE_AITER.override(use_aiter),
                ):
                    if broken is None:
                        self.assertIsInstance(
                            build_latent_mxfp4_projections(**case),
                            LatentMXFP4Projections,
                        )
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "--enable-kimi-k3-latent-mxfp4"
                        ):
                            build_latent_mxfp4_projections(**case)

    def test_gemm_is_available_on_gfx950_only(self):
        """gemm_a4w4's FP4 path is built for gfx950; other HIP targets must
        report it unavailable so the flag raises instead of launching it."""
        for gcn_arch, expected in (
            ("gfx950", True),
            ("gfx942", False),
            ("gfx1250", False),
        ):
            with (
                self.subTest(gcn_arch=gcn_arch),
                mock.patch.object(latent_mxfp4_aiter_hip.torch.version, "hip", "7.0"),
                mock.patch.object(
                    latent_mxfp4_aiter_hip.torch.cuda, "is_available", lambda: True
                ),
                mock.patch.object(
                    latent_mxfp4_aiter_hip,
                    "current_arch",
                    lambda: SimpleNamespace(hip=SimpleNamespace(gcn_arch=gcn_arch)),
                ),
                mock.patch.object(latent_mxfp4_aiter_hip, "_ops", lambda: object()),
            ):
                self.assertIs(latent_mxfp4_aiter_hip.available.__wrapped__(), expected)


if __name__ == "__main__":
    unittest.main()
