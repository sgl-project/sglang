"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import (
    kda_fused_decode_aiter_hip,
    kda_fused_decode_gfx950,
    kda_fused_verify_aiter_hip,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8
from sglang.srt.models import kimi_k3, kimi_k3_rocm_quant
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention, _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    k3_fused_decode_f_b_weight,
)
from sglang.srt.runtime_context import get_context, get_exec
from sglang.test.test_utils import CustomTestCase


def _per_channel_fp8(out_features: int, in_features: int):
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    # Spread the per-channel magnitudes so a wrongly-broadcast scale cannot
    # coincidentally reproduce the right answer.
    dense *= torch.logspace(-1, 1, out_features).unsqueeze(1)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (dense / scale).to(torch.float8_e4m3fn)
    return weight, scale, dense


class TestChannelFp8ToBf16(CustomTestCase):
    """aiter's batched absorb GEMM takes ``w_scale`` as a scalar, so a
    per-channel vector cannot reach it. The scale axis is the GEMM's
    contraction axis, so requantizing to one per-tensor scale is lossy --
    ~2.3% relative error on real K3 weights. Dequantizing to bf16 costs ~0.12%
    and lets the absorb use its bf16 path with w_scale at the 1.0 default."""

    def test_dequantizes_every_channel_with_its_own_scale(self):
        weight, scale, dense = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        out = _k3_channel_fp8_to_bf16(module, weight)

        self.assertEqual(out.dtype, torch.bfloat16)
        # Only the original fp8 rounding plus the bf16 cast separate this from
        # the checkpoint's own values; no second quantization.
        exact = weight.to(torch.float32) * scale
        torch.testing.assert_close(out.float(), exact, rtol=0.01, atol=0.0)
        # Guard the failure this replaced: one shared scale would leave the
        # low-magnitude channels far from their true values.
        worst = ((out.float() - dense).abs().sum(1) / dense.abs().sum(1)).max()
        self.assertLess(worst.item(), 0.05)

    def test_accepts_a_column_vector_scale(self):
        """Quark serializes the scale as [out]; callers may hold [out, 1]."""
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)

        flat = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.squeeze(1)), weight
        )
        column = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.clone()), weight
        )

        torch.testing.assert_close(flat, column)


class TestMergeDtypeGuard(CustomTestCase):
    """``_merge_weights_as_views`` cats ``.weight`` alone, so fusing a quantized
    projection silently drops the scale tensors that give its integer payload
    meaning. Only dtypes that carry their full value in ``.weight`` may fuse."""

    def test_rejects_weights_that_carry_a_separate_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4).to(dtype) for _ in range(2)]
                self.assertFalse(_is_unquantized_mergeable(weights))

    def test_accepts_plain_float_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4, dtype=dtype) for _ in range(2)]
                self.assertTrue(_is_unquantized_mergeable(weights))

    def test_rejects_mixed_dtypes(self):
        weights = [
            torch.zeros(4, 4, dtype=torch.bfloat16),
            torch.zeros(4, 4, dtype=torch.float16),
        ]
        self.assertFalse(_is_unquantized_mergeable(weights))

    def test_rejects_float32(self):
        """The merged buffer is only built for the two half dtypes the fused
        KDA/MoE kernels read; fp32 must not slip in through a dtype-agreement
        check that only asks whether the operands match each other."""
        weights = [torch.zeros(4, 4, dtype=torch.float32) for _ in range(2)]
        self.assertFalse(_is_unquantized_mergeable(weights))


class TestFusedDecodeFbWeight(CustomTestCase):
    """The fused KDA decode takes a dense f_b with no scale argument,
    so a Quark per-channel FP8 f_b must reach it with every channel's scale applied."""

    def test_quark_fp8_f_b_is_dequantized_and_bf16_passes_through(self):
        weight, scale, _ = _per_channel_fp8(out_features=16, in_features=8)
        quark_f_b = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=scale.squeeze(1),
            quant_method=mock.Mock(spec=QuarkLinearMethod),
            scheme=QuarkW8A8Fp8(
                {"qscheme": "per_channel"},
                {"qscheme": "per_channel", "is_dynamic": True},
            ),
        )

        out = k3_fused_decode_f_b_weight(quark_f_b)

        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(
            out.float(), weight.float() * scale, rtol=0.01, atol=0.0
        )
        # The FP8 layer keeps serving the unfused path.
        self.assertEqual(quark_f_b.weight.dtype, torch.float8_e4m3fn)

        bf16 = torch.nn.Parameter(
            torch.randn(16, 8, dtype=torch.bfloat16), requires_grad=False
        )
        plain_f_b = SimpleNamespace(weight=bf16, quant_method=mock.Mock())
        self.assertEqual(
            k3_fused_decode_f_b_weight(plain_f_b).data_ptr(), bf16.data_ptr()
        )

    def test_post_processed_quark_f_b_disables_fused_decode_instead_of_crashing(self):
        """Sharded and remote loaders transpose the FP8 f_b before post_load_weights;
        that weight must not crash the load."""
        weight, scale, _ = _per_channel_fp8(out_features=32, in_features=16)
        scheme = QuarkW8A8Fp8(
            {"qscheme": "per_channel"},
            {"qscheme": "per_channel", "is_dynamic": True},
        )
        f_b = torch.nn.Module()
        f_b.weight = torch.nn.Parameter(weight, requires_grad=False)
        f_b.weight_scale = torch.nn.Parameter(scale.squeeze(1), requires_grad=False)
        f_b.quant_method = mock.Mock(spec=QuarkLinearMethod)
        f_b.scheme = scheme
        scheme.process_weights_after_loading(f_b)

        with mock.patch.object(kimi_k3_rocm_quant, "print_warning_once") as warn:
            self.assertIsNone(k3_fused_decode_f_b_weight(f_b))
        warn.assert_called_once()


def _armable_hip_attention() -> SimpleNamespace:
    """A stand-in KDA attention whose static inputs fit the fused decode kernel."""
    heads, dim = 12, 128
    f_b_weight = torch.zeros(heads * dim, dim, dtype=torch.bfloat16)
    attention = SimpleNamespace(
        attn=SimpleNamespace(
            conv_weights=torch.zeros(3 * heads * dim, 4),
            A_log=torch.zeros(heads),
            dt_bias=torch.zeros(heads * dim),
            lower_bound=-5.0,
        ),
        f_b_proj=SimpleNamespace(weight=f_b_weight),
        use_full_rank_gate=True,
        _fused_decode_f_b_weight=lambda: f_b_weight,
        o_norm=SimpleNamespace(weight=torch.ones(dim), eps=1e-5),
        config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                dtype=SimpleNamespace(temporal=torch.bfloat16)
            )
        ),
    )
    attention._prepare_fused_verify_hip = MethodType(
        KimiK3DeltaAttention._prepare_fused_verify_hip, attention
    )
    return attention


class TestHipFusedDecodeArming(CustomTestCase):
    """_prepare_fused_decode on ROCm, on a stand-in attention module."""

    def test_failed_rearm_drops_the_previous_arming(self):
        """A re-run that cannot arm (here the backend reports unavailable) must not
        leave the previous f_b serving the fused decode or verify."""
        stale = ("stale f_b", None, 1e-5, None)
        layer = SimpleNamespace(
            conv_weights=None,
            _k3_hip_fused_decode_args=stale,
            _k3_hip_fused_decode_backend="aiter",
            _k3_hip_fused_decode_ops=kda_fused_decode_aiter_hip,
            _k3_hip_fused_verify_args=stale,
        )
        attention = SimpleNamespace(
            attn=layer,
            _kda_hip_fused_decode_ready=True,
            _kda_hip_fused_verify_ready=True,
            f_b_proj=SimpleNamespace(weight=torch.empty(12 * 128, 128)),
            config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    dtype=SimpleNamespace(temporal=torch.bfloat16)
                )
            ),
        )

        with (
            get_context().override_server_args(),
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
            mock.patch.object(
                kda_fused_decode_aiter_hip, "available", return_value=False
            ),
        ):
            KimiK3DeltaAttention._prepare_fused_decode(attention)

        self.assertFalse(attention._kda_hip_fused_decode_ready)
        self.assertFalse(attention._kda_hip_fused_verify_ready)
        self.assertIsNone(layer._k3_hip_fused_decode_args)
        self.assertIsNone(layer._k3_hip_fused_decode_ops)
        self.assertIsNone(layer._k3_hip_fused_verify_args)

    def test_fused_verify_arms_only_for_supported_verify_layouts(self):
        """The fused verify covers chain verify with a slot-indexed ring only. Tree
        speculative decoding and the host-only request ring must not probe, arm or
        warm it; an armed layer records the warmed draft-token count."""
        decode_args = (torch.empty(12, 128, 128), None, 1e-5, None)
        for topk, host_only, armed in (
            (4, False, False),
            (1, True, False),
            (1, False, True),
        ):
            layer = SimpleNamespace(
                conv_weights=None,
                dt_bias=None,
                lower_bound=-5.0,
                _k3_hip_fused_decode_args=decode_args,
                _k3_hip_fused_verify_args=None,
            )
            attention = SimpleNamespace(
                attn=layer, use_full_rank_gate=True, _kda_hip_fused_verify_ready=False
            )
            with (
                self.subTest(topk=topk, host_only=host_only),
                get_context().override_server_args(
                    speculative_num_draft_tokens=5,
                    speculative_eagle_topk=topk,
                    enable_linear_replayssm_spec=True,
                    enable_hierarchical_cache=host_only,
                ),
                envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.override(host_only),
                mock.patch.object(
                    kda_fused_verify_aiter_hip, "available", return_value=True
                ) as available,
                mock.patch.object(kda_fused_verify_aiter_hip, "warmup") as warmup,
                mock.patch.object(kimi_k3, "print_warning_once") as warn,
            ):
                KimiK3DeltaAttention._prepare_fused_verify_hip(
                    attention, state_dtype=torch.bfloat16
                )
                self.assertEqual(attention._kda_hip_fused_verify_ready, armed)
                self.assertEqual(available.called, armed)
                self.assertEqual(warmup.called, armed)
                self.assertEqual(warn.called, not armed)
                self.assertEqual(
                    layer._k3_hip_fused_verify_args,
                    (*decode_args, 5) if armed else None,
                )

    def test_replayssm_ring_keeps_decode_from_deferring_f_b(self):
        """ReplaySSM decode must keep f_b projected and skip unused fused weights
        and warmup, rather than reporting an armed kernel that never runs."""
        for backend, fused_ops in (
            ("gfx950", kda_fused_decode_gfx950),
            ("aiter", kda_fused_decode_aiter_hip),
        ):
            for flags, armed in (
                ({}, True),
                ({"enable_linear_replayssm": True}, False),
                ({"enable_linear_replayssm_spec": True}, False),
            ):
                attention = _armable_hip_attention()
                with (
                    self.subTest(backend=backend, flags=flags),
                    get_context().override_server_args(**flags),
                    mock.patch.object(kimi_k3, "_is_hip", True),
                    envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override(backend),
                    envs.SGLANG_OPT_HIP_K3_KDA_FLYDSL_VERIFY.override(False),
                    mock.patch.object(
                        fused_ops, "available", return_value=True
                    ) as available,
                    mock.patch.object(
                        attention,
                        "_fused_decode_f_b_weight",
                        wraps=attention._fused_decode_f_b_weight,
                    ) as f_b_weight,
                    mock.patch.object(fused_ops, "warmup") as warmup,
                    mock.patch.object(kimi_k3, "print_warning_once") as warn,
                ):
                    KimiK3DeltaAttention._prepare_fused_decode(attention)

                    self.assertEqual(attention._kda_hip_fused_decode_ready, armed)
                    self.assertFalse(attention._kda_hip_fused_verify_ready)
                    self.assertEqual(
                        attention.attn._k3_hip_fused_decode_args is not None, armed
                    )
                    self.assertEqual(
                        attention.attn._k3_hip_fused_decode_ops is not None, armed
                    )
                    self.assertEqual(available.called, armed)
                    self.assertEqual(f_b_weight.called, armed)
                    self.assertEqual(warmup.called, armed)
                    self.assertEqual(warn.called, not armed)

    def test_replayssm_ring_preserves_fused_verify_without_decode_warmup(self):
        """Skipping fused ring decode must not disable supported fused verify
        or warm the unused decode kernel."""
        for flags in (
            {"enable_linear_replayssm": True},
            {"enable_linear_replayssm_spec": True},
        ):
            attention = _armable_hip_attention()
            with (
                self.subTest(flags=flags),
                get_context().override_server_args(
                    speculative_num_draft_tokens=5,
                    speculative_eagle_topk=1,
                    **flags,
                ),
                mock.patch.object(kimi_k3, "_is_hip", True),
                envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
                envs.SGLANG_OPT_HIP_K3_KDA_FLYDSL_VERIFY.override(True),
                envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.override(False),
                mock.patch.object(
                    kda_fused_decode_aiter_hip, "available", return_value=True
                ),
                mock.patch.object(
                    kda_fused_decode_aiter_hip, "warmup"
                ) as decode_warmup,
                mock.patch.object(
                    kda_fused_verify_aiter_hip, "available", return_value=True
                ),
                mock.patch.object(
                    kda_fused_verify_aiter_hip, "warmup"
                ) as verify_warmup,
                mock.patch.object(kimi_k3, "print_warning_once"),
            ):
                KimiK3DeltaAttention._prepare_fused_decode(attention)

                self.assertFalse(attention._kda_hip_fused_decode_ready)
                self.assertTrue(attention._kda_hip_fused_verify_ready)
                self.assertIsNotNone(attention.attn._k3_hip_fused_decode_args)
                self.assertIsNone(attention.attn._k3_hip_fused_decode_ops)
                self.assertEqual(
                    attention.attn._k3_hip_fused_verify_args,
                    (*attention.attn._k3_hip_fused_decode_args, 5),
                )
                decode_warmup.assert_not_called()
                verify_warmup.assert_called_once()
                self.assertEqual(
                    verify_warmup.call_args.kwargs["replay_ring"],
                    get_exec().mamba.enable_linear_replayssm_spec,
                )

    def test_block_fp8_f_b_arms_from_the_merged_bf16_copy(self):
        """A block-FP8 f_b_proj holds FP8 with no Quark scheme; the fused decode must
        take the BF16 copy _merge_bfa_weights() dequantized, and warn when none exists."""
        merged = torch.randn(12 * 128, 128, dtype=torch.bfloat16)
        attention = SimpleNamespace(
            use_full_rank_gate=True,
            _bfa_f_b_w=merged,
            f_b_proj=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.zeros(12 * 128, 128, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
                quant_method=mock.Mock(),
            ),
        )
        self.assertIs(KimiK3DeltaAttention._fused_decode_f_b_weight(attention), merged)

        attention._bfa_f_b_w = None
        with mock.patch.object(kimi_k3, "print_warning_once") as warn:
            self.assertIsNone(KimiK3DeltaAttention._fused_decode_f_b_weight(attention))
        warn.assert_called_once()


class TestHipFusedVerifyRejection(CustomTestCase):
    def test_fused_accept_keeps_verify_off_the_stale_committed_state(self):
        """Fused accept skips the commit scatter, so the committed pool is stale
        between verifies; the AITER verify reads that pool and must reject it."""
        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        backend.state_pool = SimpleNamespace(replayssm_request_ring=False)
        for accept_lens_pool, rejection in (
            (None, None),
            (torch.ones(2, dtype=torch.int32), "fused-accept state"),
        ):
            backend.accept_lens_pool = accept_lens_pool
            with self.subTest(fused_accept=accept_lens_pool is not None):
                self.assertEqual(
                    backend._hip_fused_verify_rejection(
                        ragged_layout=None,
                        retrieve_parent_token=None,
                        replayssm_on=False,
                    ),
                    rejection,
                )


if __name__ == "__main__":
    unittest.main()
