"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
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
from sglang.srt.models.kimi_k3_amd import kda as kimi_k3_kda_hip
from sglang.srt.models.kimi_k3_amd.kda import (
    KimiK3HipFusedVerify,
    build_fused_verify_hip,
    prepare_fused_kda_hip,
)
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
    return SimpleNamespace(
        attn=SimpleNamespace(
            conv_weights=torch.zeros(3 * heads * dim, 4),
            A_log=torch.zeros(heads),
            dt_bias=torch.zeros(heads * dim),
            lower_bound=-5.0,
        ),
        f_b_proj=SimpleNamespace(weight=f_b_weight),
        use_full_rank_gate=True,
        _fused_decode_f_b_weight=lambda: f_b_weight,
        o_norm=SimpleNamespace(weight=torch.ones(dim, dtype=torch.bfloat16), eps=1e-5),
        config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                dtype=SimpleNamespace(temporal=torch.bfloat16)
            )
        ),
    )


class TestHipFusedDecodeArming(CustomTestCase):
    """prepare_fused_kda_hip on a stand-in attention module."""

    def test_failed_rearm_drops_the_previous_arming(self):
        """A re-run that cannot arm (here the backend reports unavailable) must not
        leave the previous f_b serving the fused decode or verify."""
        stale = ("stale f_b", None, 1e-5, None)
        layer = SimpleNamespace(
            conv_weights=None,
            _k3_hip_fused_decode_args=stale,
            _k3_hip_fused_decode_backend="aiter",
            _k3_hip_fused_decode_ops=kda_fused_decode_aiter_hip,
            kda_fused_verify="stale hook",
        )
        attention = SimpleNamespace(
            attn=layer,
            _kda_hip_fused_decode_ready=True,
            f_b_proj=SimpleNamespace(weight=torch.empty(12 * 128, 128)),
            config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    dtype=SimpleNamespace(temporal=torch.bfloat16)
                )
            ),
        )

        with (
            get_context().override_server_args(),
            envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
            mock.patch.object(
                kda_fused_decode_aiter_hip, "available", return_value=False
            ),
        ):
            prepare_fused_kda_hip(attention)

        self.assertFalse(attention._kda_hip_fused_decode_ready)
        self.assertIsNone(layer._k3_hip_fused_decode_args)
        self.assertIsNone(layer._k3_hip_fused_decode_ops)
        self.assertIsNone(layer.kda_fused_verify)

    def test_fused_verify_arms_only_for_supported_verify_layouts(self):
        """The fused verify covers chain verify with a slot-indexed ring only. Tree
        speculative decoding and the host-only request ring must not probe, arm or
        warm it; neither does a non-BF16 activation dtype. An armed layer installs a
        hook for the warmed draft-token count."""
        decode_args = (torch.empty(12, 128, 128), None, 1e-5, None)
        for topk, host_only, dtype, armed in (
            (4, False, torch.bfloat16, False),
            (1, True, torch.bfloat16, False),
            (1, False, torch.float16, False),
            (1, False, torch.bfloat16, True),
        ):
            layer = SimpleNamespace(
                conv_weights=None,
                dt_bias=None,
                lower_bound=-5.0,
                _k3_hip_fused_decode_args=decode_args,
                kda_fused_verify=None,
            )
            with (
                self.subTest(topk=topk, host_only=host_only, dtype=dtype),
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
                mock.patch.object(kimi_k3_kda_hip, "print_warning_once") as warn,
            ):
                hook = build_fused_verify_hip(
                    layer=layer,
                    use_full_rank_gate=True,
                    state_dtype=torch.bfloat16,
                    activation_dtype=dtype,
                )
                self.assertEqual(available.called, armed)
                self.assertEqual(warmup.called, armed)
                self.assertEqual(warn.called, not armed)
                self.assertEqual(hook is not None, armed)
                if armed:
                    self.assertIsInstance(hook, KimiK3HipFusedVerify)
                    self.assertEqual(hook.num_draft_tokens, 5)

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
                    envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override(backend),
                    mock.patch.object(
                        fused_ops, "available", return_value=True
                    ) as available,
                    mock.patch.object(
                        attention,
                        "_fused_decode_f_b_weight",
                        wraps=attention._fused_decode_f_b_weight,
                    ) as f_b_weight,
                    mock.patch.object(fused_ops, "warmup") as warmup,
                    mock.patch.object(kimi_k3_kda_hip, "print_warning_once") as warn,
                ):
                    prepare_fused_kda_hip(attention)

                    self.assertEqual(attention._kda_hip_fused_decode_ready, armed)
                    self.assertIsNone(attention.attn.kda_fused_verify)
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
                    linear_attn_verify_backend="aiter",
                    **flags,
                ),
                envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
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
                mock.patch.object(kimi_k3_kda_hip, "print_warning_once"),
            ):
                prepare_fused_kda_hip(attention)

                self.assertFalse(attention._kda_hip_fused_decode_ready)
                self.assertIsNotNone(attention.attn._k3_hip_fused_decode_args)
                self.assertIsNone(attention.attn._k3_hip_fused_decode_ops)
                self.assertEqual(attention.attn.kda_fused_verify.num_draft_tokens, 5)
                decode_warmup.assert_not_called()
                verify_warmup.assert_called_once()
                self.assertEqual(
                    verify_warmup.call_args.kwargs["replay_ring"],
                    get_exec().mamba.enable_linear_replayssm_spec,
                )

    def test_fp16_activations_leave_the_fused_verify_unarmed(self):
        """The kernel takes BF16 f_a, but a Quark FP8 f_b arms its BF16 copy under
        any activation dtype; --dtype float16 must keep verify on the Triton chain."""
        attention = _armable_hip_attention()
        attention.o_norm.weight = torch.ones(128, dtype=torch.float16)
        with (
            get_context().override_server_args(
                speculative_num_draft_tokens=5,
                speculative_eagle_topk=1,
                linear_attn_verify_backend="aiter",
            ),
            envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
            mock.patch.object(
                kda_fused_decode_aiter_hip, "available", return_value=True
            ),
            mock.patch.object(kda_fused_decode_aiter_hip, "warmup"),
            mock.patch.object(
                kda_fused_verify_aiter_hip, "available", return_value=True
            ),
            mock.patch.object(kda_fused_verify_aiter_hip, "warmup"),
            mock.patch.object(kimi_k3_kda_hip, "print_warning_once"),
        ):
            prepare_fused_kda_hip(attention)

        self.assertIsNone(attention.attn.kda_fused_verify)

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


class TestFusedVerifyOp(CustomTestCase):
    """The adapter is fail-closed: an AITER op that predates the metadata it reads
    (state_dtypes, max_tokens, num_warps) must leave the fused verify unarmed."""

    def _install_aiter(self, op) -> None:
        module = "aiter.ops.flydsl.linear_attention_kernels"
        stubs = {
            "aiter": SimpleNamespace(),
            "aiter.ops": SimpleNamespace(),
            "aiter.ops.flydsl": SimpleNamespace(),
            module: SimpleNamespace(flydsl_kda_verify_with_f_b=op),
        }
        patcher = mock.patch.dict(sys.modules, stubs)
        patcher.start()
        self.addCleanup(patcher.stop)
        kda_fused_verify_aiter_hip._op.cache_clear()
        self.addCleanup(kda_fused_verify_aiter_hip._op.cache_clear)

    def test_op_without_num_warps_is_not_used(self):
        op = SimpleNamespace(state_dtypes=(), max_tokens=8)
        self._install_aiter(op)

        with mock.patch.object(
            kda_fused_verify_aiter_hip, "print_warning_once"
        ) as warn:
            self.assertIsNone(kda_fused_verify_aiter_hip._op())
        warn.assert_called_once()
        self.assertIn("num_warps", warn.call_args.args[0])

        op.num_warps = (4, 8)
        kda_fused_verify_aiter_hip._op.cache_clear()
        self.assertIs(kda_fused_verify_aiter_hip._op(), op)


class TestFusedVerifyUnsupported(CustomTestCase):
    def test_request_indexed_ring_keeps_the_unfused_verify(self):
        """The fused kernel writes the ReplaySSM ring at the pool slot; a ring
        indexed per request (unified memory, host-only checkpoints) must take the
        unfused chain, which writes at the request's ring rows."""
        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        for request_ring, replayssm_on, unsupported in (
            (False, True, None),
            (True, False, None),
            (True, True, "request-indexed ReplaySSM ring"),
        ):
            backend.state_pool = SimpleNamespace(replayssm_request_ring=request_ring)
            with self.subTest(request_ring=request_ring, replayssm_on=replayssm_on):
                self.assertEqual(
                    backend._fused_verify_unsupported(
                        ragged_layout=None,
                        retrieve_parent_token=None,
                        replayssm_on=replayssm_on,
                    ),
                    unsupported,
                )

    def test_unsupported_batch_never_reaches_the_kernel(self):
        """A batch the backend marks unsupported must skip the kernel entirely and
        take the unfused gate, whose output is the plain f_b projection."""
        weight = torch.zeros(12, 128, 128, dtype=torch.bfloat16)
        torch.manual_seed(0)
        weight.normal_()
        hook = KimiK3HipFusedVerify(
            f_b_weight=weight,
            norm_weight=None,
            norm_eps=1e-5,
            a_log=None,
            num_draft_tokens=5,
        )
        with (
            mock.patch.object(
                kda_fused_verify_aiter_hip, "covered", side_effect=AssertionError
            ),
            mock.patch.object(
                kda_fused_verify_aiter_hip, "run", side_effect=AssertionError
            ),
            mock.patch.object(kda_fused_verify_aiter_hip, "warn_rejected_once") as warn,
        ):
            out = hook.forward(
                layer=None,
                mixed_qkv=None,
                a=None,
                b=None,
                conv_states=None,
                ssm_states=None,
                cache_indices=None,
                intermediate_state_indices=None,
                intermediate_state_cache=None,
                intermediate_conv_window_cache=None,
                num_draft_tokens=5,
                replay_ring=None,
                unsupported="request-indexed ReplaySSM ring",
            )

        self.assertIsNone(out)
        warn.assert_called_once_with("request-indexed ReplaySSM ring")
        # 15 tokens is above the tiny-GEMM limit of 12, so this is F.linear.
        a = torch.randn(1, 15, 128).to(torch.bfloat16)
        gate = hook.unfused_gate(a)
        self.assertEqual(gate.shape, (1, 15, 12, 128))
        want = torch.nn.functional.linear(a[0], weight.flatten(0, 1))
        torch.testing.assert_close(gate, want.reshape(1, 15, 12, 128))


if __name__ == "__main__":
    unittest.main()
