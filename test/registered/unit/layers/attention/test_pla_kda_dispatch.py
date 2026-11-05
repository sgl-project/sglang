"""PLA dispatch for the SAIL KDA/GDN backends: kernel slots, layout, fallback.

The PLA operators are out-of-tree; these cases pin which kernel slot serves
which mode, what the operators are handed, and when a batch stays on the
community Triton implementation.
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.sail.attention import gdn_backend as sail_gdn_backend
from sglang.srt.hardware_backend.sail.attention import gdn_triton as sail_gdn_triton
from sglang.srt.hardware_backend.sail.attention import kda_backend as sail_kda_backend
from sglang.srt.hardware_backend.sail.attention import kda_flashkda as sail_flashkda
from sglang.srt.hardware_backend.sail.kernels import kda as sail_kda
from sglang.srt.layers.attention.linear.kernels import gdn_triton as base_gdn_triton
from sglang.srt.layers.attention.linear.kernels import kda_flashkda, kda_triton
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fused_decode_coverage_inputs(num_heads):
    head_dim = 128
    seg = num_heads * head_dim
    batch_size = 2
    num_slots = 3
    return (
        torch.empty(batch_size, 3 * seg, dtype=torch.bfloat16),
        torch.empty(batch_size, seg, dtype=torch.bfloat16),
        torch.empty(batch_size, num_heads, dtype=torch.bfloat16),
        torch.empty(num_slots, 3, 3 * seg, dtype=torch.bfloat16),
        torch.empty(num_slots, num_heads, head_dim, head_dim, dtype=torch.float32),
        torch.arange(batch_size, dtype=torch.int32),
        torch.empty(batch_size, seg, dtype=torch.bfloat16),
    )


def _fused_decode_attention(num_heads):
    head_dim = 128
    seg = num_heads * head_dim
    layer = SimpleNamespace(
        conv_weights=torch.empty(3 * seg, 4, dtype=torch.float32),
        bias=None,
        A_log=torch.empty(num_heads, dtype=torch.float32),
        dt_bias=torch.empty(seg, dtype=torch.float32),
        num_v_heads=num_heads,
    )
    attention = SimpleNamespace(
        attn=layer,
        o_norm=SimpleNamespace(
            weight=torch.empty(head_dim, dtype=torch.float32), eps=1e-5
        ),
        _kda_fused_decode_ready=False,
    )
    return attention, layer


def _pla_decode_modules(fused_packed):
    pla_module = ModuleType("pla")
    decode_module = ModuleType("pla.decode")
    pla_module.decode = decode_module
    decode_module.fused_sigmoid_gating_delta_rule_forward_k_last = Mock()
    decode_module.fused_sigmoid_gating_delta_rule_forward_k_last_packed = fused_packed
    return {"pla": pla_module, "pla.decode": decode_module}


def _flashkda_pla_modules(flashkda_fwd):
    pla_module = ModuleType("pla")
    prefill_module = ModuleType("pla.prefill")
    flashkda_module = ModuleType("pla.prefill.flashkdapro")
    pla_module.prefill = prefill_module
    prefill_module.flashkdapro = flashkda_module
    flashkda_module.flashkda_fwd = flashkda_fwd
    return {
        "pla": pla_module,
        "pla.prefill": prefill_module,
        "pla.prefill.flashkdapro": flashkda_module,
    }


class TestPLAKDADispatch(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())

    def test_kda_dispatcher_pins_unsupported_backends_to_triton(self):
        dispatcher = sail_kda_backend.SailKDAKernelDispatcher(
            LinearAttnKernelBackend.FLASHINFER,
            LinearAttnKernelBackend.NVIDIA_KDA,
            LinearAttnKernelBackend.NV_CUTEDSL,
        )
        self.assertIsInstance(
            dispatcher.decode_kernel, sail_kda_backend.SailTritonKDAKernel
        )
        self.assertIs(dispatcher.decode_kernel, dispatcher.verify_kernel)
        self.assertIsInstance(dispatcher.extend_kernel, kda_triton.TritonKDAKernel)
        self.assertEqual(dispatcher.prefill_backend, LinearAttnKernelBackend.TRITON)

    def test_kda_dispatcher_honors_flashkda_prefill_backend(self):
        dispatcher = sail_kda_backend.SailKDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.FLASHKDA,
            LinearAttnKernelBackend.TRITON,
        )
        self.assertIsInstance(
            dispatcher.extend_kernel, sail_flashkda.SailFlashKDAKernel
        )
        self.assertEqual(dispatcher.prefill_backend, LinearAttnKernelBackend.FLASHKDA)

    def test_flashkda_extend_dispatches_to_pla_operator(self):
        num_heads, head_dim, packed_tokens = 12, 128, 64
        q = torch.zeros((1, packed_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        k, v, g = torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q)
        beta = torch.full((1, packed_tokens, num_heads), 0.5, dtype=torch.float32)
        cache_indices = torch.tensor([1], dtype=torch.int32)
        query_start_loc = torch.tensor([0, packed_tokens], dtype=torch.int32)
        flashkda_fwd = Mock(
            side_effect=lambda **kwargs: (
                kwargs["out"].fill_(0.25),
                kwargs["final_state"].copy_(kwargs["initial_state"] + 1),
            )
        )
        modules = _flashkda_pla_modules(flashkda_fwd)
        for extend_kwargs in ({}, {"return_intermediate_states": False}):
            with self.subTest(extend_kwargs=extend_kwargs):
                ssm_states = torch.zeros(
                    (2, num_heads, head_dim, head_dim), dtype=torch.float32
                )
                flashkda_fwd.reset_mock()
                with (
                    envs.SGLANG_SAIL_PLA.override(True),
                    patch.object(kda_flashkda, "_load_flash_kda") as load_flash_kda,
                    patch.dict(sys.modules, modules),
                ):
                    output = sail_flashkda.SailFlashKDAKernel().extend(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        ssm_states=ssm_states,
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc,
                        A_log=torch.zeros((1, 1, num_heads, 1), dtype=torch.float32),
                        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32),
                        lower_bound=-5.0,
                        **extend_kwargs,
                    )
                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.shape, q.shape)
                self.assertTrue(torch.all(output == 0.25))
                self.assertTrue(torch.all(ssm_states[1] == 1))
                self.assertEqual(torch.count_nonzero(ssm_states[0]).item(), 0)
                load_flash_kda.assert_not_called()
                flashkda_fwd.assert_called_once()
                kwargs = flashkda_fwd.call_args.kwargs
                self.assertEqual(
                    kwargs["q"].shape, (packed_tokens, num_heads, head_dim)
                )
                self.assertEqual(kwargs["beta"].shape, (packed_tokens, num_heads))
                self.assertEqual(kwargs["beta"].dtype, torch.bfloat16)
                self.assertEqual(kwargs["A_log"].shape, (num_heads,))
                self.assertEqual(kwargs["dt_bias"].shape, (num_heads, head_dim))
                self.assertEqual(kwargs["cu_seqlens"].dtype, torch.int64)

    def test_flashkda_fallback_preserves_return_contract(self):
        for track_ssm, lower_bound, is_spec_decode in (
            (True, -5.0, False),
            (False, None, False),
            (False, -5.0, True),
        ):
            with self.subTest(
                track_ssm=track_ssm,
                lower_bound=lower_bound,
                is_spec_decode=is_spec_decode,
            ):
                kernel = sail_flashkda.SailFlashKDAKernel()
                output = torch.zeros((1, 64, 12, 128), dtype=torch.bfloat16)
                result = (output, sentinel.intermediate_states) if track_ssm else output
                track_state = sentinel.track_state if track_ssm else None
                track_chunk_idx = sentinel.track_chunk_idx if track_ssm else None
                with (
                    envs.SGLANG_SAIL_PLA.override(True),
                    patch.object(kernel, "_flashkda_extend") as fused_extend,
                    patch(
                        "sglang.kernels.ops.attention.fla.kda.chunk_kda",
                        return_value=result,
                    ) as chunk_kda,
                ):
                    actual = kernel.extend(
                        sentinel.q,
                        sentinel.k,
                        sentinel.v,
                        sentinel.g,
                        sentinel.beta,
                        ssm_states=sentinel.ssm_states,
                        cache_indices=sentinel.cache_indices,
                        query_start_loc=sentinel.query_start_loc,
                        A_log=sentinel.A_log,
                        dt_bias=sentinel.dt_bias,
                        lower_bound=lower_bound,
                        beta_is_raw=True,
                        is_spec_decode=is_spec_decode,
                        return_intermediate_states=track_ssm,
                        track_state=track_state,
                        track_chunk_idx=track_chunk_idx,
                    )
                self.assertIs(actual, result)
                fused_extend.assert_not_called()
                chunk_kda.assert_called_once_with(
                    q=sentinel.q,
                    k=sentinel.k,
                    v=sentinel.v,
                    g=sentinel.g,
                    beta=sentinel.beta,
                    initial_state=sentinel.ssm_states,
                    initial_state_indices=sentinel.cache_indices,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=sentinel.query_start_loc,
                    A_log=sentinel.A_log,
                    dt_bias=sentinel.dt_bias,
                    lower_bound=lower_bound,
                    beta_is_raw=True,
                    output_intermediate_states=track_ssm,
                    track_state=track_state,
                    track_chunk_idx=track_chunk_idx,
                )

    def test_flashkda_ignores_the_cuda_sequence_length_window(self):
        kernel = sail_flashkda.SailFlashKDAKernel()
        for tokens in (48, 4096):
            with self.subTest(tokens=tokens):
                query_start_loc = torch.tensor([0, tokens], dtype=torch.int32)
                with envs.SGLANG_SAIL_PLA.override(True):
                    self.assertFalse(
                        kernel._should_fall_back(-5.0, False, query_start_loc, [tokens])
                    )
                    self.assertTrue(
                        kernel._should_fall_back(None, False, query_start_loc, [tokens])
                    )
                    self.assertTrue(
                        kernel._should_fall_back(-5.0, True, query_start_loc, [tokens])
                    )
                with envs.SGLANG_SAIL_PLA.override(False):
                    self.assertTrue(
                        kernel._should_fall_back(-5.0, False, query_start_loc, [tokens])
                    )

    def test_fused_decode_coverage_covers_pla_head_counts(self):
        with envs.SGLANG_SAIL_PLA.override(True):
            for num_heads in (12, 24, 48, 96):
                self.assertTrue(
                    sail_kda.fused_decode_covered(
                        *_fused_decode_coverage_inputs(num_heads)
                    )
                )
            self.assertFalse(
                sail_kda.fused_decode_covered(*_fused_decode_coverage_inputs(6))
            )
        with envs.SGLANG_SAIL_PLA.override(False):
            self.assertFalse(
                sail_kda.fused_decode_covered(*_fused_decode_coverage_inputs(24))
            )
        # The community contract is unchanged by the SAIL overlay.
        self.assertFalse(kda_fused_decode.covered(*_fused_decode_coverage_inputs(24)))
        self.assertTrue(kda_fused_decode.covered(*_fused_decode_coverage_inputs(6)))

    def test_fused_decode_skips_graph_padding_rows(self):
        num_heads, head_dim = 24, 128
        seg = num_heads * head_dim
        mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate = (
            _fused_decode_coverage_inputs(num_heads)
        )
        cache_indices[1] = -1
        pla_forward = Mock(side_effect=lambda **kwargs: kwargs["out"].fill_(0.25))
        pla_module = ModuleType("pla")
        decode_module = ModuleType("pla.decode")
        kda_module = ModuleType("pla.decode.kda")
        pla_module.decode = decode_module
        decode_module.kda = kda_module
        kda_module.fused_kda_decode_mega_forward = pla_forward
        with (
            envs.SGLANG_SAIL_PLA.override(True),
            patch.dict(
                sys.modules,
                {
                    "pla": pla_module,
                    "pla.decode": decode_module,
                    "pla.decode.kda": kda_module,
                },
            ),
        ):
            output = sail_kda.fused_decode(
                mixed_qkv,
                a,
                b,
                conv_states,
                torch.empty(3 * seg, dtype=torch.float32),
                torch.empty(num_heads, dtype=torch.float32),
                torch.empty(seg, dtype=torch.float32),
                onorm_gate,
                torch.empty(head_dim, dtype=torch.float32),
                ssm_states,
                cache_indices,
                onorm_eps=1e-5,
                fused_weight=torch.empty(3, 4, seg, dtype=torch.float32),
                actual_batch_size=1,
                lower_bound=-5.0,
            )
        self.assertEqual(output.shape, (1, 2, num_heads, head_dim))
        self.assertTrue(torch.all(output[:, :1] == 0.25))
        self.assertEqual(torch.count_nonzero(output[:, 1:]).item(), 0)
        pla_forward.assert_called_once()
        kwargs = pla_forward.call_args.kwargs
        self.assertEqual(kwargs["x"].shape, (1, 3 * seg))
        self.assertEqual(kwargs["state_indices"].tolist(), [0])
        self.assertEqual(kwargs["out"].shape, (1, 1, num_heads, head_dim))

    def test_packed_decode_returns_the_decode_output_layout(self):
        batch_size, num_heads, head_v_dim = 3, 24, 128
        seg = num_heads * head_v_dim

        # The operator fills the caller's [B, 1, HV, V] buffer in place.
        def fused_packed(*args):
            out = args[10]
            out.fill_(0.5)
            return out

        for kernel_class in (
            sail_kda_backend.SailTritonKDAKernel,
            sail_gdn_triton.SailTritonGDNKernel,
        ):
            with self.subTest(kernel=kernel_class.__name__):
                with (
                    envs.SGLANG_SAIL_PLA.override(True),
                    patch.dict(sys.modules, _pla_decode_modules(fused_packed)),
                ):
                    output = kernel_class().packed_decode(
                        torch.zeros(batch_size, 3 * seg, dtype=torch.bfloat16),
                        torch.zeros(batch_size, seg, dtype=torch.bfloat16),
                        torch.zeros(batch_size, num_heads, dtype=torch.bfloat16),
                        A_log=torch.zeros(1, num_heads, dtype=torch.float32),
                        dt_bias=torch.zeros(seg, dtype=torch.float32),
                        scale=head_v_dim**-0.5,
                        ssm_states=torch.zeros(
                            2, num_heads, head_v_dim, head_v_dim, dtype=torch.float32
                        ),
                        cache_indices=torch.zeros(batch_size, dtype=torch.int32),
                        num_v_heads=num_heads,
                        head_v_dim=head_v_dim,
                    )
                # Packed decode hands back [1, B, HV, V], not the [B, 1, ...]
                # buffer the operator fills.
                self.assertEqual(output.shape, (1, batch_size, num_heads, head_v_dim))
                self.assertTrue(torch.all(output == 0.5))

    def test_gdn_dispatcher_swaps_every_triton_slot(self):
        dispatcher = sail_gdn_backend.SailGDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )
        sail_kernel = dispatcher.decode_kernel
        self.assertIsInstance(sail_kernel, sail_gdn_triton.SailTritonGDNKernel)
        self.assertIs(dispatcher.extend_kernel, sail_kernel)
        self.assertIs(dispatcher.verify_kernel, sail_kernel)
        self.assertIs(dispatcher.tree_verify_kernel, sail_kernel)
        self.assertEqual(
            dispatcher.supports_packed_decode,
            base_gdn_triton.TritonGDNKernel.supports_packed_decode,
        )

    def test_gdn_kernel_falls_back_without_the_pla_switch(self):
        kernel = sail_gdn_triton.SailTritonGDNKernel()
        with (
            envs.SGLANG_SAIL_PLA.override(False),
            patch.object(
                base_gdn_triton,
                "fused_sigmoid_gating_delta_rule_update",
                return_value=sentinel.community_output,
            ) as community_update,
        ):
            output = kernel.decode(
                sentinel.q,
                sentinel.k,
                sentinel.v,
                sentinel.a,
                sentinel.b,
                A_log=sentinel.A_log,
                dt_bias=sentinel.dt_bias,
                ssm_states=sentinel.ssm_states,
                cache_indices=sentinel.cache_indices,
                query_start_loc=sentinel.query_start_loc,
            )
        self.assertIs(output, sentinel.community_output)
        community_update.assert_called_once()

    def test_prepares_fused_decode_handoff_for_k3_tp_layouts(self):
        with (
            patch.object(kimi_k3, "is_sail", return_value=True),
            patch.object(kimi_k3, "_is_hip", False),
            patch.object(kimi_k3, "_is_npu", False),
            envs.SGLANG_SAIL_PLA.override(True),
        ):
            for num_heads in (12, 24, 48, 96):
                with self.subTest(num_heads=num_heads):
                    attention, layer = _fused_decode_attention(num_heads)
                    kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
                    self.assertTrue(attention._kda_fused_decode_ready)
                    self.assertEqual(
                        layer._k3_fused_decode_weight.shape, (3, 4, num_heads * 128)
                    )

    def test_community_preparation_keeps_its_head_limit(self):
        with (
            patch.object(kimi_k3, "is_sail", return_value=False),
            patch.object(kimi_k3, "_is_hip", False),
            patch.object(kimi_k3, "_is_npu", False),
            patch.object(kimi_k3, "rank0_log"),
        ):
            attention, layer = _fused_decode_attention(24)
            kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
            self.assertFalse(attention._kda_fused_decode_ready)
            self.assertFalse(hasattr(layer, "_k3_fused_decode_args"))
            attention, layer = _fused_decode_attention(12)
            kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
            self.assertTrue(attention._kda_fused_decode_ready)
            self.assertFalse(hasattr(layer, "_k3_fused_decode_weight"))


if __name__ == "__main__":
    unittest.main()
