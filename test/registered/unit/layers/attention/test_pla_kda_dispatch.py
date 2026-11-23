"""PLA dispatch for the SAIL KDA/GDN backends: kernel slots, layout, fallback.

The PLA operators are out-of-tree; these cases pin which kernel slot serves
which mode, what the operators are handed, and when a batch stays on the
community Triton implementation.
"""

import builtins
import inspect
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.kernels.ops.attention.linear import kda_replayssm_spec_decode as replayssm
from sglang.kernels.ops.mamba import mamba_state_scatter_triton as scatter
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.sail.attention import gdn_backend as sail_gdn_backend
from sglang.srt.hardware_backend.sail.attention import gdn_triton as sail_gdn_triton
from sglang.srt.hardware_backend.sail.attention import kda_backend as sail_kda_backend
from sglang.srt.hardware_backend.sail.attention import kda_flashkda as sail_flashkda
from sglang.srt.hardware_backend.sail.kernels import kda as sail_kda
from sglang.srt.layers.attention.linear.kernels import gdn_triton as base_gdn_triton
from sglang.srt.layers.attention.linear.kernels import kda_flashkda, kda_triton
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnBackends,
    LinearAttnKernelBackend,
)
from sglang.srt.models import kimi_k3
from sglang.srt.platforms.interface import SRTPlatform
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
        _k3_fused_decode_args=None,
    )
    attention = SimpleNamespace(
        attn=layer,
        local_num_heads=num_heads,
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

    def test_kda_backend_checks_see_the_bound_kernels(self):
        sail_backend = sail_kda_backend.SailKDAAttnBackend
        for prefill, expected_prefill in (
            (LinearAttnKernelBackend.FLASHINFER, LinearAttnKernelBackend.TRITON),
            (LinearAttnKernelBackend.FLASHKDA, LinearAttnKernelBackend.FLASHKDA),
        ):
            with self.subTest(prefill=prefill):
                requested = LinearAttnBackends(
                    decode=LinearAttnKernelBackend.FLASHINFER,
                    prefill=prefill,
                    verify=LinearAttnKernelBackend.FLASHINFER,
                )
                effective = sail_backend._effective_backends(requested)
                self.assertEqual(effective.decode, LinearAttnKernelBackend.TRITON)
                self.assertEqual(effective.prefill, expected_prefill)
                self.assertEqual(effective.verify, LinearAttnKernelBackend.TRITON)
                # Triton verify does not seed from the accepted checkpoint, so
                # the post-verify SSM commit scatter must stay on.
                self.assertFalse(sail_backend._can_fuse_accept_state(effective.verify))
                dispatcher = sail_backend._create_kernel_dispatcher(None, effective)
                self.assertEqual(dispatcher.prefill_backend, effective.prefill)

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
            # The PLA operator keeps the community layout and dtype contract.
            inputs = list(_fused_decode_coverage_inputs(24))
            inputs[4] = inputs[4].to(torch.bfloat16)
            self.assertFalse(sail_kda.fused_decode_covered(*inputs))
            inputs = list(_fused_decode_coverage_inputs(24))
            inputs[5] = inputs[5].to(torch.int64)
            self.assertFalse(sail_kda.fused_decode_covered(*inputs))
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
                    weight = layer._k3_fused_decode_weight
                    self.assertEqual(weight.shape, (3, 4, num_heads * 128))
                    # A weight reload refreshes the buffer a captured graph reads.
                    fused_args = layer._k3_fused_decode_args
                    layer.conv_weights.fill_(0.75)
                    kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
                    self.assertIs(layer._k3_fused_decode_weight, weight)
                    self.assertIs(layer._k3_fused_decode_args, fused_args)
                    self.assertTrue(torch.all(weight == 0.75))

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
            self.assertIsNone(layer._k3_fused_decode_args)
            attention, layer = _fused_decode_attention(12)
            kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
            self.assertTrue(attention._kda_fused_decode_ready)
            self.assertFalse(hasattr(layer, "_k3_fused_decode_weight"))

    def test_mtp_override_preserves_the_parent_interface_and_ring_fallback(self):
        parent = sail_kda_backend.KDAAttnBackend
        override = sail_kda_backend.SailKDAAttnBackend._run_dspark_cutedsl_mtp
        signature = inspect.signature(parent._run_dspark_cutedsl_mtp)
        self.assertEqual(signature, inspect.signature(override, eval_str=True))
        kwargs = {name: getattr(sentinel, name) for name in signature.parameters}
        for enabled in (False, True):
            for ring in (None, sentinel.ring):
                with self.subTest(enabled=enabled, ring=ring):
                    kwargs["replayssm_ring_indices"] = ring
                    with (
                        envs.SGLANG_SAIL_PLA.override(enabled),
                        patch.object(parent, "_run_dspark_cutedsl_mtp") as native,
                        patch.object(sail_kda, "decode_mtp") as pla,
                    ):
                        actual = override(**kwargs)
                    if not enabled or ring is not None:
                        self.assertIs(actual, native.return_value)
                        native.assert_called_once_with(**kwargs)
                        pla.assert_not_called()
                    else:
                        self.assertIs(actual, pla.return_value)
                        pla.assert_called_once_with(
                            **{
                                k: v
                                for k, v in kwargs.items()
                                if k != "replayssm_ring_indices"
                            }
                        )
                        native.assert_not_called()

    def test_tracking_schedule_is_platform_specific(self):
        for source in (None, torch.empty(0), torch.tensor([1])):
            backend = SimpleNamespace(
                forward_metadata=SimpleNamespace(track_ssm_h_src=source)
            )
            for track in (False, True):
                self.assertEqual(
                    sail_kda_backend.KDAAttnBackend._needs_intermediate_states(
                        backend, track
                    ),
                    track,
                )
                self.assertEqual(
                    sail_kda_backend.SailKDAAttnBackend._needs_intermediate_states(
                        backend, track
                    ),
                    track and source is not None and source.numel() > 0,
                )

    def test_sail_prefill_default_is_per_runner_and_explicit_flags_win(self):
        from sglang.srt.layers.attention.linear.utils import (
            resolve_linear_attn_backends,
        )

        for enabled in (False, True):
            for kda in (False, True):
                runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=Mock()))
                with (
                    envs.SGLANG_SAIL_PLA.override(enabled),
                    patch(
                        "sglang.srt.configs.model_config.uses_kda_attention",
                        return_value=kda,
                    ),
                ):
                    default = sail_kda_backend.sail_pla_prefill_default(
                        runner.model_config
                    )
                self.assertEqual(default, "flashkda" if enabled and kda else None)
                for explicit in (None, "triton"):
                    mamba = SimpleNamespace(
                        linear_attn_backend="triton",
                        linear_attn_decode_backend=None,
                        linear_attn_prefill_backend=explicit,
                        linear_attn_verify_backend=None,
                    )
                    with patch(
                        "sglang.srt.layers.attention.linear.utils.get_exec",
                        return_value=SimpleNamespace(mamba=mamba),
                    ):
                        actual = resolve_linear_attn_backends(prefill_default=default)
                    self.assertEqual(
                        actual.prefill.value, explicit or default or "triton"
                    )
                    self.assertIsNone(mamba.linear_attn_decode_backend)
                    self.assertEqual(mamba.linear_attn_prefill_backend, explicit)

    def test_sail_pla_configuration_respects_opt_out_and_missing_wheel(self):
        with (
            envs.SGLANG_SAIL_PLA.override(False),
            patch.dict(sys.modules, {"pla": ModuleType("pla")}),
        ):
            sail_kda.configure_pla()
            self.assertFalse(envs.SGLANG_SAIL_PLA.get())
            envs.SGLANG_SAIL_PLA.clear()
            sail_kda.configure_pla()
            self.assertTrue(envs.SGLANG_SAIL_PLA.get())
        with (
            envs.SGLANG_SAIL_PLA.override(True),
            patch.dict(sys.modules, {"pla": None}),
        ):
            sail_kda.configure_pla()
            self.assertFalse(envs.SGLANG_SAIL_PLA.get())

    def test_scatter_dispatch_preserves_imported_consumers(self):
        for name in (
            "fused_mamba_state_scatter_with_mask",
            "fused_conv_window_scatter_with_mask",
        ):
            consumer = getattr(scatter, name)
            vendor = Mock(return_value=sentinel.vendor)
            fake_module = ModuleType(
                "sglang.kernels.ops.mamba.mamba_state_scatter_sail"
            )
            setattr(fake_module, name, vendor)
            with (
                patch.object(scatter, "is_sail_runtime", return_value=False),
                patch.dict(sys.modules, {fake_module.__name__: None}),
            ):
                consumer(
                    sentinel.dst,
                    sentinel.src,
                    sentinel.dst_indices,
                    torch.empty(0, dtype=torch.int32),
                )
                vendor.assert_not_called()
            with (
                patch.object(scatter, "is_sail_runtime", return_value=True),
                patch.dict(sys.modules, {fake_module.__name__: fake_module}),
            ):
                self.assertIs(
                    consumer(
                        sentinel.dst,
                        sentinel.src,
                        sentinel.dst_indices,
                        sentinel.steps,
                        sentinel.src_indices,
                    ),
                    sentinel.vendor,
                )
            vendor.assert_called_once_with(
                sentinel.dst,
                sentinel.src,
                sentinel.dst_indices,
                sentinel.steps,
                sentinel.src_indices,
            )

    def test_sail_backend_selection_needs_no_registration(self):
        from sglang.srt.layers.attention import attention_registry

        with (
            patch.object(attention_registry, "_is_sail", True),
            patch.dict(
                attention_registry.OOT_LINEAR_ATTENTION_BACKENDS, {}, clear=True
            ),
            patch("sglang.srt.platforms.current_platform", SRTPlatform()),
        ):
            for base, module, name in (
                (
                    sail_kda_backend.KDAAttnBackend,
                    sail_kda_backend,
                    "SailKDAAttnBackend",
                ),
                (
                    sail_gdn_backend.GDNAttnBackend,
                    sail_gdn_backend,
                    "SailGDNAttnBackend",
                ),
            ):
                with patch.object(module, name) as factory:
                    actual = attention_registry._create_linear_attention_backend(
                        sentinel.runner, base
                    )
                    self.assertIs(actual, factory.return_value)
                    factory.assert_called_once_with(sentinel.runner)
            self.assertFalse(attention_registry.OOT_LINEAR_ATTENTION_BACKENDS)

        class CustomKDA(sail_kda_backend.KDAAttnBackend):
            pass

        class CustomGDN(sail_gdn_backend.GDNAttnBackend):
            pass

        for backend in (CustomKDA, CustomGDN):
            self.assertIs(
                sail_kda_backend.get_linear_attention_backend(backend), backend
            )

    def test_cuda_backend_selection_does_not_import_sail_or_pla(self):
        from sglang.srt.layers.attention import attention_registry

        original_import = builtins.__import__

        def import_without_sail(name, *args, **kwargs):
            if name.startswith("sglang.srt.hardware_backend.sail") or name == "pla":
                raise AssertionError(f"Unexpected CUDA import: {name}")
            return original_import(name, *args, **kwargs)

        with (
            envs.SGLANG_SAIL_PLA.override(True),
            patch.object(attention_registry, "_is_sail", False),
            patch("sglang.srt.platforms.current_platform", SRTPlatform()),
            patch("builtins.__import__", side_effect=import_without_sail),
        ):
            for backend in (
                sail_kda_backend.KDAAttnBackend,
                sail_gdn_backend.GDNAttnBackend,
            ):
                with patch.object(
                    backend, "__init__", return_value=None
                ) as constructor:
                    actual = attention_registry._create_linear_attention_backend(
                        sentinel.runner, backend
                    )
                self.assertIsInstance(actual, backend)
                constructor.assert_called_once_with(sentinel.runner)

    def test_sail_commit_and_native_fallback_preserve_all_indices(self):
        kwargs = dict(
            spec_state=SimpleNamespace(
                temporal=sentinel.temporal,
                replayssm_rawv=torch.empty(1, 2, 1, 4, 128),
                replayssm_rawk=torch.empty(1, 2, 1, 4, 128),
                replayssm_g=sentinel.g,
                replayssm_beta=sentinel.beta,
                conv=[sentinel.conv0, sentinel.conv1],
                intermediate_conv_window=[sentinel.window0, sentinel.window1],
            ),
            state_batch_indices=sentinel.slots,
            accept_lens=sentinel.accept_lens,
            last_correct_step_indices=sentinel.last_step,
            mamba_track_indices=sentinel.track_slots,
            mamba_steps_to_track=sentinel.track_steps,
            ring_indices=sentinel.ring,
            src_indices_raw=sentinel.source,
            null_block_id=7,
        )
        for covered in (True, False):
            with (
                patch.object(
                    sail_kda, "commit_after_verify", return_value=covered
                ) as pla,
                patch.object(replayssm, "commit_kda_replayssm_spec_all_layers") as fold,
                patch.object(
                    scatter, "fused_conv_window_scatter_with_mask"
                ) as conv_scatter,
            ):
                sail_kda.commit_after_verify_or_fallback(**kwargs)
            pla.assert_called_once_with(**kwargs)
            if covered:
                fold.assert_not_called()
                conv_scatter.assert_not_called()
            else:
                fold.assert_called_once()
                self.assertIs(fold.call_args.kwargs["ring_indices"], sentinel.ring)
                self.assertIs(
                    fold.call_args.kwargs["mamba_track_indices"], sentinel.track_slots
                )
                self.assertIs(
                    fold.call_args.kwargs["accept_lens"], sentinel.accept_lens
                )
                self.assertEqual(conv_scatter.call_count, 4)
                for i in range(2):
                    for offset, slots, steps in (
                        (0, sentinel.slots, sentinel.last_step),
                        (1, sentinel.track_slots, sentinel.track_steps),
                    ):
                        self.assertEqual(
                            conv_scatter.call_args_list[2 * i + offset].args,
                            (
                                kwargs["spec_state"].conv[i],
                                kwargs["spec_state"].intermediate_conv_window[i],
                                slots,
                                steps,
                                sentinel.source,
                            ),
                        )

    def test_both_commit_callers_keep_cuda_and_sail_separate(self):
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid
        from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardStateSlots
        from sglang.srt.speculative import spec_utils

        pool = SimpleNamespace(
            replayssm_is_kda=True,
            replayssm_spec_fold=True,
            replayssm_request_ring=True,
            replayssm_cache_base=None,
            get_speculative_mamba2_params_all_layers=lambda: sentinel.state,
        )
        backend = hybrid.HybridLinearAttnBackend.__new__(hybrid.HybridLinearAttnBackend)
        backend.linear_attn_backend = SimpleNamespace(state_pool=pool)
        slots = ForwardStateSlots(
            torch.tensor([3, 4]), torch.tensor([8, 9]), torch.tensor([1, 2])
        )
        steps, track_steps = torch.tensor([1, 0]), torch.tensor([0, -1])
        request_rows = torch.tensor([11, 12])
        target_worker = Mock()
        target_worker.model_runner.token_to_kv_pool.layer_router.part_view.return_value = pool
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            req_pool_indices=request_rows,
            mamba_track_indices=slots.track,
        )
        for sail in (False, True):
            for direct in (False, True):
                with (
                    self.subTest(sail=sail, direct=direct),
                    patch.object(hybrid, "is_sail", return_value=sail),
                    patch.object(spec_utils, "is_sail", return_value=sail),
                    patch.object(
                        hybrid, "pp_spec_stable_rows_enabled", return_value=True
                    ),
                    patch.object(
                        spec_utils, "pp_spec_stable_rows_enabled", return_value=True
                    ),
                    patch.object(spec_utils, "mambaish_config", return_value=True),
                    patch.object(spec_utils, "built_megakernel", return_value=None),
                    patch.object(
                        replayssm, "commit_kda_replayssm_after_verify"
                    ) as native,
                    patch.object(sail_kda, "commit_after_verify_or_fallback") as pla,
                ):
                    if direct:
                        backend.update_mamba_state_after_mtp_verify(
                            steps,
                            slots.track,
                            track_steps,
                            None,
                            req_pool_indices=request_rows,
                            state_slots=slots,
                        )
                    else:
                        spec_utils.commit_mamba_states_after_verify(
                            target_worker,
                            batch,
                            steps + 1,
                            torch.tensor([[0, 1], [2, -1]]),
                            2,
                            prepared_step_indices=(steps, track_steps),
                            state_slots=slots,
                        )
                selected, unused = (pla, native) if sail else (native, pla)
                selected.assert_called_once()
                unused.assert_not_called()
                kwargs = selected.call_args.kwargs
                self.assertIs(kwargs["spec_state"], sentinel.state)
                for name, expected in (
                    ("state_batch_indices", slots.working),
                    ("mamba_track_indices", slots.track),
                    ("ring_indices", slots.ring),
                    ("src_indices_raw", request_rows),
                    ("accept_lens", steps + 1),
                    ("mamba_steps_to_track", track_steps),
                ):
                    torch.testing.assert_close(kwargs[name], expected)

    def test_sail_commit_rejects_uncovered_indexing_before_mutation(self):
        kwargs = dict(
            spec_state=sentinel.spec_state,
            state_batch_indices=sentinel.slots,
            accept_lens=sentinel.accept_lens,
            last_correct_step_indices=sentinel.last_step,
        )
        with envs.SGLANG_SAIL_PLA.override(True):
            for extra in (
                {"ring_indices": sentinel.ring},
                {"src_indices_raw": sentinel.source},
            ):
                self.assertFalse(sail_kda.commit_after_verify(**kwargs, **extra))
        with envs.SGLANG_SAIL_PLA.override(False):
            self.assertFalse(sail_kda.commit_after_verify(**kwargs))


if __name__ == "__main__":
    unittest.main()
