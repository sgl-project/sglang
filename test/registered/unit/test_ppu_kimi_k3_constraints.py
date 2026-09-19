"""CPU checks for PPU-only Kimi-K3 backend restrictions."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.jit.utils import arch as jit_arch
from sglang.kernels.ops.attention import kda_fused_decode
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16DeepGemmMoE,
)
from sglang.srt.models import kimi_k3


def test_kda_dispatcher_forces_unsupported_backends_to_triton_on_ppu():
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    with (
        patch.object(kda_backend, "is_ppu", return_value=True),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.FLASHINFER,
            LinearAttnKernelBackend.NVIDIA_KDA,
            LinearAttnKernelBackend.NV_CUTEDSL,
        )

    assert dispatcher.decode_kernel is triton_kernel
    assert dispatcher.extend_kernel is triton_kernel
    assert dispatcher.verify_kernel is triton_kernel


def test_kda_dispatcher_honors_flashkda_prefill_backend_on_ppu():
    triton_kernel = type("TritonKernel", (), {"supports_packed_decode": False})()
    flashkda_kernel = object()
    with (
        patch.object(kda_backend, "is_ppu", return_value=True),
        patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
        patch.object(
            kda_flashkda, "FlashKDAKernel", return_value=flashkda_kernel
        ) as kernel_class,
        patch.object(kda_backend, "rank0_log"),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.FLASHKDA,
            LinearAttnKernelBackend.TRITON,
        )

    assert dispatcher.decode_kernel is triton_kernel
    assert dispatcher.extend_kernel is flashkda_kernel
    assert dispatcher.verify_kernel is triton_kernel
    kernel_class.assert_called_once_with()


def test_ppu_flashkda_dispatches_to_platform_operator():
    num_heads = 12
    head_dim = 128
    packed_tokens = 64
    q = torch.zeros((1, packed_tokens, num_heads, head_dim), dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros_like(q)
    beta = torch.full((1, packed_tokens, num_heads), 0.5, dtype=torch.float32)
    ssm_states = torch.zeros((2, num_heads, head_dim, head_dim), dtype=torch.float32)
    cache_indices = torch.tensor([1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, packed_tokens], dtype=torch.int32)
    flashkda_fwd = Mock(
        side_effect=lambda **kwargs: (
            kwargs["out"].zero_(),
            kwargs["final_state"].copy_(kwargs["initial_state"]),
        )
    )
    pla_module = ModuleType("pla")
    prefill_module = ModuleType("pla.prefill")
    flashkda_module = ModuleType("pla.prefill.flashkdapro")
    pla_module.prefill = prefill_module
    prefill_module.flashkdapro = flashkda_module
    flashkda_module.flashkda_fwd = flashkda_fwd

    with (
        patch.object(kda_flashkda, "is_ppu", return_value=True),
        patch.object(kda_flashkda, "_load_flash_kda") as load_flash_kda,
        patch.dict(
            sys.modules,
            {
                "pla": pla_module,
                "pla.prefill": prefill_module,
                "pla.prefill.flashkdapro": flashkda_module,
            },
        ),
    ):
        output = kda_flashkda.FlashKDAKernel().extend(
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
        )

    assert output.shape == q.shape
    load_flash_kda.assert_not_called()
    flashkda_fwd.assert_called_once()
    kwargs = flashkda_fwd.call_args.kwargs
    assert kwargs["q"].shape == (packed_tokens, num_heads, head_dim)
    assert kwargs["beta"].shape == (packed_tokens, num_heads)
    assert kwargs["beta"].dtype == torch.bfloat16
    assert kwargs["A_log"].shape == (num_heads,)
    assert kwargs["dt_bias"].shape == (num_heads * head_dim,)
    assert kwargs["cu_seqlens"].dtype == torch.int64


def test_ppu_flashkda_only_bypasses_cuda_sequence_length_fallback():
    kernel = kda_flashkda.FlashKDAKernel()
    short_query_start_loc = torch.tensor([0, 48], dtype=torch.int32)
    long_query_start_loc = torch.tensor([0, 4096], dtype=torch.int32)

    with patch.object(kda_flashkda, "is_ppu", return_value=True):
        assert not kernel._should_fall_back(-5.0, False, short_query_start_loc, [48])
        assert not kernel._should_fall_back(-5.0, False, long_query_start_loc, [4096])
        assert kernel._should_fall_back(None, False, short_query_start_loc, [48])
        assert kernel._should_fall_back(-5.0, True, short_query_start_loc, [48])

    with patch.object(kda_flashkda, "is_ppu", return_value=False):
        assert kernel._should_fall_back(-5.0, False, short_query_start_loc, [48])
        assert kernel._should_fall_back(-5.0, False, long_query_start_loc, [4096])


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


def test_ppu_fused_decode_bypasses_community_coverage():
    with (
        patch.object(kda_fused_decode, "is_ppu", return_value=True),
        patch.object(kda_fused_decode, "_SUPPORTED_HEADS", {12, 24, 48, 96}),
    ):
        for num_heads in (12, 24, 48, 96):
            assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(num_heads))
        assert not kda_fused_decode.covered(*_fused_decode_coverage_inputs(6))

    with (
        patch.object(kda_fused_decode, "is_ppu", return_value=False),
        patch.object(kda_fused_decode, "_SUPPORTED_HEADS", {3, 6, 12}),
    ):
        assert kda_fused_decode.covered(*_fused_decode_coverage_inputs(6))
        assert not kda_fused_decode.covered(*_fused_decode_coverage_inputs(24))


def test_ppu_fused_decode_dispatches_tp4_without_graph_padding_rows():
    num_heads = 24
    head_dim = 128
    seg = num_heads * head_dim
    mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate = (
        _fused_decode_coverage_inputs(num_heads)
    )
    cache_indices[1] = -1
    pla_forward = Mock(side_effect=lambda **kwargs: kwargs["out"].zero_())
    pla_module = ModuleType("pla")
    decode_module = ModuleType("pla.decode")
    kda_module = ModuleType("pla.decode.kda")
    pla_module.decode = decode_module
    decode_module.kda = kda_module
    kda_module.fused_kda_decode_mega_forward = pla_forward

    with (
        patch.object(kda_fused_decode, "is_ppu", return_value=True),
        patch.dict(
            sys.modules,
            {
                "pla": pla_module,
                "pla.decode": decode_module,
                "pla.decode.kda": kda_module,
            },
        ),
    ):
        output = kda_fused_decode.kda_fused_decode(
            mixed_qkv,
            a,
            b,
            conv_states,
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(4, seg, dtype=torch.float32),
            torch.empty(3 * seg, dtype=torch.float32),
            torch.empty(num_heads, dtype=torch.float32),
            torch.empty(seg, dtype=torch.float32),
            onorm_gate,
            torch.empty(head_dim, dtype=torch.float32),
            ssm_states,
            cache_indices,
            scale=head_dim**-0.5,
            onorm_eps=1e-5,
            lower_bound=-5.0,
            fused_weight=torch.empty(3, 4, seg, dtype=torch.float32),
            actual_batch_size=1,
        )

    assert output.shape == (1, 2, num_heads, head_dim)
    assert torch.count_nonzero(output[:, 1:]).item() == 0
    pla_forward.assert_called_once()
    kwargs = pla_forward.call_args.kwargs
    assert kwargs["x"].shape == (1, 3 * seg)
    assert kwargs["state_indices"].tolist() == [0]
    assert kwargs["out"].shape == (1, 1, num_heads, head_dim)


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


def test_ppu_prepares_fused_decode_handoff_for_k3_tp_layouts():
    with (
        patch.object(kimi_k3, "is_ppu", return_value=True),
        patch.object(kimi_k3, "_is_hip", False),
    ):
        for num_heads in (12, 24, 48, 96):
            attention, layer = _fused_decode_attention(num_heads)
            kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)

            assert attention._kda_fused_decode_ready
            assert layer._k3_fused_decode_weight.shape == (
                3,
                4,
                num_heads * 128,
            )


def test_non_ppu_keeps_community_fused_decode_preparation_limit():
    with (
        patch.object(kimi_k3, "is_ppu", return_value=False),
        patch.object(kimi_k3, "_is_hip", False),
        patch.object(kimi_k3, "rank0_log"),
    ):
        attention, layer = _fused_decode_attention(24)
        kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
        assert not attention._kda_fused_decode_ready
        assert not hasattr(layer, "_k3_fused_decode_args")

        attention, layer = _fused_decode_attention(12)
        kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(attention)
        assert attention._kda_fused_decode_ready
        assert not hasattr(layer, "_k3_fused_decode_weight")


def test_ppu_disables_pdl_and_megamoe():
    with patch.object(jit_arch, "is_ppu_runtime", return_value=True):
        assert not jit_arch.is_arch_support_pdl.__wrapped__()

    with patch.object(overrides_module, "is_ppu", return_value=True):
        try:
            overrides_module._a2a_backend_overrides(
                type("View", (), {"moe_a2a_backend": "megamoe"})()
            )
        except ValueError as exc:
            assert "MegaMoE is not supported on PPU" in str(exc)
        else:
            raise AssertionError("PPU MegaMoE backend must be rejected")


def test_ppu_w4a16_deep_gemm_moe_accepts_k3_situ():
    scheme = object.__new__(CompressedTensorsWNA16DeepGemmMoE)
    scheme.moe_runner_config = SimpleNamespace(activation="situ")
    scheme.runner = Mock()
    scheme.runner.run.return_value = sentinel.combine_input
    scheme.get_deep_gemm_quant_info = Mock(return_value=sentinel.quant_info)

    result = scheme.apply_weights(sentinel.layer, sentinel.dispatch_output)

    assert result is sentinel.combine_input
    scheme.get_deep_gemm_quant_info.assert_called_once_with(sentinel.layer)
    scheme.runner.run.assert_called_once_with(
        sentinel.dispatch_output, sentinel.quant_info
    )
