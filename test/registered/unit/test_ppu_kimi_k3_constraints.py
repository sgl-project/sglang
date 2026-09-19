"""CPU checks for PPU-only Kimi-K3 backend restrictions."""

from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.jit.utils import arch as jit_arch
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16DeepGemmMoE,
)


def test_kda_dispatcher_forces_every_mode_to_triton_on_ppu():
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
