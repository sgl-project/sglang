"""Device-aware operator facade for Ascend NPU kernels."""

from typing import TYPE_CHECKING, Tuple

from sglang.srt.hardware_backend.npu.device_capabilities import (
    NPUFeature,
    supports_npu_feature,
)

if TYPE_CHECKING:
    import torch


def _triton_gemma_rms_norm(
    input: "torch.Tensor",
    weight: "torch.Tensor",
    residual: "torch.Tensor | None",
    eps: float,
) -> Tuple["torch.Tensor", "torch.Tensor | None"]:
    if input.numel() == 0:
        residual_sum = None if residual is None else input + residual
        return input.clone(), residual_sum

    from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm

    original_shape = input.shape
    input_2d = input.contiguous().view(-1, original_shape[-1])
    residual_2d = None
    if residual is not None:
        residual_2d = residual.contiguous().view_as(input_2d)

    norm_output, residual_sum = add_gemma_rms_norm(
        input_2d,
        weight.contiguous(),
        residual_2d,
        eps,
    )
    norm_output = norm_output.view(original_shape)
    if residual is not None:
        residual_sum = residual_sum.view(original_shape)
    return norm_output, residual_sum


class NPUDeviceOperator:
    """Expose semantic operator APIs across Ascend device families."""

    @staticmethod
    def gemma_rms_norm(
        input: "torch.Tensor",
        weight: "torch.Tensor",
        eps: float,
    ) -> "torch.Tensor":
        """Apply Gemma RMSNorm without exposing product checks to callers."""
        if input.numel() == 0:
            return input.clone()

        input = input.contiguous()
        weight = weight.contiguous()
        if supports_npu_feature(NPUFeature.NATIVE_GEMMA_RMS_NORM):
            import torch_npu

            return torch_npu.npu_gemma_rms_norm(input, weight, eps)[0]

        if supports_npu_feature(NPUFeature.TRITON_GEMMA_RMS_NORM):
            return _triton_gemma_rms_norm(input, weight, None, eps)[0]

        import torch_npu

        return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)[0]

    @staticmethod
    def add_gemma_rms_norm(
        input: "torch.Tensor",
        weight: "torch.Tensor",
        residual: "torch.Tensor",
        eps: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Add a residual and apply Gemma RMSNorm without mutating inputs."""
        if input.numel() == 0:
            return input.clone(), input + residual

        input = input.contiguous()
        weight = weight.contiguous()
        residual = residual.contiguous()
        if supports_npu_feature(NPUFeature.NATIVE_GEMMA_RMS_NORM):
            import torch_npu

            norm_output, _, residual_sum = torch_npu.npu_add_rms_norm(
                residual, input, 1.0 + weight, eps
            )
            return norm_output, residual_sum

        if supports_npu_feature(NPUFeature.TRITON_GEMMA_RMS_NORM):
            norm_output, residual_sum = _triton_gemma_rms_norm(
                input, weight, residual, eps
            )
            return norm_output, residual_sum

        import torch_npu

        residual_sum = input + residual
        norm_output = torch_npu.npu_rms_norm(residual_sum, 1.0 + weight, eps)[0]
        return norm_output, residual_sum
