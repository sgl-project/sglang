# Adapted from https://github.com/vllm-project/vllm/blob/8ca7a71df787ad711ad3ac70a5bd2eb2bb398938/tests/quantization/test_fp8.py

import pytest
import torch

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.utils import is_cuda


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant_per_tensor(dtype) -> None:

    def quantize_ref_per_tensor(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(torch.float8_e4m3fn)
        return qweight

    def dequantize_per_tensor(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight

    # Note that we use a shape % 8 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 8.
    x = (torch.randn(size=(11, 11), device="cuda") * 13).to(dtype)

    # Test Per Tensor Dynamic quantization
    # scale = max(abs(x)) / FP8_E4M3_MAX
    y, scale = scaled_fp8_quant(x, None)
    ref_y = quantize_ref_per_tensor(x, scale)
    torch.testing.assert_close(y, ref_y)
    torch.testing.assert_close(
        dequantize_per_tensor(y, scale, dtype),
        dequantize_per_tensor(ref_y, scale, dtype),
    )

    # Test Per Tensor Static quantization
    y, _ = scaled_fp8_quant(x, scale)
    ref_y = quantize_ref_per_tensor(x, scale)
    torch.testing.assert_close(y, ref_y)
    torch.testing.assert_close(
        dequantize_per_tensor(y, scale, dtype),
        dequantize_per_tensor(ref_y, scale, dtype),
    )


if is_cuda:

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_scaled_fp8_quant_per_token_dynamic(dtype) -> None:
        def quantize_ref_per_token(tensor, inv_scale):
            # The reference implementation that fully aligns to
            # the kernel being tested.
            finfo = torch.finfo(torch.float8_e4m3fn)
            scale = inv_scale.reciprocal()
            qweight = (tensor.to(torch.float32) * scale).clamp(
                min=finfo.min, max=finfo.max
            )
            qweight = qweight.to(torch.float8_e4m3fn)
            return qweight

        def dequantize_per_token(tensor, inv_scale, dtype):
            fake_qweight = tensor.to(dtype)
            dq_weight = fake_qweight * inv_scale
            return dq_weight

        # Note that we use a shape % 8 = 0,
        # because per_token_quant_fp8 is vectorized by 8 elements.
        x = (torch.randn(size=(11, 16), device="cuda") * 13).to(dtype)

        # Test Per Tensor Dynamic quantization
        # scale = max(abs(x)) / FP8_E4M3_MAX
        y, scale = scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        ref_y = quantize_ref_per_token(x, scale)
        torch.testing.assert_close(y, ref_y)
        torch.testing.assert_close(
            dequantize_per_token(y, scale, dtype),
            dequantize_per_token(ref_y, scale, dtype),
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_scaled_fp8_quant_with_padding(dtype) -> None:
        original_rows = 5
        x = (torch.randn(size=(original_rows, 16), device="cuda") * 13).to(dtype)

        padding_size = 10

        # Test with dynamic quantization
        y_dynamic, scale_dynamic = scaled_fp8_quant(
            x, None, num_token_padding=padding_size
        )

        # Verify output shape has the padded size
        assert y_dynamic.shape[0] == padding_size
        assert y_dynamic.shape[1] == x.shape[1]

        # Verify that the actual data in the non-padded region is correctly quantized
        y_without_padding, scale_without_padding = scaled_fp8_quant(x, None)
        torch.testing.assert_close(y_dynamic[:original_rows], y_without_padding)

        # Test with static quantization
        # First get a scale
        _, scale = scaled_fp8_quant(x, None)

        # Then use it for static quantization with padding
        y_static, _ = scaled_fp8_quant(x, scale, num_token_padding=padding_size)

        # Verify output shape has the padded size
        assert y_static.shape[0] == padding_size
        assert y_static.shape[1] == x.shape[1]

        # Verify that the actual data in the non-padded region is correctly quantized
        y_static_without_padding, _ = scaled_fp8_quant(x, scale)
        torch.testing.assert_close(y_static[:original_rows], y_static_without_padding)

        # Test with per-token dynamic quantization
        y_per_token, scale_per_token = scaled_fp8_quant(
            x, None, num_token_padding=padding_size, use_per_token_if_dynamic=True
        )

        # Verify output shape has the padded size
        assert y_per_token.shape[0] == padding_size
        assert y_per_token.shape[1] == x.shape[1]

        # Verify that the actual data in the non-padded region is correctly quantized
        y_per_token_without_padding, scale_per_token_without_padding = scaled_fp8_quant(
            x, None, use_per_token_if_dynamic=True
        )
        torch.testing.assert_close(
            y_per_token[:original_rows], y_per_token_without_padding
        )
        torch.testing.assert_close(
            scale_per_token[:original_rows], scale_per_token_without_padding
        )


if __name__ == "__main__":
    # Run the specific test function directly
    pytest.main([__file__])
