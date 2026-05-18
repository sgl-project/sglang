import itertools
import unittest
from typing import Optional, Tuple

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

def act_quant_pytorch(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization (PyTorch native).
    This is a pure PyTorch implementation equivalent to the Triton kernel version.
    It performs per-block FP8 quantization along the last dimension.
    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and
            its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks used for quantization.
            Default is 128.
        scale_fmt (Optional[str], optional): If not None, scales are rounded to
            powers of 2. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # FP8 e4m3fn constants
    fp8_max = 448.0
    fp8_min = -448.0

    # Flatten all dims except last
    orig_shape = x.shape
    N = x.size(-1)
    x_flat = x.view(-1, N).float()  # (M, N)
    M = x_flat.size(0)
    num_groups = N // block_size

    # Reshape into blocks: (M, num_groups, block_size)
    x_blocked = x_flat.view(M, num_groups, block_size)

    # Compute per-block absolute max -> (M, num_groups)
    amax = x_blocked.abs().amax(dim=2)

    # Clamp to avoid division by zero
    amax = amax.clamp(min=1e-4)

    # Compute scale
    round_scale = scale_fmt is not None
    if round_scale:
        # Round scale to nearest power of 2 (ceiling in log2 space)
        scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    else:
        scale = amax / fp8_max

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    # scale shape: (M, num_groups) -> broadcast to (M, num_groups, block_size)
    y = x_blocked / scale.unsqueeze(2)
    y = y.clamp(fp8_min, fp8_max)

    # Reshape output back to original shape and cast to fp8
    y = y.view(orig_shape).to(torch.float8_e4m3fn)

    # Reshape scale to match expected output shape: (*orig_shape[:-1], num_groups)
    s = scale.view(*orig_shape[:-1], num_groups)

    return y, s


def _assert_fp8_equal(ref: torch.Tensor, out: torch.Tensor) -> None:
    assert ref.dtype == torch.float8_e4m3fn
    assert out.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        ref.view(torch.uint8), out.view(torch.uint8), atol=0, rtol=0
    )


class TestActQuantCPU(CustomTestCase):
    shapes = [(1, 128), (3, 256), (2, 3, 128), (2, 2, 384)]
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    scale_fmts = [None, "power2"]

    def _run_case(self, shape, dtype, scale_fmt):
        torch.manual_seed(1234)
        x = (torch.randn(shape, dtype=torch.float32) * 3.0).to(dtype).contiguous()

        ref_y, ref_scale = act_quant_pytorch(x, block_size=128, scale_fmt=scale_fmt)
        out_y, out_scale = torch.ops.sgl_kernel.act_quant_cpu(x, 128, scale_fmt)

        _assert_fp8_equal(ref_y, out_y)
        torch.testing.assert_close(ref_scale, out_scale, atol=0, rtol=0)

    def test_act_quant_cpu(self):
        for shape, dtype, scale_fmt in itertools.product(
            self.shapes, self.dtypes, self.scale_fmts
        ):
            with self.subTest(shape=shape, dtype=dtype, scale_fmt=scale_fmt):
                self._run_case(shape, dtype, scale_fmt)

    def test_zeros_use_min_scale(self):
        x = torch.zeros((2, 128), dtype=torch.bfloat16)
        ref_y, ref_scale = act_quant_pytorch(x, block_size=128)
        out_y, out_scale = torch.ops.sgl_kernel.act_quant_cpu(x, 128, None)

        _assert_fp8_equal(ref_y, out_y)
        torch.testing.assert_close(ref_scale, out_scale, atol=0, rtol=0)

    def test_invalid_block_size(self):
        x = torch.randn((2, 129), dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, "Last dimension size must be divisible"
        ):
            torch.ops.sgl_kernel.act_quant_cpu(x, 128, None)


if __name__ == "__main__":
    unittest.main()
