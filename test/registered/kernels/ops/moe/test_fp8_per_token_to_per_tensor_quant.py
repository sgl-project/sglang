"""Unit test for ``fp8_per_token_to_per_tensor_quant_triton`` across hidden sizes.

W4AFP8 DeepEP low-latency requantizes the fp8 dispatch payload with this kernel
before the first CUTLASS grouped GEMM.  The payload's hidden size is only
guaranteed to be a multiple of the fp8 scale-group size (128) -- e.g. 3584 for
Kimi-K3 -- so the kernel must handle a ``k`` tail that does not fill a whole
``K_BLOCK_SIZE`` (1024) block, and must still leave the rows past ``masked_m``
untouched.
"""

import pytest
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    fp8_per_token_to_per_tensor_quant_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

dev = "cuda"
FP8 = torch.float8_e4m3fn
K_SCALE_BLOCK_SIZE = 128
# Every value the kernel can produce below is a multiple of 0.25, so this
# sentinel cannot be matched by a kernel that wrongly writes a padding row.
SENTINEL = 0.375
OUTPUT_SCALE = 2.0


def _build(num_experts, m, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Integers in [-8, 8] with power-of-two per-token-group scales keep every
    # intermediate exactly representable in e4m3, so the reference below matches
    # bit-for-bit regardless of the rounding mode of the final cast.
    x = torch.randint(-8, 9, (num_experts, m, k), generator=g).float()
    exps = torch.randint(-1, 2, (num_experts, m, k // K_SCALE_BLOCK_SIZE), generator=g)
    x_scale = torch.pow(2.0, exps.float())
    return x.to(dev).to(FP8), x_scale.to(dev)


def _ref(x, x_scale):
    dequant = x.float() * x_scale.repeat_interleave(K_SCALE_BLOCK_SIZE, dim=2)
    return (dequant * (1.0 / OUTPUT_SCALE)).to(FP8)


# 7168: exact multiple of K_BLOCK_SIZE (the DeepSeek-V3 hidden size).
# 3584 / 1152: only 128-aligned, so the last k block is partially masked.
@pytest.mark.parametrize("k", [7168, 3584, 1152])
def test_masked_rows_and_k_tail(k):
    num_experts, m = 4, 48
    masked = [0, 1, 17, m]

    x, x_scale = _build(num_experts, m, k, seed=k)
    masked_m = torch.tensor(masked, dtype=torch.int32, device=dev)
    output_scale = torch.tensor([OUTPUT_SCALE], dtype=torch.float32, device=dev)
    output = torch.full((num_experts, m, k), SENTINEL, device=dev).to(FP8)

    fp8_per_token_to_per_tensor_quant_triton(
        x=x,
        x_scale=x_scale,
        masked_m=masked_m,
        output_scale=output_scale,
        output=output,
    )

    ref = _ref(x, x_scale)
    for e, valid in enumerate(masked):
        torch.testing.assert_close(
            output[e, :valid].float(), ref[e, :valid].float(), rtol=0, atol=0
        )
        # Padding rows are not part of any expert's GEMM problem size and must
        # stay as the caller left them.
        padding = output[e, valid:].float()
        torch.testing.assert_close(
            padding, torch.full_like(padding, SENTINEL), rtol=0, atol=0
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
