"""Pre-build the AITER blockscale B-preshuffle CKTile module."""

import time

import aiter
import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_cktile


def main():
    m, n, k = 1216, 7168, 7168
    kernel_name = (
        "a8w8_blockscale_cktile_192x256x128_4x2x1_" "16x16x128_intrawave_0x1x0_3"
    )
    device = torch.device("cuda:0")

    x = torch.zeros((m, k), dtype=aiter.dtypes.fp8, device=device)
    # Zero is invariant under the (16, 16) weight shuffle.
    weight = torch.zeros((n, k), dtype=aiter.dtypes.fp8, device=device)
    x_scale = torch.ones((m, k // 128), dtype=torch.float32, device=device)
    weight_scale = torch.ones((n // 128, k // 128), dtype=torch.float32, device=device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    start = time.time()
    gemm_a8w8_blockscale_bpreshuffle_cktile(
        x, weight, x_scale, weight_scale, out, True, kernel_name
    )
    torch.cuda.synchronize()
    print(
        "AITER blockscale B-preshuffle CKTile warmup completed "
        f"in {time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
