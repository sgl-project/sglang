import pytest
import torch
from sgl_kernel import gptq_gemm as aot_gptq_gemm
from sgl_kernel import gptq_shuffle as aot_gptq_shuffle

from sglang.kernels.ops.quantization.gptq import gptq_gemm, gptq_shuffle
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def pack_rows(values: torch.Tensor, bit: int) -> torch.Tensor:
    pack = 32 // bit
    out = torch.zeros(
        (values.shape[0] // pack, values.shape[1]),
        dtype=torch.int32,
        device=values.device,
    )
    for index in range(pack):
        out |= values[index::pack].to(torch.int32) << (index * bit)
    return out


def pack_cols(values: torch.Tensor, bit: int) -> torch.Tensor:
    pack = 32 // bit
    out = torch.zeros(
        (values.shape[0], values.shape[1] // pack),
        dtype=torch.int32,
        device=values.device,
    )
    for index in range(pack):
        out |= values[:, index::pack].to(torch.int32) << (index * bit)
    return out


def make_args(m: int, n: int, k: int, bit: int = 4, group_size: int = 128):
    torch.manual_seed(7)
    groups = k // group_size
    weight = torch.randn(k, n, dtype=torch.float16, device="cuda")
    grouped = weight.reshape(groups, group_size, n)
    maximum = grouped.amax(dim=1, keepdim=True)
    minimum = grouped.amin(dim=1, keepdim=True)
    scales = ((maximum - minimum) / (2**bit - 1)).clamp(min=1e-6)
    zeros = (-minimum / scales).round()
    quant = ((grouped / scales + zeros).round().clamp(0, 2**bit - 1)).to(torch.uint8)
    q_weight = pack_rows(quant.reshape(k, n), bit)
    q_zeros = pack_cols((zeros.to(torch.uint8) - 1).reshape(groups, n), bit)
    g_idx = torch.arange(k, dtype=torch.int32, device="cuda") // group_size
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    return a, q_weight, q_zeros, scales.squeeze(1), g_idx, False, bit


# The M <= 8 AOT path accumulates into torch.empty() with atomicAdd, so two
# invocations are intentionally not bit-deterministic. M > 8 uses the
# reconstruct + cuBLAS path and is suitable for an exact migration check.
@pytest.mark.parametrize("m", [9, 128])
@pytest.mark.parametrize("n,k", [(256, 256), (2048, 2048)])
def test_gptq_gemm_matches_aot(m: int, n: int, k: int):
    args = make_args(m, n, k)
    expected = aot_gptq_gemm(*args)
    actual = gptq_gemm(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gptq_shuffle_matches_aot():
    _, q_weight, _, _, _, _, bit = make_args(1, 256, 256)
    q_perm = torch.randperm(256, dtype=torch.int32, device="cuda")
    expected = q_weight.clone()
    actual = q_weight.clone()
    aot_gptq_shuffle(expected, q_perm, bit)
    gptq_shuffle(actual, q_perm, bit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
