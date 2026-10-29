from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import quantize_fp4_indexer_ref

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
GROUP_SIZE = 32


@pytest.mark.parametrize("num_tokens", [1, 7, 96])
def test_quantize_fp4_indexer_tensor(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    x[0, :8] = torch.tensor(
        [-8.0, -6.0, -3.0, -1.5, 0.0, 0.5, 2.0, 8.0],
        device=get_device(),
        dtype=torch.bfloat16,
    )

    x_fp4, x_sf = quantize_fp4_indexer_tensor(x)
    ref_fp4, ref_sf = quantize_fp4_indexer_ref(x)

    torch.testing.assert_close(x_fp4.view(torch.uint8), ref_fp4)
    torch.testing.assert_close(x_sf, ref_sf)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Vectorized prefill dispatch targets Blackwell",
)
@pytest.mark.parametrize("rows", [4096, 4097, 16384, 524288])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rne", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_prefill_quantization_matches_single_row_and_replay(rows, dtype, rne, strided):
    from sglang.kernels.ops.quantization.fp4_indexer_quant import (
        _quantize_fp4_indexer_kernel,
    )

    x = torch.randn(rows, 256 if strided else 128, device="cuda", dtype=dtype)
    if strided:
        x = x[:, ::2]

    def reference():
        q = torch.empty(rows, 64, device="cuda", dtype=torch.int8)
        sf = torch.empty(rows, device="cuda", dtype=torch.int32)
        _quantize_fp4_indexer_kernel[(rows,)](
            x.contiguous(),
            q,
            sf,
            BLOCK_N=128,
            GROUP_N=32,
            RNE=rne,
        )
        return q, sf

    for _ in range(3):
        quantize_fp4_indexer_tensor(x, rne)
        reference()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize_fp4_indexer_tensor(x, rne)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda", dtype=dtype
    )
    for scale in (0.0, 1e-6, 1.0, 1e3):
        x.normal_().mul_(scale)
        x[:2] = boundaries.repeat(16)
        x[1].neg_()
        graph.replay()
        for a, b in zip(actual, reference()):
            assert torch.equal(a, b)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Vectorized prefill dispatch targets Blackwell",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rne", [False, True])
def test_prefill_quantization_nonfinite_group_replay(dtype, rne):
    from sglang.kernels.ops.quantization.fp4_indexer_quant import (
        _quantize_fp4_indexer_kernel,
    )

    rows = 4097  # Also exercise the final partial CTA.
    x = torch.randn(rows, HEAD_DIM, device="cuda", dtype=dtype)
    expected = (
        torch.empty(rows, FP4_DIM, device="cuda", dtype=torch.int8),
        torch.empty(rows, device="cuda", dtype=torch.int32),
    )
    for _ in range(3):
        quantize_fp4_indexer_tensor(x, rne)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize_fp4_indexer_tensor(x, rne)
    for value in (float("inf"), float("-inf"), float("nan"), 0.0):
        x.normal_()
        x[:, 0] = value
        x[:, 32:64] = value
        x[-1, 96:] = value
        graph.replay()
        _quantize_fp4_indexer_kernel[(rows,)](
            x, *expected, BLOCK_N=HEAD_DIM, GROUP_N=GROUP_SIZE, RNE=rne
        )
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
