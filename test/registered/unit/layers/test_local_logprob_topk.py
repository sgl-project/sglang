"""Exact tiled Top-N and unchanged normalizer contract."""

import pytest
import torch
from sglang.srt.layers.local_logprob_topk import (
    can_use_local_topk,
    local_logsumexp_topk,
)
from sglang.srt.layers.logprob_processor import compute_row_log_normalizer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,cols,k",
    [(128, 37984, 20), (64, 38720, 20), (7, 2051, 20), (7, 19, 5), (3, 131072, 32)],
)
def test_finite(dtype, rows, cols, k):
    torch.manual_seed(314159)
    x = torch.randn((rows, cols), dtype=dtype, device="cuda")
    m, ls, values, ids = local_logsumexp_topk(x, k)
    rm, rls = compute_row_log_normalizer(x)
    torch.testing.assert_close(m, rm, atol=0, rtol=0)
    torch.testing.assert_close(ls, rls, atol=0, rtol=0)
    torch.testing.assert_close(values, x.topk(k, dim=-1).values.float(), atol=0, rtol=0)
    torch.testing.assert_close(values, x.gather(1, ids).float(), atol=0, rtol=0)
    assert torch.all(ids.sort(dim=-1).values[:, 1:] != ids.sort(dim=-1).values[:, :-1])


def test_strides_ties_and_chunk_consistency():
    torch.manual_seed(314159)
    x = torch.randint(-2, 3, (7, 4102), device="cuda").float()[:, ::2]
    whole = local_logsumexp_topk(x, 20)
    for lo, hi, k in [(0, 3, 20), (3, 7, 5)]:
        chunk = local_logsumexp_topk(x[lo:hi], k)
        assert torch.equal(chunk[0], whole[0][lo:hi])
        assert torch.equal(chunk[1], whole[1][lo:hi])
        torch.testing.assert_close(
            chunk[2], x[lo:hi].gather(1, chunk[3]), atol=0, rtol=0
        )


def test_resource_fallback():
    assert not can_use_local_topk(torch.empty((1, 37984), device="cuda"), 33)
    assert not can_use_local_topk(torch.empty((1, 131073), device="cuda"), 32)
    assert not can_use_local_topk(torch.empty((1, 10), device="cuda"), 11)
    assert not can_use_local_topk(torch.empty((0, 10), device="cuda"), 5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,cols", [(128, 37984), (7, 2051), (1, 1)])
def test_top_one(dtype, rows, cols):
    torch.manual_seed(314159)
    x = torch.randn((rows, cols), dtype=dtype, device="cuda")
    maximum, log_sum, values, indices = local_logsumexp_topk(x, 1)
    reference_max, reference_sum = compute_row_log_normalizer(x)
    torch.testing.assert_close(maximum, reference_max, atol=0, rtol=0)
    torch.testing.assert_close(log_sum, reference_sum, atol=0, rtol=0)
    torch.testing.assert_close(values, x.topk(1, dim=-1).values.float(), atol=0, rtol=0)
    torch.testing.assert_close(values, x.gather(1, indices).float(), atol=0, rtol=0)


@pytest.mark.parametrize("cols", [4096, 37984])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [1, 20])
def test_nonfinite_cuda_contract(cols, dtype, k):
    x = torch.zeros((7, cols), device="cuda", dtype=dtype)
    x[0].fill_(float("-inf"))
    x[1].fill_(float("inf"))
    x[2, 0] = float("nan")
    x[3, 0] = x[3, -1] = float("inf")
    x[4, -1] = float("nan")
    x[5, 0], x[5, 2050] = float("inf"), float("nan")
    x[6].fill_(float("nan"))
    actual = local_logsumexp_topk(x, k)
    reference = compute_row_log_normalizer(x)
    for a, b in zip(actual[:2], reference):
        torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(
        actual[2], x.topk(k, dim=-1).values.float(), atol=0, rtol=0, equal_nan=True
    )
    torch.testing.assert_close(
        actual[2], x.gather(1, actual[3]).float(), atol=0, rtol=0, equal_nan=True
    )
