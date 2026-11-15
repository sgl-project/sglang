"""The fused MoE finalize must match an fp32 gather-and-sum reference: weighted
sum of the permuted rows in k order, -1 rows skipped, shared row added last."""

from __future__ import annotations

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils.compile.parallel import parallel_compile
from sglang.kernels.ops.moe import moe_finalize_fuse_shared as finalize_ops
from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
WEIGHT_DTYPES = [torch.float32, torch.bfloat16]


@pytest.fixture(scope="module", autouse=True)
def _build_all():
    # (weight dtype, top_k, hidden, shared, pdl) of every call below; the hidden sizes stand for
    # the two vector widths the tests reach (8: 4096, 4: 1028)
    combos = itertools.product(
        WEIGHT_DTYPES, (6, 8, 10, 16), (4096, 1028), (True, False), (False, True)
    )
    parallel_compile(
        [lambda c=c: finalize_ops._jit_finalize_module(*c) for c in combos]
    )


def _inputs(
    num_tokens, hidden, top_k, weight_dtype, *, skip=0.0, shared=True, pad=0, seed=0
):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = num_tokens * top_k + 256
    gemm2_out = torch.randn(
        rows, hidden + pad, device=DEVICE, dtype=torch.bfloat16, generator=g
    )
    idx = torch.randperm(rows, device=DEVICE, generator=g)[: num_tokens * top_k].to(
        torch.int32
    )
    if skip:
        idx[torch.rand(idx.shape, device=DEVICE, generator=g) < skip] = -1
    weights = torch.rand(num_tokens, top_k, device=DEVICE, generator=g).to(weight_dtype)
    shared_output = (
        torch.randn(
            num_tokens, hidden, device=DEVICE, dtype=torch.bfloat16, generator=g
        )
        if shared
        else None
    )
    return gemm2_out, idx, weights, shared_output


def _reference(gemm2_out, idx, weights, shared_output, top_k):
    num_tokens = weights.shape[0]
    width = shared_output.shape[1] if shared_output is not None else gemm2_out.shape[1]
    idx = idx.view(num_tokens, top_k).long()
    acc = torch.zeros(num_tokens, width, device=DEVICE, dtype=torch.float32)
    for k in range(top_k):
        valid = (idx[:, k] >= 0).unsqueeze(1)
        rows = gemm2_out[idx[:, k].clamp(min=0), :width].float()
        acc += torch.where(valid, weights[:, k : k + 1].float() * rows, 0.0)
    if shared_output is not None:
        acc += shared_output.float()
    return acc.to(torch.bfloat16)


def _assert_close(out, ref):
    # The kernel fuses multiply-add (FFMA); the reference rounds the product.
    torch.testing.assert_close(out.float(), ref.float(), rtol=2**-7, atol=2**-7)


@pytest.mark.parametrize(
    "hidden,top_k",
    [
        (7168, 8),
        (6144, 8),
        (5120, 8),
        (4096, 6),
        (7168, 6),
        (3584, 16),
        (4096, 10),
        (2880, 8),
        (1028, 10),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 3, 64, 192, 384, 1000])
@pytest.mark.parametrize("weight_dtype", WEIGHT_DTYPES)
def test_matches_reference(hidden, top_k, num_tokens, weight_dtype):
    args = _inputs(num_tokens, hidden, top_k, weight_dtype, seed=num_tokens + hidden)
    out = moe_finalize_fuse_shared(*args, top_k, enable_pdl=False)
    _assert_close(out, _reference(*args, top_k))


@pytest.mark.parametrize("top_k", [6, 16])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("pad", [0, 128])
@pytest.mark.parametrize(
    "enable_pdl,routing_ready", [(False, False), (True, False), (True, True)]
)
def test_skipped_rows_padding_and_pdl(top_k, shared, pad, enable_pdl, routing_ready):
    if not shared and pad:
        pytest.skip("without a shared output the result keeps the padded width")
    args = _inputs(97, 3584, top_k, torch.float32, skip=0.3, shared=shared, pad=pad)
    out = moe_finalize_fuse_shared(
        *args, top_k, enable_pdl=enable_pdl, routing_ready=routing_ready
    )
    assert out.shape == (97, 3584 + (0 if shared else pad))
    _assert_close(out, _reference(*args, top_k))


@pytest.mark.parametrize("top_k", [6, 16])
def test_into_caller_buffer(top_k):
    # A row-strided view stands in for a caller-owned (e.g. symmetric-memory) buffer.
    gemm2_out, idx, weights, _ = _inputs(37, 3584, top_k, torch.bfloat16, skip=0.1)
    buf = torch.full((37, 4096), 7.0, device=DEVICE, dtype=torch.bfloat16)
    out = moe_finalize_fuse_shared(
        gemm2_out, idx, weights, None, top_k, enable_pdl=True, out=buf[:, :3584]
    )
    assert out.data_ptr() == buf.data_ptr()
    _assert_close(buf[:, :3584], _reference(gemm2_out, idx, weights, None, top_k))
    assert torch.all(buf[:, 3584:] == 7.0)


def test_all_slots_skipped_and_nonfinite():
    gemm2_out, idx, weights, shared = _inputs(16, 7168, 8, torch.float32)
    idx[:8] = -1  # every slot of token 0 skipped: its output is its shared row
    gemm2_out[idx[8].item(), 5] = float("inf")
    gemm2_out[idx[9].item(), 9] = float("nan")
    out = moe_finalize_fuse_shared(gemm2_out, idx, weights, shared, 8)
    ref = _reference(gemm2_out, idx, weights, shared, 8)
    assert torch.equal(out[0], shared[0])
    assert torch.equal(out.isnan(), ref.isnan()) and torch.equal(
        out.isinf(), ref.isinf()
    )
    _assert_close(out.nan_to_num(0, 0, 0), ref.nan_to_num(0, 0, 0))


def test_many_tokens_per_cta_and_empty():
    # Past one resident wave each CTA walks several tokens.
    for top_k in (8, 16):
        args = _inputs(5000, 3584, top_k, torch.float32, skip=0.1)
        _assert_close(moe_finalize_fuse_shared(*args, top_k), _reference(*args, top_k))
    empty = _inputs(0, 7168, 8, torch.float32)
    assert moe_finalize_fuse_shared(*empty, 8).shape == (0, 7168)


def test_deterministic():
    args = _inputs(64, 7168, 8, torch.float32, skip=0.1)
    assert torch.equal(
        moe_finalize_fuse_shared(*args, 8), moe_finalize_fuse_shared(*args, 8)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
