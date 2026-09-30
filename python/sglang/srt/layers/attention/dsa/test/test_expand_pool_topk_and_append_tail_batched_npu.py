"""Accuracy test for _expand_pool_topk_and_append_tail_batched_npu.

The NPU Triton kernel lives in ``dsa_indexer_kpool._expand_pool_topk_and_append_tail_batched_npu``
and is a drop-in replacement for the PyTorch implementation of
``IndexerKPool._expand_pool_topk_and_append_tail_batched``. This test calls both
implementations on identical inputs and compares the output token-level index
tensors.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
    _expand_pool_topk_and_append_tail_batched_npu,
)

DEVICE = "npu"


# ============================================================================
# region auxiliary functions
# ============================================================================


def _assert_close_by_dtype(cal, ref):
    """Select the precision tolerance from the output dtype."""
    assert cal.dtype == ref.dtype, f"dtype mismatch: {cal.dtype} vs {ref.dtype}"
    if cal.dtype == torch.float32:
        torch.testing.assert_close(ref, cal, rtol=1e-5, atol=1e-5, equal_nan=True)
    elif cal.dtype == torch.float16:
        torch.testing.assert_close(ref, cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif cal.dtype == torch.bfloat16:
        torch.testing.assert_close(ref, cal, rtol=5e-3, atol=5e-3, equal_nan=True)
    elif cal.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        assert torch.equal(cal, ref), f"Integer tensors not equal for {cal.dtype}"
    elif cal.dtype == torch.bool:
        assert torch.equal(cal, ref), "Boolean tensors not equal"
    else:
        raise ValueError(f"Unsupported dtype: {cal.dtype}")


def _expand_pool_topk_and_append_tail_batched_ref(
    pool_indices: torch.Tensor,       # [n_real, n_pool_topk] int32
    seq_lens_per_token: torch.Tensor,  # [n_real] int32
    n_real: int,
    num_q_padded: int,
    pool_size: int = 4,
) -> torch.Tensor:
    """Reference: PyTorch implementation from dsa_indexer_kpool.py (original)."""
    n_pool_topk = pool_indices.shape[1]
    index_topk = n_pool_topk * pool_size
    tail_pool = pool_size - 1
    device = pool_indices.device

    n_real = min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])

    offsets = torch.arange(pool_size, device=device, dtype=torch.int32)
    token_indices = (
        pool_indices[:n_real].to(torch.int32).unsqueeze(-1) * pool_size + offsets
    )  # [n_real, n_pool_topk, pool_size]
    token_indices = token_indices.reshape(n_real, index_topk)

    out_cols = index_topk + tail_pool
    out = torch.full(
        (num_q_padded, out_cols), -1, dtype=torch.int32, device=device
    )
    out[:n_real, :index_topk] = token_indices

    seq_lens_used = seq_lens_per_token[:n_real]
    pool_lens_per_token = torch.div(
        seq_lens_used, pool_size, rounding_mode="floor"
    ).to(torch.int32)
    tail_starts = pool_lens_per_token * pool_size
    tail_counts = (seq_lens_used - tail_starts).to(torch.int32)

    for t in range(tail_pool):
        mask = tail_counts > t
        val = (tail_starts + t).to(torch.int32)
        out[:n_real, index_topk + t] = torch.where(
            mask, val, torch.full_like(val, -1)
        )

    return out


# endregion

# ============================================================================
# region precision test
# ============================================================================

# Fixed padding verifies that the kernel masks rows beyond n_real.
PAD_ROWS = 13


def test_op(n_real: int, num_q_padded: int):
    """Compare the NPU Triton kernel against the PyTorch reference."""
    torch.manual_seed(20)

    n_pool_topk = 512
    pool_size = 4
    index_topk = n_pool_topk * pool_size

    # Add unused input rows and verify that the kernel reads only valid rows.
    total_rows = n_real + PAD_ROWS
    pool_indices = torch.randint(
        0, 10000, (total_rows, n_pool_topk), dtype=torch.int32, device=DEVICE
    )
    seq_lens_per_token = (
        torch.arange(1, total_rows + 1, dtype=torch.int32, device=DEVICE) * pool_size
        + torch.randint(0, pool_size, (total_rows,), device=DEVICE, dtype=torch.int32)
    )

    tri_out = _expand_pool_topk_and_append_tail_batched_npu(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
        index_topk,
    )

    ref_out = _expand_pool_topk_and_append_tail_batched_ref(
        pool_indices, seq_lens_per_token, n_real, num_q_padded, pool_size
    )

    _assert_close_by_dtype(tri_out, ref_out)
    print(f"[PASSED] n_real={n_real}, num_q_padded={num_q_padded}")


@pytest.mark.parametrize(
    "n_real, num_q_padded",
    [
        (2048, 2048),
        (1024, 2048),
        (1035, 2048),
        (4095, 4096),
    ],
)
def test_expand_pool_topk_and_append_tail_batched(n_real: int, num_q_padded: int):
    test_op(n_real, num_q_padded)


# endregion

# ============================================================================
# region main
# ============================================================================

if __name__ == "__main__":
    test_op(2048, 2048)
    test_op(1024, 2048)
    test_op(1035, 2048)
    test_op(4095, 4096)


# endregion
