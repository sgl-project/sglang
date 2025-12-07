# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
from csrc.attn.vmoba_attn.vmoba import moba_attn_varlen


def generate_test_data(
    batch_size, total_seqlen, num_heads, head_dim, dtype, device="cuda"
):
    """
    Generates random data for testing the variable-length attention function.
    """
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)

    # Generate sequence lengths for each item in the batch
    if batch_size > 1:
        # Ensure sequence lengths are reasonably distributed
        avg_seqlen = total_seqlen // batch_size
        seqlens = [
            random.randint(avg_seqlen // 2, avg_seqlen + avg_seqlen // 2)
            for _ in range(batch_size - 1)
        ]
        remaining_len = total_seqlen - sum(seqlens)
        if remaining_len > 0:
            seqlens.append(remaining_len)
        else:  # Adjust if sum exceeds total_seqlen
            seqlens.append(avg_seqlen)
            current_sum = sum(seqlens)
            seqlens[-1] -= current_sum - total_seqlen
        # Ensure all lengths are positive
        seqlens = [max(1, s) for s in seqlens]
        # Final adjustment to match total_seqlen
        seqlens[-1] += total_seqlen - sum(seqlens)

    else:
        seqlens = [total_seqlen]

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
        device=device,
        dtype=torch.int32,
    )
    max_seqlen = max(seqlens) if seqlens else 0

    q = torch.randn(
        (total_seqlen, num_heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    k = torch.randn(
        (total_seqlen, num_heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    v = torch.randn(
        (total_seqlen, num_heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    return q, k, v, cu_seqlens, max_seqlen


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("total_seqlen", [512, 1024])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("moba_chunk_size", [64])
@pytest.mark.parametrize("moba_topk", [2, 4])
@pytest.mark.parametrize("select_mode", ["topk", "threshold"])
@pytest.mark.parametrize("threshold_type", ["query_head", "head_global", "overall"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moba_attn_varlen_forward(
    batch_size,
    total_seqlen,
    num_heads,
    head_dim,
    moba_chunk_size,
    moba_topk,
    select_mode,
    threshold_type,
    dtype,
):
    """
    Tests the forward pass of moba_attn_varlen for basic correctness.
    It checks output shape, dtype, and for the presence of NaNs/Infs.
    """
    if dtype == torch.float32:
        pytest.skip("float32 is not supported in flash attention")

    q, k, v, cu_seqlens, max_seqlen = generate_test_data(
        batch_size, total_seqlen, num_heads, head_dim, dtype
    )

    # Ensure chunk size is not larger than the smallest sequence length
    min_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).min().item()
    if moba_chunk_size > min_seqlen:
        pytest.skip(
            "moba_chunk_size is larger than the minimum sequence length in the batch"
        )

    try:
        output = moba_attn_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk,
            select_mode=select_mode,
            threshold_type=threshold_type,
            simsum_threshold=0.5,  # A reasonable default for threshold mode
        )
    except Exception as e:
        pytest.fail(f"moba_attn_varlen forward pass failed with exception: {e}")

    # 1. Check output shape
    assert (
        output.shape == q.shape
    ), f"Expected output shape {q.shape}, but got {output.shape}"

    # 2. Check output dtype
    assert (
        output.dtype == q.dtype
    ), f"Expected output dtype {q.dtype}, but got {output.dtype}"

    # 3. Check for NaNs or Infs in the output
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf values"
