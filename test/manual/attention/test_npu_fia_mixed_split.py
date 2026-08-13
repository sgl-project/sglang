"""A5-only byte-equivalence probe for mixed FIA split.

Run on an Ascend 950 host with:

    pytest -q test/manual/attention/test_npu_fia_mixed_split.py -s
"""

import math

import pytest
import torch
from sglang.srt.hardware_backend.npu.utils import supports_fia_mixed_split

torch_npu = pytest.importorskip("torch_npu")


def _run_fia(
    query,
    key,
    value,
    block_table,
    mask,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_heads,
    num_kv_heads,
    head_dim,
    block_size,
):
    output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        block_size=block_size,
        block_table=block_table,
        atten_mask=mask,
        sparse_mode=3,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        scale=1.0 / math.sqrt(head_dim),
    )
    return output


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2)])
def test_single_and_prefill_first_split_are_byte_identical(
    num_heads, num_kv_heads
):
    if not supports_fia_mixed_split():
        pytest.skip("FIA mixed split is only validated on Ascend 950 / A5")

    torch.manual_seed(0)
    device = torch.device("npu")
    dtype = torch.bfloat16
    head_dim = 128
    block_size = 128
    num_prefill_reqs = 2
    num_prefill_tokens = 33
    num_tokens = 35

    query = torch.randn(
        (num_tokens, num_heads, head_dim), dtype=dtype, device=device
    )
    key = torch.randn(
        (4, block_size, num_kv_heads * head_dim), dtype=dtype, device=device
    )
    value = torch.randn_like(key)
    block_table = torch.arange(4, dtype=torch.int32, device=device).view(4, 1)
    mask = torch.triu(
        torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1
    )
    q_cumulative = [16, 33, 34, 35]
    kv_lengths = torch.tensor([64, 80, 81, 96], dtype=torch.int32)

    single = _run_fia(
        query,
        key,
        value,
        block_table,
        mask,
        q_cumulative,
        kv_lengths,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    )

    prefill = _run_fia(
        query[:num_prefill_tokens],
        key,
        value,
        block_table[:num_prefill_reqs],
        mask,
        q_cumulative[:num_prefill_reqs],
        kv_lengths[:num_prefill_reqs],
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    )
    decode = _run_fia(
        query[num_prefill_tokens:],
        key,
        value,
        block_table[num_prefill_reqs:],
        mask,
        [1, 2],
        kv_lengths[num_prefill_reqs:],
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    )
    split = torch.empty_like(single)
    split[:num_prefill_tokens].copy_(prefill)
    split[num_prefill_tokens:].copy_(decode)
    torch_npu.npu.synchronize()

    if not torch.equal(single, split):
        mismatch_count = torch.count_nonzero(single != split).item()
        max_abs_diff = (single.float() - split.float()).abs().max().item()
        pytest.fail(
            "single and split FIA outputs are not byte-identical: "
            f"mismatches={mismatch_count}, max_abs_diff={max_abs_diff}, "
            f"shape={tuple(single.shape)}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}"
        )
