import importlib
import math

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

MODULE_NAME = "sglang.kernels.ops.attention.minimax_sparse.prefill.sgl_native_q8kv8"
FP8 = torch.float8_e4m3fn


def test_native_q8kv8_requires_page_aligned_sparse_blocks():
    native_module = importlib.import_module(MODULE_NAME)

    native_module._validate_page_contract(block_size_k=128, page_size=128)
    with pytest.raises(
        ValueError,
        match="requires page_size=block_size_k=128",
    ):
        native_module._validate_page_contract(block_size_k=128, page_size=64)


def _make_case(num_q_heads: int, num_kv_heads: int):
    torch.manual_seed(7)
    device = "cuda"
    total_q, head_dim, block_size = 2, 128, 128
    seq_len, prefix_len, topk = 256, 254, 2
    q = (torch.randn(total_q, num_q_heads, head_dim, device=device) * 0.2).to(FP8)
    k = (torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 0.2).to(FP8)
    v = (torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 0.2).to(FP8)
    page_order = torch.tensor([1, 0], device=device, dtype=torch.int64)
    req_to_token = (
        (
            page_order[:, None] * block_size
            + torch.arange(block_size, device=device, dtype=torch.int64)[None, :]
        )
        .reshape(1, seq_len)
        .to(torch.int32)
    )
    slot_ids = torch.zeros(1, device=device, dtype=torch.int64)
    topk_idx = (
        torch.tensor([0, 1], device=device, dtype=torch.int32)
        .view(1, 1, topk)
        .expand(num_kv_heads, total_q, topk)
        .contiguous()
    )
    cu_seqlens = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix_len], device=device, dtype=torch.int32)
    return (
        q,
        k,
        v,
        req_to_token,
        slot_ids,
        topk_idx,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        block_size,
    )


def _reference(
    q,
    k,
    v,
    req_to_token,
    topk_idx,
    prefix_len,
    block_size,
    sm_scale,
    q_scale,
    k_scale,
    v_scale,
):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    out = torch.zeros(
        total_q, num_q_heads, head_dim, device=q.device, dtype=torch.float32
    )
    qf = q.float() * q_scale
    kf = k.float() * k_scale
    vf = v.float() * v_scale
    mapping = req_to_token[0].long()

    for qi in range(total_q):
        q_position = prefix_len + qi
        for qh in range(num_q_heads):
            kvh = qh // group_size
            running_max = -math.inf
            running_sum = 0.0
            accumulator = torch.zeros(head_dim, device=q.device, dtype=torch.float32)
            for selected_block in topk_idx[kvh, qi].tolist():
                if selected_block < 0:
                    continue
                for offset in range(block_size):
                    logical_position = selected_block * block_size + offset
                    if logical_position > q_position:
                        continue
                    slot = mapping[logical_position]
                    score = torch.dot(qf[qi, qh], kf[slot, kvh]).item() * sm_scale
                    new_max = max(running_max, score)
                    old_scale = math.exp(running_max - new_max)
                    probability = math.exp(score - new_max)
                    probability_fp8 = (
                        torch.tensor(probability, device=q.device, dtype=torch.float32)
                        .to(FP8)
                        .float()
                        .item()
                    )
                    accumulator = (
                        accumulator * old_scale + probability_fp8 * vf[slot, kvh]
                    )
                    running_sum = running_sum * old_scale + probability
                    running_max = new_max
            if running_sum:
                out[qi, qh] = accumulator / running_sum
    return out.to(torch.bfloat16)


@pytest.mark.skipif(
    not is_sm90_supported(), reason="native Q8KV8 score-only prefill requires SM90 CUDA"
)
def test_native_q8kv8_score_handles_varlen_paged_batches():
    native_module = importlib.import_module(MODULE_NAME)
    torch.manual_seed(11)
    device = "cuda"
    q_lens = [7, 65]
    seq_values = [130, 257]
    prefix_values = [123, 192]
    total_q = sum(q_lens)
    num_q_heads = 8
    max_seq_len = max(seq_values)
    page_size = 128
    page_counts = [(seq_len + page_size - 1) // page_size for seq_len in seq_values]
    max_slots = sum(page_counts) * page_size

    q = (torch.randn(total_q, num_q_heads, 128, device=device) * 0.2).to(FP8)
    k = (torch.randn(max_slots, 1, 128, device=device) * 0.2).to(FP8)
    req_to_token = torch.zeros(
        len(q_lens), max_seq_len, dtype=torch.int32, device=device
    )
    page_offset = 0
    for batch, (seq_len, page_count) in enumerate(zip(seq_values, page_counts)):
        page_order = torch.randperm(page_count, device=device) + page_offset
        slots = (
            page_order[:, None] * page_size
            + torch.arange(page_size, device=device, dtype=torch.int64)[None, :]
        ).reshape(-1)
        req_to_token[batch, :seq_len] = slots[:seq_len].to(torch.int32)
        page_offset += page_count

    slot_ids = torch.arange(len(q_lens), dtype=torch.int64, device=device)
    cu_seqlens = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_values, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(prefix_values, dtype=torch.int32, device=device)
    sm_scale = 128**-0.5
    q_scale, k_scale = 0.75, 1.25

    actual = native_module.sgl_native_q8kv8_sparse_prefill_score(
        q=q,
        k_cache=k,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_k=max_seq_len,
        block_size_k=128,
        page_size=128,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
    )

    expected = torch.full_like(actual, float("-inf"))
    for batch, q_len in enumerate(q_lens):
        q_start = cu_seqlens[batch].item()
        for local_q in range(q_len):
            q_idx = q_start + local_q
            q_position = prefix_values[batch] + local_q
            for q_head in range(num_q_heads):
                for block in range(math.ceil(seq_values[batch] / 128)):
                    begin = block * 128
                    end = min(begin + 128, seq_values[batch], q_position + 1)
                    if begin >= end:
                        continue
                    slots = req_to_token[batch, begin:end].long()
                    logits = k[slots, 0].float() @ q[q_idx, q_head].float()
                    expected[q_head, q_idx, block] = (
                        logits.max() * sm_scale * q_scale * k_scale
                    )

    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not is_sm90_supported(), reason="native Q8KV8 sparse GQA requires SM90 CUDA"
)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [(16, 16), (16, 8), (16, 4), (16, 2), (16, 1)],
)
@pytest.mark.parametrize(
    "q_scale,k_scale,v_scale", [(1.0, 1.0, 1.0), (0.75, 1.25, 0.625)]
)
def test_native_q8kv8_matches_fp8_probability_reference(
    num_q_heads, num_kv_heads, q_scale, k_scale, v_scale
):
    native_module = importlib.import_module(MODULE_NAME)
    (
        q,
        k,
        v,
        req_to_token,
        slot_ids,
        topk_idx,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        block_size,
    ) = _make_case(num_q_heads, num_kv_heads)
    sm_scale = q.shape[-1] ** -0.5

    actual = native_module.sgl_native_q8kv8_sparse_prefill(
        q=q,
        k_cache=k,
        v_cache=v,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        topk_idx=topk_idx,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        block_size_k=block_size,
        page_size=block_size,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = _reference(
        q,
        k,
        v,
        req_to_token,
        topk_idx,
        prefix_lens.item(),
        block_size,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
    )

    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
