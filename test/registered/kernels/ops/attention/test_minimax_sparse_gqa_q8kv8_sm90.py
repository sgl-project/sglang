import importlib
import math

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

MODULE_NAME = "sglang.kernels.ops.attention.minimax_sparse.prefill.sgl_native_q8kv8"
FP8 = torch.float8_e4m3fn


def _make_case(num_q_heads: int, num_kv_heads: int):
    torch.manual_seed(7)
    device = "cuda"
    total_q, head_dim, block_size = 2, 128, 128
    seq_len, prefix_len, topk = 256, 254, 2
    q = (torch.randn(total_q, num_q_heads, head_dim, device=device) * 0.2).to(FP8)
    k = (torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 0.2).to(FP8)
    v = (torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 0.2).to(FP8)
    req_to_token = torch.randperm(seq_len, device=device, dtype=torch.int64).to(
        torch.int32
    )[None, :]
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


@pytest.mark.skipif(
    not is_sm90_supported(), reason="native Q8KV8 sparse GQA requires SM90 CUDA"
)
def test_native_q8kv8_accepts_strided_q_with_contiguous_last_dim():
    native_module = importlib.import_module(MODULE_NAME)
    args = list(_make_case(4, 1))
    q = args[0]
    expected = native_module.sgl_native_q8kv8_sparse_prefill(
        q=q,
        k_cache=args[1],
        v_cache=args[2],
        req_to_token=args[3],
        slot_ids=args[4],
        topk_idx=args[5],
        cu_seqlens=args[6],
        seq_lens=args[7],
        prefix_lens=args[8],
        block_size_k=args[9],
    )
    storage = torch.empty(
        q.shape[0], q.shape[1], q.shape[2] + 1, device=q.device, dtype=q.dtype
    )
    q_strided = storage[..., : q.shape[-1]]
    q_strided.copy_(q)
    assert not q_strided.is_contiguous()
    assert q_strided.stride(-1) == 1
    args[0] = q_strided

    actual = native_module.sgl_native_q8kv8_sparse_prefill(
        q=args[0],
        k_cache=args[1],
        v_cache=args[2],
        req_to_token=args[3],
        slot_ids=args[4],
        topk_idx=args[5],
        cu_seqlens=args[6],
        seq_lens=args[7],
        prefix_lens=args[8],
        block_size_k=args[9],
    )
    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
