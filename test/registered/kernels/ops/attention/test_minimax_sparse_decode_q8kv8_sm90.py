import importlib
import inspect

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

MODULE = "sglang.kernels.ops.attention.minimax_sparse.decode.sgl_native_q8kv8"
FP8 = torch.float8_e4m3fn


def test_native_decode_split_count_is_graph_static():
    native = importlib.import_module(MODULE)

    assert native._choose_num_splits(batch_size=1, num_kv_heads=1, topk=32) == 32
    assert native._choose_num_splits(batch_size=8, num_kv_heads=1, topk=32) == 32
    assert native._choose_num_splits(batch_size=64, num_kv_heads=1, topk=32) == 4
    assert native._choose_num_splits(batch_size=64, num_kv_heads=8, topk=32) == 1


def test_native_decode_requires_page_aligned_sparse_blocks():
    native = importlib.import_module(MODULE)

    native._validate_page_contract(block_size_k=128, page_size=128)
    with pytest.raises(ValueError, match="page_size=block_size_k=128"):
        native._validate_page_contract(block_size_k=128, page_size=64)


def test_sparse_decode_exposes_independent_native_switch():
    sparse = importlib.import_module(
        "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
    )
    parameters = inspect.signature(sparse.minimax_sparse_decode).parameters

    assert "use_sgl_native_q8kv8_decode" in parameters
    assert "sgl_native_q8kv8_decode_strict" not in parameters


def _reference(q, k, v, req_to_token, slot_ids, seq_lens, topk_idx, scales):
    q_scale, k_scale, v_scale, sm_scale = scales
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    out = torch.zeros(
        batch, num_q_heads, head_dim, dtype=torch.float32, device=q.device
    )
    qf = q.float() * q_scale
    kf = k.float() * k_scale
    vf = v.float() * v_scale
    for b in range(batch):
        mapping = req_to_token[slot_ids[b].item()].long()
        for qh in range(num_q_heads):
            kvh = qh // group_size
            logits = []
            values = []
            for selected_block in topk_idx[kvh, b].tolist():
                if selected_block < 0:
                    continue
                begin = selected_block * 128
                end = min(begin + 128, seq_lens[b].item())
                if begin >= end:
                    continue
                slots = mapping[begin:end]
                logits.append((kf[slots, kvh] @ qf[b, qh]) * sm_scale)
                values.append(vf[slots, kvh])
            if logits:
                scores = torch.cat(logits)
                probs = torch.softmax(scores, dim=0).to(FP8).float()
                out[b, qh] = probs @ torch.cat(values)
    return out.to(torch.bfloat16)


def _case(batch=3, num_q_heads=8, num_kv_heads=1, topk=32):
    torch.manual_seed(7)
    max_blocks = 40
    page_size = 128
    max_len = max_blocks * page_size
    pages = batch * max_blocks
    q = (torch.randn(batch, num_q_heads, 128, device="cuda") * 0.2).to(FP8)
    k = (torch.randn(pages * page_size, num_kv_heads, 128, device="cuda") * 0.2).to(FP8)
    v = (torch.randn_like(k.float()) * 0.2).to(FP8)
    req_to_token = torch.empty(batch, max_len, dtype=torch.int32, device="cuda")
    for b in range(batch):
        page_order = torch.randperm(max_blocks, device="cuda") + b * max_blocks
        req_to_token[b] = (
            page_order[:, None] * page_size
            + torch.arange(page_size, device="cuda")[None, :]
        ).reshape(-1)
    slot_ids = torch.arange(batch, dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor(
        [max_len - 17, 9 * page_size + 3, page_size + 1][:batch],
        dtype=torch.int32,
        device="cuda",
    )
    topk_idx = torch.full(
        (num_kv_heads, batch, topk), -1, dtype=torch.int32, device="cuda"
    )
    topk_idx[:, 0, :topk] = torch.arange(topk, device="cuda", dtype=torch.int32)
    if batch > 1:
        topk_idx[:, 1, :5] = torch.tensor([0, 2, 4, 6, 8], device="cuda")
    # Batch 2 intentionally has no selected blocks: every split is empty.
    return q, k, v, req_to_token, slot_ids, seq_lens, topk_idx


@pytest.mark.skipif(not is_sm90_supported(), reason="requires SM90 CUDA")
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (8, 2), (8, 8)])
def test_native_decode_matches_reference_with_minus_one_and_empty_splits(
    num_q_heads, num_kv_heads
):
    native = importlib.import_module(MODULE)
    args = _case(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
    scales = (0.75, 1.25, 0.625, 128**-0.5)

    actual = native.sgl_native_q8kv8_sparse_decode(
        q=args[0],
        k_cache=args[1],
        v_cache=args[2],
        req_to_token=args[3],
        slot_ids=args[4],
        seq_lens=args[5],
        topk_idx=args[6],
        block_size_k=128,
        page_size=128,
        sm_scale=scales[3],
        q_scale=scales[0],
        k_scale=scales[1],
        v_scale=scales[2],
    )
    expected = _reference(*args, scales)

    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not is_sm90_supported(), reason="requires SM90 CUDA")
def test_native_decode_cuda_graph_replay_uses_updated_indices_and_scales():
    native = importlib.import_module(MODULE)
    q, k, v, req_to_token, slot_ids, seq_lens, topk_idx = _case(
        batch=1, num_q_heads=8, num_kv_heads=1
    )
    scales = (0.75, 1.25, 0.625, 128**-0.5)

    # Build the JIT module before capture.
    native.sgl_native_q8kv8_sparse_decode(
        q,
        k,
        v,
        req_to_token,
        slot_ids,
        seq_lens,
        topk_idx,
        128,
        128,
        scales[3],
        scales[0],
        scales[1],
        scales[2],
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native.sgl_native_q8kv8_sparse_decode(
            q,
            k,
            v,
            req_to_token,
            slot_ids,
            seq_lens,
            topk_idx,
            128,
            128,
            scales[3],
            scales[0],
            scales[1],
            scales[2],
        )

    topk_idx.fill_(-1)
    topk_idx[..., :3] = torch.tensor([1, 7, 13], device="cuda")
    graph.replay()
    expected = _reference(q, k, v, req_to_token, slot_ids, seq_lens, topk_idx, scales)
    torch.testing.assert_close(captured.float(), expected.float(), atol=3e-2, rtol=3e-2)


def test_native_decode_uses_consistent_fail_closed_provider_contract():
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import minimax_sparse_backend as backend
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse

    assert hasattr(envs, "SGLANG_ENABLE_MINIMAX_SGL_NATIVE_Q8KV8_DECODE")
    assert (
        "sgl_native_q8kv8_decode_strict"
        not in inspect.signature(minimax_sparse.minimax_sparse_decode).parameters
    )
    assert backend._native_q8kv8_decode_contract(
        is_npu=False,
        is_sm90=True,
        fp8_attn_gemm=False,
        main_pool_dtype=torch.float8_e4m3fn,
        block_size_k=128,
        page_size=128,
    )


def test_native_decode_quantizes_only_main_query_without_full_fp8_mode():
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        _quantize_sgl_native_decode_query,
    )

    q = torch.tensor([1.0, -0.5], dtype=torch.bfloat16)
    actual = _quantize_sgl_native_decode_query(q, enabled=True, q_scale=0.5)
    assert actual.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(actual.float(), torch.tensor([2.0, -1.0]))


def test_forward_decode_routes_main_query_through_native_fp8_quantization():
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxSparseAttnBackend,
    )

    source = inspect.getsource(MiniMaxSparseAttnBackend.forward_decode)
    assert "_quantize_sgl_native_decode_query(" in source
