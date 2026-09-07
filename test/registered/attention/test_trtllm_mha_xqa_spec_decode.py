"""Real-kernel coverage for multi-token XQA speculative decode."""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"
PAGE_SIZE = 64
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
Q_LENS = (2, 3)
SEQ_LENS = (193, 257)
SHARED_PREFIX_LEN = 128
K_GLOBAL_SCALE = 0.75
V_GLOBAL_SCALE = 1.25


def _build_page_table():
    max_pages = math.ceil(max(SEQ_LENS) / PAGE_SIZE)
    num_shared_pages = SHARED_PREFIX_LEN // PAGE_SIZE
    shared_pages = list(range(num_shared_pages))
    next_page = num_shared_pages
    rows = []
    for seq_len in SEQ_LENS:
        num_pages = math.ceil(seq_len / PAGE_SIZE)
        num_unique_pages = num_pages - num_shared_pages
        unique_pages = list(range(next_page, next_page + num_unique_pages))
        next_page += num_unique_pages
        row = shared_pages + unique_pages
        row += [0] * (max_pages - len(row))
        rows.append(row)
    return (
        torch.tensor(rows, dtype=torch.int32, device=DEVICE),
        next_page,
    )


def _build_query_and_mask():
    total_q = sum(Q_LENS)
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    cu_seqlens_q = torch.tensor(
        [0, Q_LENS[0], total_q], dtype=torch.int32, device=DEVICE
    )
    mask = TRTLLMHAAttnBackend._build_xqa_causal_mask(
        num_tokens=total_q,
        max_q_len=max(Q_LENS),
        device=DEVICE,
        cu_seqlens_q=cu_seqlens_q,
    )
    return q, cu_seqlens_q, mask


def _reference(q, k_tokens, v_tokens, block_tables):
    outputs = []
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    q_offset = 0
    num_pages = k_tokens.shape[0] // PAGE_SIZE
    k_pages = k_tokens.view(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    v_pages = v_tokens.view(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    for req, (q_len, seq_len) in enumerate(zip(Q_LENS, SEQ_LENS)):
        num_request_pages = math.ceil(seq_len / PAGE_SIZE)
        page_ids = block_tables[req, :num_request_pages].long()
        k = (
            k_pages[page_ids]
            .permute(2, 0, 1, 3)
            .reshape(NUM_KV_HEADS, -1, HEAD_DIM)[:, :seq_len]
        )
        v = (
            v_pages[page_ids]
            .permute(2, 0, 1, 3)
            .reshape(NUM_KV_HEADS, -1, HEAD_DIM)[:, :seq_len]
        )
        k = k.repeat_interleave(group_size, dim=0).float()
        v = v.repeat_interleave(group_size, dim=0).float()
        request_q = q[q_offset : q_offset + q_len].permute(1, 0, 2).float()
        scores = torch.einsum("hqd,hkd->hqk", request_q, k) / math.sqrt(HEAD_DIM)
        kv_positions = torch.arange(seq_len, device=DEVICE).view(1, 1, -1)
        prefix_len = seq_len - q_len
        q_positions = (prefix_len + torch.arange(q_len, device=DEVICE)).view(1, -1, 1)
        scores.masked_fill_(kv_positions > q_positions, float("-inf"))
        output = torch.einsum("hqk,hkd->hqd", torch.softmax(scores, dim=-1), v)
        outputs.append(output.permute(1, 0, 2))
        q_offset += q_len
    return torch.cat(outputs, dim=0).to(torch.bfloat16)


def _make_backend(workspace):
    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.is_xqa_impl = True
    backend.workspace_buffer = workspace
    backend.max_context_len = max(SEQ_LENS)
    backend.page_size = PAGE_SIZE
    return backend


def _reshape_paged_cache(backend, k_tokens, v_tokens, head_dim):
    layer = SimpleNamespace(
        tp_k_head_num=NUM_KV_HEADS,
        tp_v_head_num=NUM_KV_HEADS,
    )
    return backend._reshape_paged_kv_cache(k_tokens, v_tokens, layer, head_dim)


def _run_ragged(backend, q, kv_cache, block_tables, cu_seqlens_q, mask, **kwargs):
    bmm1_scale = kwargs.pop("bmm1_scale", 1.0 / math.sqrt(HEAD_DIM))
    bmm2_scale = kwargs.pop("bmm2_scale", 1.0)
    return backend._run_ragged_q_decode(
        query=q,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32, device=DEVICE),
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=-1,
        sinks=None,
        max_q_len=max(Q_LENS),
        cu_seqlens_q=cu_seqlens_q,
        mask=mask,
        **kwargs,
    )


@unittest.skipUnless(
    is_sm90_supported() or is_sm120_supported(),
    "XQA speculative decode requires SM90 or SM120",
)
class TestXQASpecDecode(CustomTestCase):
    def test_fp8_ragged_mask_matches_paged_reference(self):
        torch.manual_seed(7)
        block_tables, num_pages = _build_page_table()
        q, cu_seqlens_q, mask = _build_query_and_mask()
        k_tokens = torch.randn(
            num_pages * PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=DEVICE,
        ).to(torch.float8_e4m3fn)
        v_tokens = torch.randn_like(k_tokens, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        backend = _make_backend(workspace)
        kv_cache = _reshape_paged_cache(backend, k_tokens, v_tokens, HEAD_DIM)

        output = _run_ragged(
            backend,
            q,
            kv_cache,
            block_tables,
            cu_seqlens_q,
            mask,
        )
        expected = _reference(q, k_tokens, v_tokens, block_tables)
        torch.testing.assert_close(
            output.float(), expected.float(), atol=6e-2, rtol=6e-2
        )

    @unittest.skipUnless(
        is_sm120_supported(), "native NVFP4 XQA coverage requires SM120"
    )
    def test_nvfp4_ragged_mask_with_shared_prefix_pages(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        torch.manual_seed(11)
        block_tables, num_pages = _build_page_table()
        q, cu_seqlens_q, mask = _build_query_and_mask()
        quant_method = NVFP4KVCacheMethod(num_layers=1, device=DEVICE)
        num_tokens = num_pages * PAGE_SIZE
        buffers = quant_method.create_buffers(
            num_tokens, NUM_KV_HEADS, HEAD_DIM, 1, DEVICE
        )
        k_source = torch.randn(
            num_tokens,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        v_source = torch.randn_like(k_source)
        locations = torch.arange(num_tokens, device=DEVICE)
        quant_method.k_scales_gpu.fill_(K_GLOBAL_SCALE)
        quant_method.v_scales_gpu.fill_(V_GLOBAL_SCALE)
        quant_method.k_scales_float[0] = K_GLOBAL_SCALE
        quant_method.v_scales_float[0] = V_GLOBAL_SCALE
        quant_method.quantize_and_store(
            buffers["k_buffer"][0],
            buffers["v_buffer"][0],
            buffers["k_scale_buffer"][0],
            buffers["v_scale_buffer"][0],
            locations,
            k_source,
            v_source,
            k_scale=quant_method.k_scales_gpu[:1],
            v_scale=quant_method.v_scales_gpu[:1],
        )

        k_fp4 = buffers["k_buffer"][0]
        v_fp4 = buffers["v_buffer"][0]
        k_scales = buffers["k_scale_buffer"][0].view(torch.float8_e4m3fn)
        v_scales = buffers["v_scale_buffer"][0].view(torch.float8_e4m3fn)
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        k_reference = NVFP4KVQuantizeUtil.dequantize(
            k_fp4, k_scales, quant_method.k_scales_gpu[:1]
        )
        v_reference = NVFP4KVQuantizeUtil.dequantize(
            v_fp4, v_scales, quant_method.v_scales_gpu[:1]
        )
        workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        backend = _make_backend(workspace)
        kv_cache = _reshape_paged_cache(backend, k_fp4, v_fp4, HEAD_DIM // 2)
        kv_cache_sf = _reshape_paged_cache(backend, k_scales, v_scales, HEAD_DIM // 16)

        output = _run_ragged(
            backend,
            q,
            kv_cache,
            block_tables,
            cu_seqlens_q,
            mask,
            bmm1_scale=K_GLOBAL_SCALE / math.sqrt(HEAD_DIM),
            bmm2_scale=V_GLOBAL_SCALE,
            kv_cache_sf=kv_cache_sf,
        )
        expected = _reference(q, k_reference, v_reference, block_tables)
        torch.testing.assert_close(
            output.float(), expected.float(), atol=1e-1, rtol=1e-1
        )


if __name__ == "__main__":
    unittest.main()
