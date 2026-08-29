"""SM100 parity tests for SGLang's TRT-LLM-native NVFP4 KV layout."""

import math

import pytest
import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    NVFP4KVCacheMethod,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_sm100_supported(), reason="TRT-LLM native NVFP4 layout requires SM100"
)


@torch.inference_mode()
def test_nvfp4_native_layout_matches_flashinfer_reference():
    from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache

    torch.manual_seed(7)
    pages, heads, page_size, head_dim = 4, 4, 16, 128
    total_tokens = pages * page_size
    k_global_scale = torch.tensor([0.025], dtype=torch.float32, device="cuda")
    v_global_scale = torch.tensor([0.03125], dtype=torch.float32, device="cuda")

    k_nhd = torch.randn(
        pages, page_size, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    v_nhd = torch.randn_like(k_nhd)
    k_hnd = k_nhd.permute(0, 2, 1, 3).contiguous()
    v_hnd = v_nhd.permute(0, 2, 1, 3).contiguous()

    method = NVFP4KVCacheMethod(num_layers=1, device="cuda", page_size=page_size)
    method.configure_attention_backends("trtllm_mha", "trtllm_mha")
    buffers = method.create_buffers(
        total_tokens, heads, head_dim, layer_num=1, device="cuda"
    )

    # A non-monotonic scatter exercises page boundaries and every token mod-4
    # position used by TRT-LLM's V-scale interleave.
    loc = torch.randperm(total_tokens, device="cuda")
    method.quantize_and_store(
        buffers["k_buffer"][0],
        buffers["v_buffer"][0],
        buffers["k_scale_buffer"],
        buffers["v_scale_buffer"],
        loc,
        k_nhd.reshape(total_tokens, heads, head_dim)[loc],
        v_nhd.reshape(total_tokens, heads, head_dim)[loc],
        k_scale=k_global_scale,
        v_scale=v_global_scale,
        native_k_scale_buffer=buffers["native_k_scale_buffer"][0],
        native_v_scale_buffer=buffers["native_v_scale_buffer"][0],
    )
    torch.cuda.synchronize()

    (ref_k, ref_v), (ref_ks, ref_vs), _, _ = nvfp4_quantize_paged_kv_cache(
        k_hnd,
        v_hnd,
        kv_layout="HND",
        k_global_sf=1.0 / k_global_scale,
        v_global_sf=1.0 / v_global_scale,
    )

    got_k = (
        buffers["k_buffer"][0]
        .view(pages, page_size, heads, head_dim // 2)
        .permute(0, 2, 1, 3)
    )
    got_v = (
        buffers["v_buffer"][0]
        .view(pages, page_size, heads, head_dim // 2)
        .permute(0, 2, 1, 3)
    )
    got_ks = buffers["native_k_scale_buffer"][0].view(torch.float8_e4m3fn)
    got_vs = buffers["native_v_scale_buffer"][0].view(torch.float8_e4m3fn)

    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(got_ks.float(), ref_ks.float(), rtol=0, atol=0)
    torch.testing.assert_close(got_vs.float(), ref_vs.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "total_tokens,max_kv_len,page_table_width",
    [
        (64, 64, 1),
        # Mirror the Qwen3.5 server's short-prompt launch: the active sequence
        # occupies only part of one page, while the kernel receives the model's
        # full context limit and a correspondingly wide page-table stride.
        (26, 262144, 4096),
    ],
)
@torch.inference_mode()
def test_nvfp4_native_prefill_attention_matches_bf16_reference(
    total_tokens: int, max_kv_len: int, page_table_width: int
):
    """Exercise SGLang's writer and FlashInfer's context kernel together."""
    import flashinfer

    torch.manual_seed(11)
    pages, page_size = 1, 64
    q_heads, kv_heads, head_dim = 16, 2, 256
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    q = torch.randn(
        total_tokens, q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        total_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    v = torch.randn_like(k)

    method = NVFP4KVCacheMethod(num_layers=1, device="cuda", page_size=page_size)
    method.configure_attention_backends("trtllm_mha", "trtllm_mha")
    buffers = method.create_buffers(
        pages * page_size, kv_heads, head_dim, layer_num=1, device="cuda"
    )
    method.quantize_and_store(
        buffers["k_buffer"][0],
        buffers["v_buffer"][0],
        None,
        None,
        torch.arange(total_tokens, device="cuda"),
        k,
        v,
        k_scale=global_scale,
        v_scale=global_scale,
        native_k_scale_buffer=buffers["native_k_scale_buffer"][0],
        native_v_scale_buffer=buffers["native_v_scale_buffer"][0],
    )

    k_cache = (
        buffers["k_buffer"][0]
        .view(pages, page_size, kv_heads, head_dim // 2)
        .permute(0, 2, 1, 3)
    )
    v_cache = (
        buffers["v_buffer"][0]
        .view(pages, page_size, kv_heads, head_dim // 2)
        .permute(0, 2, 1, 3)
    )
    block_scales = (
        buffers["native_k_scale_buffer"][0].view(torch.float8_e4m3fn),
        buffers["native_v_scale_buffer"][0].view(torch.float8_e4m3fn),
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    out = torch.empty_like(q_fp8)
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=q_fp8,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=torch.zeros(
            256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        ),
        block_tables=torch.zeros(
            (1, page_table_width), dtype=torch.int32, device="cuda"
        ),
        seq_lens=torch.tensor([total_tokens], dtype=torch.int32, device="cuda"),
        max_q_len=total_tokens,
        max_kv_len=max_kv_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        batch_size=1,
        cum_seq_lens_q=torch.tensor(
            [0, total_tokens], dtype=torch.int32, device="cuda"
        ),
        cum_seq_lens_kv=torch.tensor(
            [0, total_tokens], dtype=torch.int32, device="cuda"
        ),
        out=out,
        kv_cache_sf=block_scales,
        causal=True,
    )

    # Compare against the same FP8 query and BF16 K/V before KV quantization.
    # The threshold mirrors FlashInfer's native NVFP4 attention regression.
    repeat = q_heads // kv_heads
    reference = (
        torch.nn.functional.scaled_dot_product_attention(
            q_fp8.bfloat16().transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    cosine = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), reference.float().reshape(-1), dim=0
    )
    assert cosine.item() > 0.86, f"native NVFP4 prefill cosine={cosine.item():.4f}"


@torch.inference_mode()
def test_nvfp4_native_target_verify_matches_bf16_reference():
    """Exercise the multi-query-token GenMHA path used by TARGET_VERIFY."""
    import flashinfer

    torch.manual_seed(17)
    page_size, prefix, verify_len = 32, 40, 4
    seq_len = prefix + verify_len
    pages = math.ceil(seq_len / page_size)
    q_heads, kv_heads, head_dim = 8, 2, 128
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    q = torch.randn(verify_len, q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)

    method = NVFP4KVCacheMethod(num_layers=1, device="cuda", page_size=page_size)
    method.configure_attention_backends("trtllm_mha", "trtllm_mha")
    buffers = method.create_buffers(
        pages * page_size, kv_heads, head_dim, layer_num=1, device="cuda"
    )
    method.quantize_and_store(
        buffers["k_buffer"][0],
        buffers["v_buffer"][0],
        None,
        None,
        torch.arange(seq_len, device="cuda"),
        k,
        v,
        k_scale=global_scale,
        v_scale=global_scale,
        native_k_scale_buffer=buffers["native_k_scale_buffer"][0],
        native_v_scale_buffer=buffers["native_v_scale_buffer"][0],
    )

    kv_cache = (
        buffers["k_buffer"][0]
        .view(pages, page_size, kv_heads, head_dim // 2)
        .permute(0, 2, 1, 3),
        buffers["v_buffer"][0]
        .view(pages, page_size, kv_heads, head_dim // 2)
        .permute(0, 2, 1, 3),
    )
    block_scales = (
        buffers["native_k_scale_buffer"][0].view(torch.float8_e4m3fn),
        buffers["native_v_scale_buffer"][0].view(torch.float8_e4m3fn),
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    out = torch.empty_like(q_fp8)
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q_fp8,
        kv_cache=kv_cache,
        workspace_buffer=torch.zeros(
            256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        ),
        block_tables=torch.arange(pages, dtype=torch.int32, device="cuda").view(
            1, pages
        ),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        max_seq_len=seq_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        out=out,
        kv_cache_sf=block_scales,
        q_len_per_req=verify_len,
    )

    repeat = q_heads // kv_heads
    k_ref = k.repeat_interleave(repeat, dim=1).permute(1, 0, 2).float()
    v_ref = v.repeat_interleave(repeat, dim=1).permute(1, 0, 2).float()
    q_ref = q_fp8.bfloat16().permute(1, 0, 2).float()
    scores = torch.einsum("hqd,hkd->hqk", q_ref, k_ref) / math.sqrt(head_dim)
    key_positions = torch.arange(seq_len, device="cuda").view(1, 1, -1)
    query_positions = (prefix + torch.arange(verify_len, device="cuda")).view(1, -1, 1)
    scores.masked_fill_(key_positions > query_positions, float("-inf"))
    reference = torch.einsum(
        "hqk,hkd->hqd", torch.softmax(scores, dim=-1), v_ref
    ).permute(1, 0, 2)

    cosine = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), reference.reshape(-1), dim=0
    )
    assert (
        cosine.item() > 0.86
    ), f"native NVFP4 target-verify cosine={cosine.item():.4f}"


@torch.inference_mode()
def test_nvfp4_native_scale_move_preserves_logical_rows():
    from sglang.srt.layers.quantization.nvfp4_kv_cache import (
        move_nvfp4_native_scales,
        nvfp4_v_scale_swizzle_indices,
    )

    pages, heads, page_size, scale_dim = 3, 2, 16, 8
    k_scale = (
        torch.arange(
            pages * heads * page_size * scale_dim,
            dtype=torch.int64,
            device="cuda",
        )
        .remainder(251)
        .to(torch.uint8)
        .view(pages, heads, page_size, scale_dim)
    )
    v_scale = torch.zeros_like(k_scale)

    # Seed V through the inverse logical mapping so its token rows have a clear
    # identity even though physical storage is interleaved.
    logical_v = (
        torch.arange(
            pages * page_size * heads * scale_dim,
            dtype=torch.int64,
            device="cuda",
        )
        .remainder(251)
        .to(torch.uint8)
        .view(pages * page_size, heads, scale_dim)
    )
    tokens = torch.arange(page_size, device="cuda")[:, None]
    scales = torch.arange(scale_dim, device="cuda")[None, :]
    sw_t, sw_s = nvfp4_v_scale_swizzle_indices(tokens, scales, scale_dim)
    for page in range(pages):
        for head in range(heads):
            v_scale[page, head, sw_t, sw_s] = logical_v[
                page * page_size : (page + 1) * page_size, head
            ]

    src = torch.tensor([1, 15, 16, 35], dtype=torch.int64, device="cuda")
    tgt = torch.tensor([46, 32, 31, 4], dtype=torch.int64, device="cuda")
    expected_k = k_scale.clone()
    expected_v = logical_v.clone()
    expected_k[tgt // page_size, :, tgt % page_size, :] = expected_k[
        src // page_size, :, src % page_size, :
    ]
    expected_v[tgt] = expected_v[src]

    move_nvfp4_native_scales(k_scale, v_scale, tgt, src)
    torch.cuda.synchronize()
    torch.testing.assert_close(k_scale, expected_k, rtol=0, atol=0)

    got_v = torch.empty_like(logical_v)
    for page in range(pages):
        for head in range(heads):
            got_v[page * page_size : (page + 1) * page_size, head] = v_scale[
                page, head, sw_t, sw_s
            ]
    torch.testing.assert_close(got_v, expected_v, rtol=0, atol=0)
