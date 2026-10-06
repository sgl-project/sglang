import importlib.util
from pathlib import Path

import pytest
import torch


def _load_kvfp4_module():
    source = (
        Path(__file__).parents[5]
        / "python/sglang/srt/layers/quantization/kvfp4_tensor.py"
    )
    spec = importlib.util.spec_from_file_location("kvfp4_tensor_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


kvfp4 = _load_kvfp4_module()
MXFP4KVQuantizeUtil = kvfp4.MXFP4KVQuantizeUtil


def test_e2m1_tables_keep_legacy_magnitude_values():
    expected_magnitude = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32
    )
    expected_signed = torch.cat((expected_magnitude, -expected_magnitude))

    torch.testing.assert_close(torch.tensor(kvfp4.E2M1_VALUES[:8]), expected_magnitude)
    torch.testing.assert_close(torch.tensor(kvfp4.E2M1_VALUES), expected_signed)


def test_mxfp4_dequantize_decodes_signed_nibbles():
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 1, -1)
    scales = torch.full((1, 1, 1), 127, dtype=torch.uint8)

    restored = MXFP4KVQuantizeUtil.batched_dequantize(packed, scales)
    expected = torch.tensor(kvfp4.E2M1_VALUES).repeat(2).view(1, 1, 32)

    torch.testing.assert_close(restored.float(), expected)


def test_mxfp4_cpu_reference_and_paged_store():
    values = [
        0.0,
        0.24,
        0.26,
        0.74,
        0.76,
        1.24,
        1.26,
        1.74,
        1.76,
        2.49,
        2.51,
        3.49,
        3.51,
        4.99,
        5.01,
        6.1,
    ]
    source = torch.tensor(values * 2, dtype=torch.bfloat16).view(1, 1, 32)
    packed, scales = MXFP4KVQuantizeUtil.batched_quantize(source)
    restored = MXFP4KVQuantizeUtil.batched_dequantize(packed, scales)

    assert packed.shape == (1, 1, 16)
    assert scales.shape == (1, 1, 1)
    assert restored.shape == source.shape

    data_cache = torch.zeros((4, 1, 16), dtype=torch.uint8)
    scale_cache = torch.zeros((4, 1, 1), dtype=torch.uint8)
    tail_cache = torch.zeros((4, 1, 64), dtype=torch.bfloat16)
    tail = torch.arange(64, dtype=torch.bfloat16).view(1, 1, 64)
    MXFP4KVQuantizeUtil.quantize_and_store_paged(
        source,
        tail,
        torch.tensor([2], dtype=torch.int32),
        data_cache,
        scale_cache,
        tail_cache,
    )

    assert torch.equal(data_cache[2], packed[0])
    assert torch.equal(scale_cache[2], scales[0])
    assert torch.equal(tail_cache[2], tail[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_rows", [1, 17, 513])
def test_mxfp4_fused_cuda_store_matches_reference(num_rows):
    torch.manual_seed(7)
    source = torch.randn(
        (num_rows, 1, 512), device="cuda", dtype=torch.bfloat16
    )
    tail = torch.randn((num_rows, 1, 64), device="cuda", dtype=torch.bfloat16)
    locations = torch.randperm(num_rows + 8, device="cuda", dtype=torch.int64)[
        :num_rows
    ]
    data_cache = torch.zeros(
        (num_rows + 8, 1, 256), device="cuda", dtype=torch.uint8
    )
    scale_cache = torch.zeros(
        (num_rows + 8, 1, 16), device="cuda", dtype=torch.uint8
    )
    tail_cache = torch.zeros(
        (num_rows + 8, 1, 64), device="cuda", dtype=torch.bfloat16
    )

    MXFP4KVQuantizeUtil.quantize_and_store_paged(
        source, tail, locations, data_cache, scale_cache, tail_cache
    )
    expected_data, expected_scales = MXFP4KVQuantizeUtil._quantize_torch(source)

    assert torch.equal(data_cache[locations], expected_data)
    assert torch.equal(scale_cache[locations], expected_scales)
    assert torch.equal(tail_cache[locations], tail)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mxfp4_fused_cuda_store_graph_replay_keeps_addresses():
    num_rows = 17
    source = torch.randn(
        (num_rows, 1, 512), device="cuda", dtype=torch.bfloat16
    )
    tail = torch.randn((num_rows, 1, 64), device="cuda", dtype=torch.bfloat16)
    locations = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    row_cache = torch.zeros((num_rows, 1, 400), device="cuda", dtype=torch.uint8)
    data_cache = row_cache[..., :256]
    scale_cache = row_cache[..., 256:272]
    tail_cache = row_cache[..., 272:].view(torch.bfloat16)

    MXFP4KVQuantizeUtil.quantize_and_store_paged(
        source, tail, locations, data_cache, scale_cache, tail_cache
    )
    torch.cuda.synchronize()
    pointers = tuple(
        tensor.data_ptr() for tensor in (row_cache, data_cache, scale_cache, tail_cache)
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        MXFP4KVQuantizeUtil.quantize_and_store_paged(
            source, tail, locations, data_cache, scale_cache, tail_cache
        )

    source.copy_(source * 0.5)
    graph.replay()
    torch.cuda.synchronize()
    expected_data, expected_scales = MXFP4KVQuantizeUtil._quantize_torch(source)

    assert pointers == tuple(
        tensor.data_ptr() for tensor in (row_cache, data_cache, scale_cache, tail_cache)
    )
    assert torch.equal(data_cache, expected_data)
    assert torch.equal(scale_cache, expected_scales)
    assert torch.equal(tail_cache, tail)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="FlashMLA MXFP4 decode requires SM100",
)
@torch.inference_mode()
def test_flashmla_fp8_and_mxfp4_match_quantized_references_and_graph_replay():
    from flash_mla import dsv4_flash_mla_with_kvcache, dsv4_get_mla_metadata
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

    torch.manual_seed(11)
    batch_size = 2
    query_heads = 64
    topk = 128
    page_size = 64
    head_dim = 576
    value_dim = 512
    total_tokens = batch_size * topk

    query = (
        torch.randn(
            (batch_size, 1, query_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    latent = (
        torch.randn((total_tokens, 1, value_dim), device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    rope = (
        torch.randn(
            (total_tokens, 1, head_dim - value_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    kv_bf16 = torch.cat((latent, rope), dim=-1)
    indices = torch.arange(total_tokens, device="cuda", dtype=torch.int32).view(
        batch_size, 1, topk
    )
    cache_seqlens = torch.full(
        (batch_size,), topk, device="cuda", dtype=torch.int32
    )
    block_table = torch.empty((batch_size, 0), device="cuda", dtype=torch.int32)
    metadata, num_splits = dsv4_get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=query_heads,
        num_heads_k=1,
        num_heads_q=query_heads,
        is_fp8_kvcache=True,
        topk=topk,
    )

    fp8_cache = quantize_k_cache(
        kv_bf16.view(total_tokens // page_size, page_size, 1, head_dim)
    )
    fp8_reference = dequantize_k_cache(fp8_cache).view(total_tokens, 1, head_dim)

    mxfp4_rows = torch.empty(
        (total_tokens, 1, 400), device="cuda", dtype=torch.uint8
    )
    packed = mxfp4_rows[..., :256]
    scales = mxfp4_rows[..., 256:272]
    rope_cache = mxfp4_rows[..., 272:].view(torch.bfloat16)
    MXFP4KVQuantizeUtil.quantize_and_store_paged(
        latent,
        rope,
        torch.arange(total_tokens, device="cuda", dtype=torch.int32),
        packed,
        scales,
        rope_cache,
    )
    mxfp4_cache = mxfp4_rows.view(
        total_tokens // page_size, page_size, 1, 400
    )
    mxfp4_reference = torch.cat(
        (MXFP4KVQuantizeUtil.batched_dequantize(packed, scales), rope_cache), dim=-1
    )

    def reference(kv: torch.Tensor):
        outputs = []
        lses = []
        scale = head_dim**-0.5
        for batch_idx in range(batch_size):
            selected = kv[indices[batch_idx, 0].long(), 0].float()
            scores = query[batch_idx, 0].float() @ selected.transpose(0, 1)
            scores.mul_(scale)
            outputs.append(torch.softmax(scores, dim=-1) @ selected[:, :value_dim])
            lses.append(torch.logsumexp(scores, dim=-1))
        return torch.stack(outputs).unsqueeze(1), torch.stack(lses).unsqueeze(-1)

    for cache, quantized_reference in (
        (fp8_cache, fp8_reference),
        (mxfp4_cache, mxfp4_reference),
    ):
        output, lse = dsv4_flash_mla_with_kvcache(
            q=query,
            k_cache=cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=value_dim,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            indices=indices,
            is_fp8_kvcache=True,
        )
        expected_output, expected_lse = reference(quantized_reference)
        torch.testing.assert_close(output.float(), expected_output, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse.float(), expected_lse, atol=2e-3, rtol=2e-3)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_lse = dsv4_flash_mla_with_kvcache(
                q=query,
                k_cache=cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                head_dim_v=value_dim,
                tile_scheduler_metadata=metadata,
                num_splits=num_splits,
                indices=indices,
                is_fp8_kvcache=True,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, output, atol=0, rtol=0)
        torch.testing.assert_close(graph_lse, lse, atol=0, rtol=0)
