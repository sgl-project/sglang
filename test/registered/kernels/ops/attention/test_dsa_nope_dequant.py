"""Group-scaled NoPE readers and the CUDA BF16 TileLang consumer.

Consumer tests use valid, fixed-width physical indices. Invalid-index attention
masking and KPool prefix/tail expansion are separate kernel contracts.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache,
    dequantize_k_cache_paged,
    dequantize_sparse_nope_cache,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires CUDA",
)


def make_keys(tokens, rope_dim=0):
    generator = torch.Generator(device="cuda").manual_seed(528 + rope_dim)
    nope = torch.randn(tokens, 1, 512, device="cuda", generator=generator)
    nope.view(tokens, 4, 128).mul_(
        torch.tensor([0.01, 0.5, 2.0, 8.0], device="cuda")[None, :, None]
    )
    rope = torch.randn(
        tokens, 1, rope_dim, device="cuda", generator=generator
    ).bfloat16()
    return nope.bfloat16(), rope


def make_cache(tokens, rope_dim=0):
    nope, rope = make_keys(tokens, rope_dim)
    packed, packed_rope = quantize_k_cache_separate(nope, rope)
    cache = torch.cat((packed, packed_rope), -1).view(torch.float8_e4m3fn)
    values = packed[:, :, :512].view(torch.float8_e4m3fn).float().view(tokens, 4, 128)
    scales = packed[:, :, 512:].view(torch.float32).view(tokens, 4, 1)
    expected = torch.cat(((values * scales).bfloat16().view(tokens, 1, 512), rope), -1)
    return cache, expected


def capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_full_and_paged_readers(rope_dim):
    cache, expected = make_cache(18, rope_dim)
    torch.testing.assert_close(dequantize_k_cache(cache), expected, rtol=0, atol=0)
    blocked = cache.view(2, 9, 1, -1)
    torch.testing.assert_close(
        dequantize_k_cache(blocked).view_as(expected), expected, rtol=0, atol=0
    )
    # Repeated physical indices and more selected entries than physical rows.
    pages = torch.tensor([17, 0, 3, 3, 8] * 4, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(
        dequantize_k_cache_paged(blocked, pages), expected[pages.long()], rtol=0, atol=0
    )
    graph, output = capture(lambda: dequantize_k_cache_paged(cache, pages))
    pages[0] = 7
    cache[:, :, 512:528].view(torch.float32).mul_(0.5)
    expected[:, :, :512].mul_(0.5)
    graph.replay()
    torch.testing.assert_close(output, expected[pages.long()], rtol=0, atol=0)


@pytest.mark.parametrize("pool_size", [7, 8, 9, 32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_sparse_reader_mapping_and_graph(pool_size, index_dtype):
    cache, expected = make_cache(pool_size)
    indices = torch.tensor(
        [[6, 2, 2, 1], [3, 0, 1, 2]], device="cuda", dtype=index_dtype
    ).t()
    assert not indices.is_contiguous()

    def check(output):
        kv, mapped = output
        assert kv.shape == (min(pool_size, indices.numel()), 1, 512)
        torch.testing.assert_close(
            kv[mapped.long()], expected[indices.long()], rtol=0, atol=0
        )

    check(dequantize_sparse_nope_cache(cache, indices))
    graph, output = capture(lambda: dequantize_sparse_nope_cache(cache, indices))
    indices[0, 0] = 5
    cache[:, :, 512:528].view(torch.float32).mul_(0.5)
    expected.mul_(0.5)
    graph.replay()
    check(output)


def test_selected_reader_preserves_invalid_mask_and_graph():
    cache, expected = make_cache(32)
    indices = torch.tensor(
        [[-1, 32, 7, 7], [0, -2, 31, 33]], device="cuda", dtype=torch.int32
    )

    def check(output):
        kv, mapped = output
        valid = (indices >= 0) & (indices < 32)
        assert (mapped[~valid] == -1).all()
        torch.testing.assert_close(
            kv[mapped[valid].long()], expected[indices[valid].long()], rtol=0, atol=0
        )

    check(dequantize_sparse_nope_cache(cache, indices))
    graph, output = capture(lambda: dequantize_sparse_nope_cache(cache, indices))
    # Both valid -> invalid and invalid -> valid must follow device-side updates.
    indices.copy_(torch.tensor([[31, 0, -1, 32], [-2, 7, 33, 7]], device="cuda"))
    graph.replay()
    check(output)


def test_full_reader_preserves_negative_mask():
    cache, expected = make_cache(4)
    indices = torch.tensor([[-1, 3, 0, -2, 3]], device="cuda", dtype=torch.int32)
    kv, mapped = dequantize_sparse_nope_cache(cache, indices)
    torch.testing.assert_close(kv, expected, rtol=0, atol=0)
    torch.testing.assert_close(mapped, indices, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 64), (3, 0)])
def test_empty_selection(shape):
    cache, _ = make_cache(32)
    indices = torch.empty(shape, device="cuda", dtype=torch.int32)
    kv, mapped = dequantize_sparse_nope_cache(cache, indices)
    assert kv.shape == (0, 1, 512)
    assert mapped.shape == indices.shape


@pytest.mark.parametrize(
    "width,dtype",
    [(512, torch.float8_e4m3fn), (656, torch.float8_e4m3fn), (528, torch.bfloat16)],
)
def test_sparse_reader_rejects_other_formats(width, dtype):
    cache = torch.empty((32, 1, width), device="cuda", dtype=dtype)
    indices = torch.zeros((1, 64), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="group-scaled FP8 NoPE KV"):
        dequantize_sparse_nope_cache(cache, indices)


@pytest.mark.parametrize("pool_size", [32, 512])
@pytest.mark.parametrize("queries", [1, 4])
def test_tilelang_consumer_and_graph(pool_size, queries):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("TileLang consumer integration is validated on SM90")
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    cache, expected_kv = make_cache(pool_size)
    q = torch.randn((queries, 16, 512), device="cuda", dtype=torch.bfloat16) * 0.05
    indices = (
        torch.arange(queries * 64, device="cuda", dtype=torch.int32).view(queries, 64)
        % pool_size
    )
    scale = 512**-0.5

    def run():
        return DeepseekSparseAttnBackend._forward_tilelang(
            None, q, cache, 512, indices, scale
        )

    def check(output):
        selected = expected_kv[indices.long(), 0].float()
        logits = torch.einsum("qhd,qkd->qhk", q.float(), selected) * scale
        expected = torch.einsum("qhk,qkd->qhd", logits.softmax(-1), selected)
        torch.testing.assert_close(
            output.view_as(q).float(), expected, rtol=0.02, atol=0.03
        )

    check(run())
    graph, output = capture(run)
    indices.copy_((indices + 3) % pool_size)
    q.mul_(0.5)
    graph.replay()
    check(output)


@pytest.mark.parametrize("pool_size", [128, 512])
def test_real_pool_write_move_and_consumer_graph(pool_size):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("TileLang consumer integration is validated on SM90")
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.runtime_context import get_parallel

    pool = MLATokenToKVPool(
        size=pool_size,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_dsa=True,
        override_kv_cache_dim=528,
    )
    keys, rope = make_keys(32)
    expected_bytes, expected_kv = make_cache(32)
    locs = torch.arange(1, 65, 2, device="cuda")
    layer = SimpleNamespace(layer_id=0)
    cache = pool.get_key_buffer(0)
    # Current main's writer resolves DCP ownership before storing group528 rows.
    with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
        pool.set_mla_kv_buffer(layer, locs, keys, rope)
    torch.testing.assert_close(
        cache.view(torch.uint8)[locs], expected_bytes.view(torch.uint8), rtol=0, atol=0
    )
    selection = torch.arange(256, device="cuda").view(4, 64) % 32
    indices = locs[selection].int()
    q = torch.randn((4, 16, 512), device="cuda", dtype=torch.bfloat16) * 0.05
    scale = 512**-0.5

    def run():
        return DeepseekSparseAttnBackend._forward_tilelang(
            None, q, cache, 512, indices, scale
        )

    def check(output):
        selected = expected_kv[selection, 0].float()
        logits = torch.einsum("qhd,qkd->qhk", q.float(), selected) * scale
        expected = torch.einsum("qhk,qkd->qhd", logits.softmax(-1), selected)
        torch.testing.assert_close(
            output.view_as(q).float(), expected, rtol=0.02, atol=0.03
        )

    check(run())
    graph, output = capture(run)
    # Relocation must retain scale words; stale source bytes must not be consumed.
    moved = locs + 64
    pool.move_kv_cache(moved, locs)
    pool.kv_buffer[0][locs] = 0
    indices.copy_(moved[selection])
    graph.replay()
    check(output)
    # A fresh writer update after capture must also reach the attention consumer.
    with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
        pool.set_mla_kv_buffer(layer, moved, keys * 0.5, rope)
    expected_kv.mul_(0.5)
    graph.replay()
    check(output)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
