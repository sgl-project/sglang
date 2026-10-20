"""DSpark cache writes and mixed target/draft NVFP4 dispatch on SM120."""

from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention import flash_mla_sm120 as sm120
from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace
from sglang.srt.mem_cache import deepseek_v4_memory_pool as pools


@pytest.fixture(autouse=True)
def clear_format_cache():
    pools.is_nvfp4_kv_cache.cache_clear()
    yield
    pools.is_nvfp4_kv_cache.cache_clear()


@pytest.mark.parametrize(
    "target,draft,expected",
    [
        ("fp8", "auto", (False, False)),
        ("nvfp4", "auto", (True, True)),
        ("nvfp4", "fp8", (True, False)),
        ("fp8", "nvfp4", (False, True)),
    ],
)
def test_target_and_draft_formats_are_independent(target, draft, expected):
    with mock.patch.object(pools, "envs") as envs, mock.patch.object(
        pools, "is_sm120_supported", return_value=True
    ):
        envs.SGLANG_SM120_KV_CACHE_FORMAT.get.return_value = target
        envs.SGLANG_SM120_DRAFT_KV_CACHE_FORMAT.get.return_value = draft
        assert (pools.is_nvfp4_kv_cache(), pools.is_nvfp4_kv_cache(True)) == expected


def test_draft_nvfp4_has_platform_gate():
    with mock.patch.object(pools, "envs") as envs, mock.patch.object(
        pools, "is_sm120_supported", return_value=False
    ):
        envs.SGLANG_SM120_KV_CACHE_FORMAT.get.return_value = "fp8"
        envs.SGLANG_SM120_DRAFT_KV_CACHE_FORMAT.get.return_value = "nvfp4"
        assert not pools.is_nvfp4_kv_cache()
        with pytest.raises(ValueError, match="SM120"):
            pools.is_nvfp4_kv_cache(True)


@pytest.mark.parametrize("fmt", ["fp8", "nvfp4"])
def test_attention_dispatch_uses_pool_format(fmt):
    # Both pools coexist in one worker; the global setting must not win.
    with mock.patch.object(
        sm120, "_sm120_default_backend", "flashinfer"
    ), mock.patch.object(
        sm120, "_sm120_kv_cache_format", "nvfp4" if fmt == "fp8" else "fp8"
    ), mock.patch.object(
        sm120, "_flash_mla_flashinfer"
    ) as fp8, mock.patch.object(
        sm120, "_flash_mla_flashinfer_nvfp4"
    ) as nvfp4:
        sm120.flash_mla_with_kvcache_sm120(
            q=torch.empty(1, 1, 64, 512),
            k_cache=None,
            indices=None,
            head_dim_v=512,
            kv_cache_format=fmt,
        )
        (nvfp4 if fmt == "nvfp4" else fp8).assert_called_once()
        (fp8 if fmt == "nvfp4" else nvfp4).assert_not_called()


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_nvfp4_cannot_fall_back_to_fp8_reader(backend):
    with mock.patch.object(sm120, "_sm120_default_backend", backend):
        with pytest.raises(ValueError, match="FlashInfer"):
            sm120.flash_mla_with_kvcache_sm120(
                q=torch.empty(1, 1, 64, 512),
                k_cache=None,
                indices=None,
                head_dim_v=512,
                kv_cache_format="nvfp4",
            )


def _pool(cache, nvfp4=True):
    # Real store methods with a small backing allocation; no server required.
    single = object.__new__(pools.DeepSeekV4SingleKVPool)
    single.is_nvfp4 = nvfp4
    single.nvfp4_page_size = 64
    single.page_size = 256
    single.kv_buffer = [cache]
    pool = object.__new__(pools.DeepSeekV4TokenToKVPool)
    pool.swa_kv_pool = single
    pool._stage_start = 3
    return pool


def _store(pool, kv, weight, freqs, positions, slots):
    pool.set_swa_key_buffer_radix_fused_norm_rope(
        3, slots, kv, weight, 1e-6, freqs, positions
    )


def test_fp8_store_stays_fused():
    cache = torch.empty(1, 256 * 584, dtype=torch.uint8)
    with mock.patch.object(
        pools, "fused_k_norm_rope_flashmla"
    ) as fp8, mock.patch.object(pools, "fused_norm_rope_inplace") as norm:
        _store(_pool(cache, nvfp4=False), None, None, None, None, None)
        fp8.assert_called_once()
        assert fp8.call_args.kwargs["kvcache"] is cache
        norm.assert_not_called()


sm120_required = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="requires SM120/SM121",
)


def _inputs(n):
    torch.manual_seed(20261019 + n)
    # Strided projection view, as in a fused QKV projection. Context lengths
    # can greatly exceed the five-row draft query block.
    storage = torch.randn(n, 640, dtype=torch.bfloat16, device="cuda")
    kv = storage[:, 128:]
    weight = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    angle = torch.randn(64, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angle), angle)
    positions = torch.arange(n, device="cuda", dtype=torch.int64) % 64
    slots = torch.arange(n, device="cuda", dtype=torch.int64) * 2 + 63
    slots[::3] = -1  # rejected/padded rows must not write to the cache
    pages = (2 * n + 128 + 63) // 64
    cache = torch.full((pages, 64, 384), 0x5A, dtype=torch.uint8, device="cuda")
    return kv, weight, freqs, positions, slots, cache


def _expected(kv, weight, freqs, positions, slots, cache):
    from flashinfer.mla import nvfp4_quantize_append_sparse_mla_cache

    # Independently pack only accepted rows. Exact bytes pin page boundaries,
    # invalid slots, scales, and preservation of untouched cache storage.
    normed = kv.clone(memory_format=torch.contiguous_format)
    fused_norm_rope_inplace(normed, weight, 1e-6, freqs, positions)
    expected = cache.clone()
    valid = slots >= 0
    nvfp4_quantize_append_sparse_mla_cache(normed[valid], slots[valid], expected)
    return expected


def _unpack_nvfp4(packed, scales):
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 7).long()] * torch.where(codes < 8, 1, -1)
    return values * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)


@sm120_required
@pytest.mark.parametrize("rows", [5, 257, 4096])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_dspark_norm_rope_append_cache_bytes(rows, slot_dtype):
    kv, w, freqs, pos, slots, cache = _inputs(rows)
    slots = slots.to(slot_dtype)
    original = kv.clone()
    expected = _expected(kv, w, freqs, pos, slots, cache)
    _store(_pool(cache), kv, w, freqs, pos, slots)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    torch.testing.assert_close(kv, original, rtol=0, atol=0)


@sm120_required
def test_dspark_store_graph_replays_after_larger_context_write():
    kv, w, freqs, pos, slots, cache = _inputs(5)
    pool = _pool(cache)
    _store(pool, kv, w, freqs, pos, slots)  # JIT outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _store(pool, kv, w, freqs, pos, slots)
    large = _inputs(4096)
    _store(_pool(large[-1]), *large[:-1])
    kv.mul_(0.5)
    expected = _expected(kv, w, freqs, pos, slots, cache)
    graph.replay()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@sm120_required
@pytest.mark.parametrize("rows", [5, 65])
def test_dspark_topk192_reader_matches_dequantized_reference(rows):
    from flashinfer import nvfp4_kv_quantize
    from flashinfer.mla import nvfp4_quantize_append_sparse_mla_cache

    torch.manual_seed(192 + rows)
    kv = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) / 10
    cache = torch.zeros(4, 64, 384, device="cuda", dtype=torch.uint8)
    nvfp4_quantize_append_sparse_mla_cache(kv, torch.arange(256, device="cuda"), cache)
    q = torch.randn(rows, 1, 64, 512, device="cuda", dtype=torch.bfloat16) / 10
    indices = (
        torch.arange(192, device="cuda", dtype=torch.int32)
        .expand(rows, 192)
        .contiguous()
    )
    lengths = torch.full((rows,), 133, device="cuda", dtype=torch.int32)
    sink = torch.zeros(64, device="cuda")

    def run():
        return sm120.flash_mla_with_kvcache_sm120(
            q=q,
            k_cache=cache,
            indices=indices[:, None],
            topk_length=lengths,
            attn_sink=sink,
            head_dim_v=512,
            kv_cache_format="nvfp4",
        )[0]

    out = run()
    # NVFP4 stores data then group scales within each page, not interleaved
    # 384-byte token records. Decode the ABI independently of the reader.
    raw = cache.reshape(4, -1)
    data = raw[:, : 64 * 352].reshape(256, 352)
    scales = raw[:, 64 * 352 :].reshape(256, 32)[:, :28].contiguous()
    nope = _unpack_nvfp4(data[:, :224], scales)
    rope = data[:, 224:].contiguous().view(torch.bfloat16).float()
    dequant = torch.cat((nope, rope), -1)[:133]
    q_flat = q.reshape(-1, 512)
    q_packed, q_scales = nvfp4_kv_quantize(
        q_flat[:, :448].contiguous(), torch.ones(1, device="cuda")
    )
    q_dequant = torch.cat(
        (_unpack_nvfp4(q_packed, q_scales), q_flat[:, 448:].float()), -1
    )
    scores = torch.einsum("bhd,kd->bhk", q_dequant.reshape(rows, 64, 512), dequant) * (
        512**-0.5
    )
    scores = torch.cat([scores, sink.view(1, 64, 1).expand(rows, 64, 1)], dim=-1)
    ref = torch.einsum("bhk,kd->bhd", scores.softmax(-1)[..., :-1], dequant)
    # Same tolerance as FlashInfer's dequantized NVFP4 attention reference:
    # P and the candidate-axis V are also quantized inside the kernel.
    torch.testing.assert_close(out[:, 0].float(), ref, atol=5e-2, rtol=5e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.testing.assert_close(captured, out, rtol=0, atol=0)
