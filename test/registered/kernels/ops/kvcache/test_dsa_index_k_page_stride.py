"""DSA index-K kernels honor the page stride of the cache they are given.

A caller may hand the index-K kernels a per-layer view ``(num_pages, page_bytes)``
whose pages are slices of larger per-page entries, so ``stride(0)`` exceeds the
page blob. Every writer and reader must address page ``p`` at ``p * stride(0)``,
write in place, and leave the bytes between blobs alone. Each case runs the same
operation on such a strided view and on a packed buffer and requires identical
blobs (or outputs), and untouched bytes outside the blobs. The boundary cases
put pages past 2 GiB from the buffer start, so page offsets overflow int32 unless
the kernels widen them, including when the page table or locations are int32.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
    can_use_dsa_fused_store,
    fused_store_index_k_cache,
)
from sglang.kernels.ops.kvcache.dsa_index_k_cache import _get_k_and_s_triton
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import (
    gather_fp4_index_k,
    store_fp4_index_k_cache,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# The DeepGEMM FP4 and CuTe DSL paged MQA logits readers need SM100.
register_cuda_ci(est_time=60, stage="nightly", runner_config="4-gpu-b200")

PAGE_SIZE = 64
HEAD_DIM = 128
FP8_PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)  # 8448
FP4_PAGE_BYTES = PAGE_SIZE * (HEAD_DIM // 2 + 4)  # 4352
SENTINEL = 0xA5
# Kimi-K3.1's DSA page entry: 24 layers x 64 tokens x 656 B fp8 latent rows,
# plus 24 x 8448 B index-K blobs, plus 1920 B padding.
LARGE_PAGE_STRIDE = 1_212_288
INT32_MAX = 2**31 - 1
# The first page whose offset at LARGE_PAGE_STRIDE exceeds int32.
FIRST_PAGE_PAST_INT32 = INT32_MAX // LARGE_PAGE_STRIDE + 1  # 1772


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if is_hip():
        pytest.skip("CUDA index-K kernels")


def _pages(num_pages: int, page_bytes: int, stride: int, offset: int, seed: int):
    """A packed buffer and a strided view with the same random blob contents;
    the strided view's surrounding bytes hold SENTINEL."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(
        0, 256, (num_pages, page_bytes), dtype=torch.uint8, device="cuda", generator=g
    )
    entries = torch.full(
        (num_pages, stride), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    strided = entries[:, offset : offset + page_bytes]
    strided.copy_(packed)
    assert strided.stride() == (stride, 1)
    return packed, entries, strided


def _assert_outside_untouched(entries: torch.Tensor, offset: int, page_bytes: int):
    before = entries[:, :offset]
    after = entries[:, offset + page_bytes :]
    assert (before == SENTINEL).all() and (after == SENTINEL).all(), (
        "bytes outside the blobs changed"
    )


def _loc(num_pages: int, num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(num_pages * PAGE_SIZE, device="cuda", generator=g)
    loc = perm[:num_tokens].to(torch.int64).contiguous()
    # Write the last page too, so the full page stride is exercised.
    last_page_slot = (num_pages - 1) * PAGE_SIZE + 5
    if not (loc == last_page_slot).any():
        loc[0] = last_page_slot
    return loc


def test_fused_store_index_k_cache_single_page():
    _skip_if_unavailable()
    if not can_use_dsa_fused_store(torch.bfloat16, torch.int64, PAGE_SIZE):
        pytest.skip("fused store JIT unavailable")
    num_pages = 1
    stride, offset = LARGE_PAGE_STRIDE, LARGE_PAGE_STRIDE - 3 * FP8_PAGE_BYTES
    packed, entries, strided = _pages(num_pages, FP8_PAGE_BYTES, stride, offset, 0)
    loc = _loc(num_pages, 50 * num_pages, 1)
    key = torch.randn(loc.numel(), HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    fused_store_index_k_cache(key, packed, loc, PAGE_SIZE)
    fused_store_index_k_cache(key, strided, loc, PAGE_SIZE)
    torch.cuda.synchronize()

    assert torch.equal(strided, packed)
    _assert_outside_untouched(entries, offset, FP8_PAGE_BYTES)


def test_fused_k_indexer_norm_rope_store():
    _skip_if_unavailable()
    from sglang.kernels.ops.attention.dsa.indexer_k import (
        fused_k_indexer_norm_rope_store,
    )

    num_pages = 5
    stride, offset = LARGE_PAGE_STRIDE, LARGE_PAGE_STRIDE - 3 * FP8_PAGE_BYTES
    packed, entries, strided = _pages(num_pages, FP8_PAGE_BYTES, stride, offset, 2)
    loc = _loc(num_pages, 200, 3)
    n = loc.numel()
    g = torch.Generator(device="cuda").manual_seed(4)
    # The key is a slice of the fused wk/weights projection output.
    kw = torch.randn(n, HEAD_DIM + 64, dtype=torch.bfloat16, device="cuda", generator=g)
    key = kw[:, :HEAD_DIM]
    weight = torch.randn(HEAD_DIM, device="cuda", generator=g)
    bias = torch.randn(HEAD_DIM, device="cuda", generator=g)
    cos_sin_cache = torch.randn(4096, 64, device="cuda", generator=g)
    positions = torch.randint(
        0, 4096, (n,), dtype=torch.int64, device="cuda", generator=g
    )

    for cache in (packed, strided):
        fused_k_indexer_norm_rope_store(
            key,
            cache,
            loc,
            weight,
            bias,
            1e-6,
            cos_sin_cache,
            positions,
            PAGE_SIZE,
        )
    torch.cuda.synchronize()

    assert torch.equal(strided, packed)
    _assert_outside_untouched(entries, offset, FP8_PAGE_BYTES)


@pytest.mark.parametrize("use_fp4_indexer", [False, True])
def test_index_key_cache_strided_layer_buffer(use_fp4_indexer):
    """The pool's index-K accessors, and the stores the indexer issues through
    them, keep working when a layer's buffer is a strided view."""
    _skip_if_unavailable()
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.runtime_context import get_parallel

    def make_pool():
        with get_parallel().override(attn_dcp_size=1):
            return DSATokenToKVPool(
                size=256,
                page_size=PAGE_SIZE,
                kv_lora_rank=512,
                dtype=torch.bfloat16,
                qk_rope_head_dim=64,
                layer_num=1,
                device="cuda",
                index_head_dim=HEAD_DIM,
                enable_memory_saver=False,
                kv_cache_dim=576,
                use_fp4_indexer=use_fp4_indexer,
            )

    ref_pool, pool = make_pool(), make_pool()
    page_bytes = FP4_PAGE_BYTES if use_fp4_indexer else FP8_PAGE_BYTES
    num_pages = ref_pool.index_key_cache.buffer[0].shape[0]
    assert ref_pool.index_key_cache.buffer[0].shape[1] == page_bytes
    offset = LARGE_PAGE_STRIDE - 2 * page_bytes
    packed, entries, strided = _pages(
        num_pages, page_bytes, LARGE_PAGE_STRIDE, offset, 11
    )
    ref_pool.index_key_cache.buffer[0] = packed
    pool.index_key_cache.buffer[0] = strided
    # The indexer writes through this accessor; it must hand out the view itself.
    assert pool.get_index_k_with_scale_buffer(0).data_ptr() == strided.data_ptr()

    slots = torch.arange(256, device="cuda", dtype=torch.int64)
    key = torch.randn(256, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    key_fp8 = key[:64].to(torch.float8_e4m3fn)
    key_scale = torch.rand(64, 1, device="cuda")
    for p in (ref_pool, pool):
        buf = p.get_index_k_with_scale_buffer(0)
        if use_fp4_indexer:
            store_fp4_index_k_cache(key, buf, slots, page_size=PAGE_SIZE, rne=True)
        else:
            fused_store_index_k_cache(key, buf, slots, PAGE_SIZE)
            # The non-fused fallback overwrites the second page.
            p.set_index_k_scale_buffer(0, slots[64:128], key_fp8, key_scale)
        source = torch.tensor([3, 70, 130], device="cuda")
        p.move_kv_cache(source.flip(0) + 64, source)

    lens = torch.tensor([256], dtype=torch.int32, device="cuda")
    table = torch.arange(4, dtype=torch.int32, device="cuda").unsqueeze(0)
    k_ref, s_ref = ref_pool.get_index_k_scale_buffer(0, lens, table, 256, 256)
    k_out, s_out = pool.get_index_k_scale_buffer(0, lens, table, 256, 256)
    torch.cuda.synchronize()

    assert torch.equal(strided, packed)
    assert torch.equal(k_out, k_ref) and torch.equal(s_out, s_ref)
    _assert_outside_untouched(entries, offset, page_bytes)


# A ~2.15 GB buffer whose last pages lie past int32 byte offsets.
BOUNDARY_NUM_PAGES = FIRST_PAGE_PAST_INT32 + 2
# A low page; the last page that starts below 2 GiB, whose blob lies above it;
# and two pages that start above it.
BOUNDARY_PAGES = (
    3,
    FIRST_PAGE_PAST_INT32 - 1,
    FIRST_PAGE_PAST_INT32,
    BOUNDARY_NUM_PAGES - 1,
)


def _boundary_pages(page_bytes: int, seed: int):
    assert FIRST_PAGE_PAST_INT32 * LARGE_PAGE_STRIDE > INT32_MAX
    offset = LARGE_PAGE_STRIDE - 3 * page_bytes
    assert (FIRST_PAGE_PAST_INT32 - 1) * LARGE_PAGE_STRIDE + offset > INT32_MAX
    packed, entries, strided = _pages(
        BOUNDARY_NUM_PAGES, page_bytes, LARGE_PAGE_STRIDE, offset, seed
    )
    return packed, entries, strided, offset


def _boundary_loc(dtype: torch.dtype) -> torch.Tensor:
    # A few slots in each boundary page, the first and last slot included.
    slots = [
        page * PAGE_SIZE + slot
        for page in BOUNDARY_PAGES
        for slot in (0, 17, PAGE_SIZE - 1)
    ]
    return torch.tensor(slots, dtype=dtype, device="cuda")


def test_get_k_and_s_past_int32_offsets():
    _skip_if_unavailable()
    page_bytes = FP8_PAGE_BYTES
    packed, entries, strided, offset = _boundary_pages(page_bytes, 20)
    p0, p1, p2, p3 = BOUNDARY_PAGES
    # An int32 page table, as the indexer builds it.
    page_table = torch.tensor(
        [[p2, p0, p3], [p1, p3, p2]], dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.tensor([150, 3 * PAGE_SIZE], dtype=torch.int64, device="cuda")

    outs = [
        _get_k_and_s_triton(
            buf=buf,
            page_indices=page_table,
            seq_lens=seq_lens,
            seq_len_sum=int(seq_lens.sum()),
            max_seq_len=int(seq_lens.max()),
            page_size=PAGE_SIZE,
            index_head_dim=HEAD_DIM,
        )
        for buf in (packed, strided)
    ]
    torch.cuda.synchronize()

    (k_ref, s_ref), (k_out, s_out) = outs
    assert torch.equal(k_out, k_ref)
    assert torch.equal(s_out, s_ref)
    _assert_outside_untouched(entries, offset, page_bytes)


@pytest.mark.parametrize("writer", ["fused_store", "fp4_store"])
def test_store_past_int32_offsets(writer):
    _skip_if_unavailable()
    if writer == "fused_store" and not can_use_dsa_fused_store(
        torch.bfloat16, torch.int64, PAGE_SIZE
    ):
        pytest.skip("fused store JIT unavailable")
    page_bytes = FP4_PAGE_BYTES if writer == "fp4_store" else FP8_PAGE_BYTES
    packed, entries, strided, offset = _boundary_pages(page_bytes, 21)
    # The FP4 writer takes int32 locations; the FP8 writer requires int64.
    loc = _boundary_loc(torch.int32 if writer == "fp4_store" else torch.int64)
    key = torch.randn(loc.numel(), HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    pages = list(BOUNDARY_PAGES)
    before = packed[pages].clone()

    for cache in (packed, strided):
        if writer == "fused_store":
            fused_store_index_k_cache(key, cache, loc, PAGE_SIZE)
        else:
            store_fp4_index_k_cache(key, cache, loc, page_size=PAGE_SIZE, rne=True)
    torch.cuda.synchronize()

    # Every written page changed, and every page, written or not, matches the
    # packed buffer at the same index.
    assert all(not torch.equal(a, b) for a, b in zip(packed[pages], before))
    assert torch.equal(strided, packed)
    _assert_outside_untouched(entries, offset, page_bytes)
    if writer == "fp4_store":
        ref = gather_fp4_index_k(packed, loc, page_size=PAGE_SIZE)
        out = gather_fp4_index_k(strided, loc, page_size=PAGE_SIZE)
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])


def _deep_gemm_or_skip(min_major: int):
    _skip_if_unavailable()
    try:
        import deep_gemm
    except ImportError:
        pytest.skip("DeepGEMM not installed")
    if torch.cuda.get_device_capability()[0] < min_major:
        pytest.skip("DeepGEMM paged MQA logits needs a newer GPU")
    return deep_gemm


def _paged_logits_case(num_pages: int, batch: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks_per_seq = num_pages // batch
    table = torch.randperm(num_pages, device="cuda", generator=g).to(torch.int32)
    table = table.view(batch, blocks_per_seq)
    lens = torch.randint(
        PAGE_SIZE,
        blocks_per_seq * PAGE_SIZE + 1,
        (batch, 1),
        dtype=torch.int32,
        device="cuda",
        generator=g,
    )
    max_len = blocks_per_seq * PAGE_SIZE
    visible = torch.arange(max_len, device="cuda")[None, :] < lens
    return table, lens, max_len, visible


def test_deep_gemm_fp8_paged_mqa_logits_reads_strided_view():
    """DeepGEMM receives the same (pages, 64, 1, 132) view the indexer builds;
    its scores over a strided cache must equal those over a packed one."""
    deep_gemm = _deep_gemm_or_skip(9)
    num_pages, batch, heads = 8, 2, 64
    keys = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM, device="cuda")
    packed = torch.empty(num_pages, FP8_PAGE_BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.arange(num_pages * PAGE_SIZE, device="cuda")
    fused_store_index_k_cache(keys.bfloat16(), packed, slots, PAGE_SIZE)
    offset = LARGE_PAGE_STRIDE - 4 * FP8_PAGE_BYTES
    _, entries, strided = _pages(
        num_pages, FP8_PAGE_BYTES, LARGE_PAGE_STRIDE, offset, 12
    )
    strided.copy_(packed)
    table, lens, max_len, visible = _paged_logits_case(num_pages, batch, 13)
    q = torch.randn(batch, 1, heads, HEAD_DIM, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, device="cuda")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens, PAGE_SIZE, num_sms)

    logits = []
    for buf in (packed, strided):
        kv = buf.view(num_pages, PAGE_SIZE, 1, HEAD_DIM + 4)
        assert kv.stride()[1:] == (HEAD_DIM + 4, HEAD_DIM + 4, 1)
        out = deep_gemm.fp8_paged_mqa_logits(
            q, kv, weights, lens, table, schedule, max_len, clean_logits=False
        )
        logits.append(torch.where(visible, out, 0))
    torch.cuda.synchronize()

    assert torch.equal(logits[1], logits[0])
    _assert_outside_untouched(entries, offset, FP8_PAGE_BYTES)


def test_deep_gemm_fp8_fp4_paged_mqa_logits_reads_strided_view():
    deep_gemm = _deep_gemm_or_skip(10)
    from sglang.kernels.ops.quantization.fp4_indexer_quant import (
        quantize_fp4_indexer_tensor,
    )

    num_pages, batch, heads = 8, 2, 64
    keys = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM, device="cuda")
    packed = torch.empty(num_pages, FP4_PAGE_BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.arange(num_pages * PAGE_SIZE, device="cuda")
    store_fp4_index_k_cache(
        keys.bfloat16(), packed, slots, page_size=PAGE_SIZE, rne=True
    )
    offset = LARGE_PAGE_STRIDE - 4 * FP4_PAGE_BYTES
    _, entries, strided = _pages(
        num_pages, FP4_PAGE_BYTES, LARGE_PAGE_STRIDE, offset, 14
    )
    strided.copy_(packed)
    table, lens, max_len, visible = _paged_logits_case(num_pages, batch, 15)
    q = torch.randn(batch, 1, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q, rne=True)
    q_fp4 = q_fp4.view(batch, 1, heads, HEAD_DIM // 2)
    q_sf = q_sf.view(batch, 1, heads)
    weights = torch.randn(batch, heads, device="cuda")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens, PAGE_SIZE, num_sms)

    logits = []
    for buf in (packed, strided):
        kv = buf.view(num_pages, PAGE_SIZE, 1, HEAD_DIM // 2 + 4)
        out = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            kv,
            weights,
            lens,
            table,
            schedule,
            max_len,
            clean_logits=False,
        )
        logits.append(torch.where(visible, out, 0))
    torch.cuda.synchronize()

    assert torch.equal(logits[1], logits[0])
    _assert_outside_untouched(entries, offset, FP4_PAGE_BYTES)


def test_cutedsl_paged_mqa_logits_reads_strided_view():
    deep_gemm = _deep_gemm_or_skip(10)
    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        pytest.skip("CuTe DSL paged MQA logits needs SM100")
    from sglang.kernels.ops.attention.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
    )

    num_pages, batch, heads = 8, 2, 64
    keys = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM, device="cuda")
    packed = torch.empty(num_pages, FP8_PAGE_BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.arange(num_pages * PAGE_SIZE, device="cuda")
    fused_store_index_k_cache(keys.bfloat16(), packed, slots, PAGE_SIZE)
    offset = LARGE_PAGE_STRIDE - 4 * FP8_PAGE_BYTES
    _, entries, strided = _pages(
        num_pages, FP8_PAGE_BYTES, LARGE_PAGE_STRIDE, offset, 16
    )
    strided.copy_(packed)
    table, lens, max_len, visible = _paged_logits_case(num_pages, batch, 17)
    q = torch.randn(batch, 1, heads, HEAD_DIM, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, device="cuda")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens, PAGE_SIZE, num_sms)

    logits = []
    for buf in (packed, strided):
        out = CuteDSLPagedMQALogitsRunner.forward(
            q,
            buf.view(num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
            weights,
            lens.view(-1),
            table,
            schedule,
            max_len,
        )
        logits.append(torch.where(visible, out, 0))
    torch.cuda.synchronize()

    assert torch.equal(logits[1], logits[0])
    _assert_outside_untouched(entries, offset, FP8_PAGE_BYTES)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
