"""HIP counterpart of ``test_fp4_indexer.py`` for the AITER FP4 DeepSeek-V4 indexer.

The CUDA path exposes the quantizer, the cache store and the fused
norm/RoPE/store as separate Triton entry points. On HIP all three collapse into
``aiter_k_indexer_fp4_cache_write``, and the query side into
``aiter_q_indexer_fp4``, so the tests below mirror the CUDA file's four cases
through those two ops. The FP4 grid and the UE8M0 scale rule are identical on
both targets, so the reference helpers are shared verbatim.

Two layout details differ from CUDA and are pinned here because nothing else
checks them: the K cache keeps payload and scale in separate buffers with the
scale token axis shuffled, and the Q scale is emitted preshuffled into the
logits kernel's ABI layout.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    compress_norm_rope_store,
)
from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    FP4KWriteMetadata,
    _decode_cta_count,
    _guard_page_table,
    aiter_fp4_paged_mqa_logits,
    aiter_k_indexer_fp4_cache_write,
    aiter_q_indexer_fp4,
    prepare_fp4_decode_workspace,
    prepare_fp4_k_write_metadata,
    prepare_fp4_prefill_workspace,
)
from sglang.srt.utils import get_device, is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()),
    reason="The FP4 indexer adapters wrap AITER CDNA4 (gfx95x) kernels.",
)

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
GROUP_SIZE = 32
SCALE_GROUPS = HEAD_DIM // GROUP_SIZE
PAGE_SIZE = 64
E2M1_MAX = 6.0
NUM_HEADS = 64
ROPE_DIM = 64
NORM_EPS = 1.0e-6
# Tokens per group along the shuffled scale axis; see _ref_store_fp4_index_cache.
SCALE_SHUFFLE_TILE = 16

_E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ceil_ue8m0_exp_ref(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _fp4_e2m1_code_ref(x: torch.Tensor) -> torch.Tensor:
    ax = torch.minimum(x.abs(), torch.tensor(E2M1_MAX, device=x.device))
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (ax > threshold).to(torch.uint8)
    sign = ((x < 0) & (idx != 0)).to(torch.uint8) * 8
    return idx | sign


def _ref_quantize_fp4_indexer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nibble-packed E2M1 payload and per-group UE8M0 exponents.

    Same rule as the CUDA reference, except the exponents stay one byte per
    group instead of being packed into an int32: the HIP cache stores them in a
    separate buffer, one byte per (group, token).
    """
    x = x.contiguous().view(-1, HEAD_DIM).float()
    groups = x.view(-1, SCALE_GROUPS, GROUP_SIZE)
    scale_raw = (groups.abs().amax(dim=-1) / E2M1_MAX).clamp_min(1.0e-4)
    scale_exp = _ceil_ue8m0_exp_ref(scale_raw)
    scale = (scale_exp << 23).contiguous().view(torch.float32)

    scaled = (groups / scale.unsqueeze(-1)).view(-1, HEAD_DIM)
    code = _fp4_e2m1_code_ref(scaled)
    packed = (code[:, 0::2].to(torch.int16) | (code[:, 1::2].to(torch.int16) << 4)).to(
        torch.uint8
    )
    return packed, scale_exp.to(torch.uint8)


def _canonical_zero(packed: torch.Tensor) -> torch.Tensor:
    """Fold negative zero onto positive zero in both nibbles.

    AITER keeps the sign bit when a lane quantizes to zero magnitude, while the
    reference clears it. Both decode to 0.0, so normalize before comparing the
    packed bytes.
    """
    lo, hi = packed & 0x0F, packed >> 4
    zero = torch.zeros_like(lo)
    lo = torch.where(lo == 0x8, zero, lo)
    hi = torch.where(hi == 0x8, zero, hi)
    return lo | (hi << 4)


def _ref_dequantize_fp4_indexer(
    packed: torch.Tensor, scale_exp: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`_ref_quantize_fp4_indexer`, for value comparisons."""
    packed = packed.reshape(-1, FP4_DIM)
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).long()
    values = torch.tensor(_E2M1_GRID + [-v for v in _E2M1_GRID], device=packed.device)[
        codes.reshape(-1, SCALE_GROUPS, GROUP_SIZE)
    ]
    factor = (scale_exp.reshape(-1, SCALE_GROUPS).to(torch.int32) << 23).view(
        torch.float32
    )
    return (values * factor.unsqueeze(-1)).reshape(-1, HEAD_DIM)


def _empty_index_k_cache(num_pages: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the split buffers ``uses_aiter_fp4_layout`` creates per layer."""
    payload = torch.zeros(
        (num_pages, 1, SCALE_GROUPS, PAGE_SIZE, GROUP_SIZE // 2),
        dtype=torch.uint8,
        device=get_device(),
    ).view(torch.float4_e2m1fn_x2)
    scale = torch.zeros(
        (num_pages, 1, SCALE_GROUPS, PAGE_SIZE), dtype=torch.uint8, device=get_device()
    )
    return payload, scale


def _ref_store_fp4_index_cache(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the payload and scale buffers a correct writer would produce.

    Payload rows stay in slot order. The scale buffer stores each page's tokens
    as the transpose of a ``SCALE_SHUFFLE_TILE x 4`` tile, so token ``t`` lands
    at ``(t % 16) * 4 + t // 16``. Rows whose ``loc`` is negative are skipped and
    must be left at zero.
    """
    payload = torch.zeros(
        (num_pages, 1, SCALE_GROUPS, PAGE_SIZE, GROUP_SIZE // 2),
        dtype=torch.uint8,
        device=x_fp4.device,
    )
    scale = torch.zeros(
        (num_pages, 1, SCALE_GROUPS, PAGE_SIZE), dtype=torch.uint8, device=x_fp4.device
    )
    for token_id in range(x_fp4.shape[0]):
        cache_loc = int(loc[token_id].item())
        if cache_loc < 0:
            continue
        page, offset = divmod(cache_loc, PAGE_SIZE)
        shuffled = (offset % SCALE_SHUFFLE_TILE) * 4 + offset // SCALE_SHUFFLE_TILE
        for group in range(SCALE_GROUPS):
            lo = group * (GROUP_SIZE // 2)
            payload[page, 0, group, offset] = x_fp4[token_id, lo : lo + GROUP_SIZE // 2]
            scale[page, 0, group, shuffled] = x_sf[token_id, group]
    return payload, scale


def _read_index_k_cache(payload, scale, loc: torch.Tensor):
    """Gather ``loc`` back out of the cache as (packed nibbles, exponents)."""
    page, offset = loc // PAGE_SIZE, loc % PAGE_SIZE
    shuffled = (offset % SCALE_SHUFFLE_TILE) * 4 + offset // SCALE_SHUFFLE_TILE
    packed = payload.view(torch.uint8)[page, 0, :, offset].reshape(-1, FP4_DIM)
    return packed, scale[page, 0, :, shuffled]


def _rope_tables(
    max_pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The bf16 cos/sin pair ``DeepseekV4AttentionMLA`` hands the FP4 adapters."""
    freqs_cis = precompute_freqs_cis(ROPE_DIM, max_pos, 0, 10000, 1, 32, 1).to(
        get_device()
    )
    return (
        freqs_cis.real.to(torch.bfloat16),
        freqs_cis.imag.to(torch.bfloat16),
        freqs_cis,
    )


def _ref_apply_rope(x, cos, sin, positions: torch.Tensor) -> torch.Tensor:
    """Interleaved (non-neox) RoPE over the trailing ``ROPE_DIM`` lanes."""
    shape = (positions.shape[0], *(1,) * (x.dim() - 2), ROPE_DIM // 2)
    c = cos.float()[positions].reshape(shape)
    s = sin.float()[positions].reshape(shape)
    pairs = x[..., ROPE_DIM:].reshape(*x.shape[:-1], ROPE_DIM // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([even * c - odd * s, even * s + odd * c], dim=-1)
    return torch.cat([x[..., :ROPE_DIM], rotated.flatten(-2)], dim=-1)


def _ref_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard, the rotation ``do_rotate_act`` applies.

    Written out in torch instead of reusing ``ops.quantization.hadamard``
    because that kernel is CUDA-only and does not build under ROCm.
    """
    h = torch.ones(1, 1, device=get_device(), dtype=torch.float32)
    while h.shape[0] < HEAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return x.float() @ (h * HEAD_DIM**-0.5)


def _ref_k_transform(k, norm_weight, cos, sin, positions) -> torch.Tensor:
    x = k.float()
    x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + NORM_EPS)
    x = x * norm_weight.float()
    return _ref_hadamard(_ref_apply_rope(x, cos, sin, positions))


def _ref_q_transform(q, cos, sin, positions) -> torch.Tensor:
    return _ref_hadamard(_ref_apply_rope(q.float(), cos, sin, positions))


def _write_index_k_cache(
    k, norm_weight, cos, sin, positions, loc, payload, scale
) -> None:
    aiter_k_indexer_fp4_cache_write(
        k=k,
        norm_weight=norm_weight,
        norm_epsilon=NORM_EPS,
        cos=cos,
        sin=sin,
        plan=None,
        out_loc=None,
        k_payload=payload,
        k_scale=scale,
        write_metadata=FP4KWriteMetadata(positions, loc),
    )


# ---------------------------------------------------------------------------
# The four cases mirrored from the CUDA file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 7, 96])
def test_quantize_fp4_indexer_tensor(num_tokens: int) -> None:
    """The fused writer's quantization matches the shared FP4 reference exactly."""
    torch.manual_seed(num_tokens)
    cos, sin, _ = _rope_tables(512)
    payload, scale = _empty_index_k_cache(-(-num_tokens // PAGE_SIZE) + 1)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    x[0, :8] = torch.tensor(
        [-8.0, -6.0, -3.0, -1.5, 0.0, 0.5, 2.0, 8.0],
        device=get_device(),
        dtype=torch.bfloat16,
    )
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=get_device(), dtype=torch.int64) * 3
    loc = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)

    _write_index_k_cache(x, norm_weight, cos, sin, positions, loc, payload, scale)

    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(
        _ref_k_transform(x, norm_weight, cos, sin, positions)
    )
    stored_fp4, stored_sf = _read_index_k_cache(payload, scale, loc)
    torch.testing.assert_close(stored_sf, ref_sf)
    torch.testing.assert_close(_canonical_zero(stored_fp4), _canonical_zero(ref_fp4))


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_index_cache_store_layout(num_tokens: int) -> None:
    """Scattered slots land in the paged layout and touch nothing else."""
    torch.manual_seed(num_tokens + 50)
    cos, sin, _ = _rope_tables(512)
    num_pages = max(2, -(-num_tokens // PAGE_SIZE) + 1)
    payload, scale = _empty_index_k_cache(num_pages)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=get_device(), dtype=torch.int64) * 3
    loc = torch.randperm(num_pages * PAGE_SIZE, device=get_device())[:num_tokens].to(
        torch.int64
    )
    # A masked row must be dropped rather than written to some default slot.
    loc[num_tokens // 2] = -1

    _write_index_k_cache(x, norm_weight, cos, sin, positions, loc, payload, scale)

    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(
        _ref_k_transform(x, norm_weight, cos, sin, positions)
    )
    expected_payload, expected_scale = _ref_store_fp4_index_cache(
        ref_fp4, ref_sf, loc, num_pages
    )
    torch.testing.assert_close(
        _canonical_zero(payload.view(torch.uint8)), _canonical_zero(expected_payload)
    )
    torch.testing.assert_close(scale, expected_scale)


# 17 and 33 straddle the 16-token scale shuffle tile, 65 the 64-token page.
@pytest.mark.parametrize("num_tokens", [1, 16, 17, 33, 65, 96])
def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -> None:
    """The real ``compress_norm_rope_store`` entry point, plan and metadata included."""
    torch.manual_seed(num_tokens + 100)
    num_pages = -(-num_tokens // PAGE_SIZE) + 1
    compress_ratio = 4
    kv = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    seq_lens = (
        torch.arange(1, num_tokens + 1, device=get_device(), dtype=torch.int64)
        * compress_ratio
    )
    req_pool_indices = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio, req_pool_indices, seq_lens
    )
    loc = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    rope_len = int(seq_lens.max().item()) + 1
    cos, sin, freqs_cis = _rope_tables(rope_len)
    payload, scale = _empty_index_k_cache(num_pages)

    metadata = prepare_fp4_k_write_metadata(plan, loc, rope_len)
    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        freq_cis=freqs_cis,
        out_loc=loc,
        kvcache=payload,
        page_size=PAGE_SIZE,
        use_fp4=True,
        kvcache_scale=scale,
        rope_cache=(cos, sin),
        fp4_k_write_metadata=metadata,
    )

    # The plan drives RoPE off the compression boundary, not the token index.
    torch.testing.assert_close(metadata.positions, seq_lens - compress_ratio)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(
        _ref_k_transform(kv, norm_weight, cos, sin, metadata.positions)
    )
    expected_payload, expected_scale = _ref_store_fp4_index_cache(
        ref_fp4, ref_sf, metadata.slots, num_pages
    )
    torch.testing.assert_close(
        _canonical_zero(payload.view(torch.uint8)), _canonical_zero(expected_payload)
    )
    torch.testing.assert_close(scale, expected_scale)


@pytest.mark.parametrize("batch_size", [1, 5, 17])
def test_fp4_fused_q_indexer_rope_hadamard_quant(batch_size: int) -> None:
    torch.manual_seed(batch_size + 200)
    cos, sin, _ = _rope_tables(256)
    q = torch.randn(
        batch_size, NUM_HEADS, HEAD_DIM, device=get_device(), dtype=torch.bfloat16
    )
    positions = (
        torch.arange(batch_size, device=get_device(), dtype=torch.int64) * 7
    ) % 63

    q_fp4, q_sf = aiter_q_indexer_fp4(q.contiguous(), cos, sin, positions)

    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(
        _ref_q_transform(q, cos, sin, positions)
    )
    torch.testing.assert_close(
        _canonical_zero(q_fp4.view(torch.uint8).reshape(-1, FP4_DIM)),
        _canonical_zero(ref_fp4),
    )
    # Unlike the CUDA path, the scales are emitted already preshuffled into the
    # logits kernel's ABI: heads split as (m_tiles, 16) and moved behind the
    # group axis. Nothing downstream reorders them, and every candidate layout
    # shares the tensor's shape, so a regression here is silent.
    m_tiles, k_tiles = NUM_HEADS // 16, HEAD_DIM // 128
    expected_sf = (
        ref_sf.reshape(batch_size, m_tiles, 16, k_tiles, SCALE_GROUPS)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    )
    torch.testing.assert_close(q_sf.reshape(expected_sf.shape), expected_sf)


# ---------------------------------------------------------------------------
# HIP-only surface: schedule bookkeeping and the paged MQA logits kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("logical_width", [1, 3, 4, 5, 8, 33])
def test_guard_page_table_pads_to_schedule_granularity(logical_width: int) -> None:
    rows = 3
    page_table = torch.arange(
        1, rows * logical_width + 1, device=get_device(), dtype=torch.int32
    ).reshape(rows, logical_width)

    guarded, max_seq_len = _guard_page_table(page_table)

    padded_width = max(4, -(-logical_width // 4) * 4)
    assert guarded.shape == (rows, padded_width + 4)
    assert guarded.dtype is torch.int32
    assert max_seq_len == padded_width * PAGE_SIZE
    torch.testing.assert_close(guarded[:, :logical_width], page_table)
    assert guarded[:, logical_width:].eq(0).all()


def test_guard_page_table_refreshes_reused_buffer() -> None:
    rows, width = 2, 6
    first = torch.arange(rows * width, device=get_device(), dtype=torch.int32).reshape(
        rows, width
    )
    guarded, _ = _guard_page_table(first)

    refreshed, _ = _guard_page_table(first + 100, out=guarded)

    assert refreshed.data_ptr() == guarded.data_ptr()
    torch.testing.assert_close(refreshed[:, :width], first + 100)
    assert refreshed[:, width:].eq(0).all()


@pytest.mark.parametrize(
    "num_queries,max_seq_len", [(1, 256), (8, 4096), (512, 256), (4096, 65536)]
)
def test_decode_cta_count_stays_within_available_chunks(
    num_queries: int, max_seq_len: int
) -> None:
    chunks_per_seq = max(1, -(-max_seq_len // 256))

    cta_count = _decode_cta_count(num_queries, max_seq_len)

    assert 1 <= cta_count <= num_queries * chunks_per_seq
    assert cta_count <= max(1024, num_queries * 4)


def _decode_plan(seq_lens: torch.Tensor, compress_ratio: int) -> CompressorDecodePlan:
    """Hand-build the 16-byte decode plan rows the metadata builder reads."""
    words = torch.zeros((seq_lens.shape[0], 4), dtype=torch.int32, device=get_device())
    words[:, 0] = seq_lens.to(torch.int32)
    return CompressorDecodePlan(compress_ratio, words.view(torch.uint8))


def _prefill_plan(
    seq_lens: torch.Tensor, ragged_ids: torch.Tensor, compress_ratio: int
) -> CompressorPrefillPlan:
    words = torch.zeros((seq_lens.shape[0], 4), dtype=torch.int32, device=get_device())
    words[:, 0] = seq_lens.to(torch.int32)
    words[:, 1] = ragged_ids.to(torch.int32)
    return CompressorPrefillPlan(
        compress_ratio,
        words.view(torch.uint8),
        torch.zeros((seq_lens.shape[0], 8), dtype=torch.uint8, device=get_device()),
    )


def test_k_write_metadata_decode_masks_unaligned_and_out_of_range() -> None:
    compress_ratio, rope_len = 4, 4096
    seq_lens = torch.tensor(
        [8, 9, 0, rope_len + compress_ratio], device=get_device(), dtype=torch.int64
    )
    out_loc = torch.tensor([11, 22, 33, 44], device=get_device(), dtype=torch.int64)

    meta = prepare_fp4_k_write_metadata(
        _decode_plan(seq_lens, compress_ratio), out_loc, rope_len
    )

    # Row 0 is the only aligned, in-range row; 1 is unaligned, 2 has a negative
    # RoPE position and 3 runs past the table. Only the slot mask has to cover
    # all three: an out-of-range position is additionally clamped to 0 to keep
    # the RoPE gather in bounds, but an unaligned row keeps its position and is
    # dropped by its -1 slot alone.
    torch.testing.assert_close(
        meta.slots,
        torch.tensor([11, -1, -1, -1], device=get_device(), dtype=torch.int64),
    )
    torch.testing.assert_close(
        meta.positions,
        torch.tensor([4, 5, 0, 0], device=get_device(), dtype=torch.int64),
    )


def test_k_write_metadata_prefill_gathers_ragged_slots() -> None:
    compress_ratio = 4
    seq_lens = torch.tensor([4, 8, 12, 16], device=get_device(), dtype=torch.int64)
    ragged_ids = torch.tensor([2, 0, 5, 9], device=get_device(), dtype=torch.int64)
    out_loc = torch.arange(6, device=get_device(), dtype=torch.int64) * 7

    meta = prepare_fp4_k_write_metadata(
        _prefill_plan(seq_lens, ragged_ids, compress_ratio), out_loc, 4096
    )

    # ragged_id 9 is past the end of out_loc and must be dropped, not clamped.
    torch.testing.assert_close(
        meta.slots,
        torch.tensor([14, 0, 35, -1], device=get_device(), dtype=torch.int64),
    )
    torch.testing.assert_close(meta.positions, seq_lens - compress_ratio)


def test_k_write_metadata_prefill_with_empty_out_loc_writes_nothing() -> None:
    seq_lens = torch.tensor([4, 8, 12], device=get_device(), dtype=torch.int64)
    plan = _prefill_plan(seq_lens, torch.zeros_like(seq_lens), 4)

    meta = prepare_fp4_k_write_metadata(
        plan, torch.empty(0, device=get_device(), dtype=torch.int64), 4096
    )

    assert meta.slots.eq(-1).all()


def _build_logits_case(
    batch: int,
    seq_len: int,
    *,
    ctx_lens: list[int] | None = None,
    shuffle_pages: bool = False,
):
    """Populate an FP4 K cache and quantized Q for one synthetic indexer step.

    Every slot of every page is written, including the tail past ``ctx_lens``,
    so a last block that is only partly in context still has live neighbours
    that must not leak into the scored range.
    """
    pages_per_seq = -(-seq_len // PAGE_SIZE)
    padded_len = pages_per_seq * PAGE_SIZE
    num_pages = batch * pages_per_seq
    cos, sin, _ = _rope_tables(max(padded_len, 256))
    payload, scale = _empty_index_k_cache(num_pages)

    physical = (
        torch.randperm(num_pages, device=get_device())
        if shuffle_pages
        else torch.arange(num_pages, device=get_device())
    )
    page_table = physical.to(torch.int32).reshape(batch, pages_per_seq)
    context = torch.tensor(
        ctx_lens if ctx_lens is not None else [seq_len] * batch,
        device=get_device(),
        dtype=torch.int32,
    )

    kv_positions = (
        torch.arange(padded_len, device=get_device(), dtype=torch.int64)
        .repeat(batch)
        .reshape(batch, padded_len)
    )
    loc = (
        page_table.long()[:, :, None] * PAGE_SIZE
        + torch.arange(PAGE_SIZE, device=get_device(), dtype=torch.int64)[None, None, :]
    ).reshape(batch, padded_len)

    k = torch.randn(
        batch, padded_len, HEAD_DIM, device=get_device(), dtype=torch.bfloat16
    )
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    _write_index_k_cache(
        k.reshape(-1, HEAD_DIM),
        norm_weight,
        cos,
        sin,
        kv_positions.reshape(-1),
        loc.reshape(-1),
        payload,
        scale,
    )
    k_ref = _ref_k_transform(
        k.reshape(-1, HEAD_DIM), norm_weight, cos, sin, kv_positions.reshape(-1)
    ).reshape(batch, padded_len, HEAD_DIM)

    q = torch.randn(
        batch, NUM_HEADS, HEAD_DIM, device=get_device(), dtype=torch.bfloat16
    )
    q_positions = (context.long() - 1).clamp_min(0)
    q_fp4, q_scale = aiter_q_indexer_fp4(q.contiguous(), cos, sin, q_positions)
    q_ref = _ref_q_transform(q, cos, sin, q_positions)

    # ``C4Indexer.compute_weights`` runs a bf16 projection and the adapter
    # forwards the result unconverted, so the kernel is fed bf16 weights.
    weights = torch.randn(batch, NUM_HEADS, device=get_device(), dtype=torch.bfloat16)
    weight_scale = HEAD_DIM**-0.5 * NUM_HEADS**-0.5

    q_dq = _ref_dequantize_fp4_indexer(
        q_fp4.view(torch.uint8), _ref_quantize_fp4_indexer(q_ref)[1]
    ).reshape(batch, NUM_HEADS, HEAD_DIM)
    k_dq = _ref_dequantize_fp4_indexer(
        *_read_index_k_cache(payload, scale, loc.reshape(-1))
    ).reshape(batch, padded_len, HEAD_DIM)

    def _logits_from(q_src, k_src) -> torch.Tensor:
        # The indexer scores each head separately, clamps it at zero and only
        # then takes the weighted sum; dropping the ReLU changes the result.
        per_head = torch.einsum("qhd,qsd->qhs", q_src.float(), k_src.float())
        return weight_scale * torch.einsum(
            "qhs,qh->qs", per_head.relu(), weights.float()
        )

    return {
        "q_fp4": q_fp4,
        "q_scale": q_scale,
        "payload": payload,
        "scale": scale,
        "weights": weights,
        "weight_scale": weight_scale,
        "page_table": page_table,
        "c4_seq_lens": context,
        # Against exactly the FP4 operands the kernel read, and against the
        # unquantized bf16 model.
        "ref_logits_fp4": _logits_from(q_dq, k_dq),
        "ref_logits_bf16": _logits_from(q_ref, k_ref),
        "context": context,
        "seq_len": seq_len,
    }


def _run_logits(case, *, is_decode: bool, decode_ws=None, prefill_ws=None):
    return aiter_fp4_paged_mqa_logits(
        q_fp4=case["q_fp4"],
        q_scale=case["q_scale"],
        k_payload=case["payload"],
        k_scale=case["scale"],
        weights=case["weights"],
        page_table=case["page_table"],
        c4_seq_lens=case["c4_seq_lens"],
        weight_scale=case["weight_scale"],
        is_decode=is_decode,
        decode_workspace=decode_ws,
        prefill_workspace=prefill_ws,
    )


def _assert_logits_agree(logits: torch.Tensor, case) -> None:
    """Check each row over its own context, against FP4 operands and bf16.

    The FP4 comparison is the tight one: the reference is fed exactly what the
    kernel read, so only reduction order differs. The bf16 comparison is a
    coarse guard that FP4 has not disturbed the ranking the indexer is about to
    top-k; a dropped ReLU or a mispaired scale lands near 0.7 there, well clear
    of the quantization noise floor. Positions past a row's context are left
    undefined by design, so they are never compared.
    """
    for row, ctx in enumerate(case["context"].tolist()):
        if ctx == 0:
            # A padded row owns no valid position; all it must do is leave the
            # rest of the batch alone, which the other iterations cover.
            continue
        got = logits[row, :ctx]
        exact = case["ref_logits_fp4"][row, :ctx]
        bf16 = case["ref_logits_bf16"][row, :ctx]
        torch.testing.assert_close(
            got, exact, rtol=2.0e-3, atol=2.0e-3, msg=f"row {row} (ctx={ctx})"
        )
        cosine = torch.nn.functional.cosine_similarity(got, bf16, dim=-1).item()
        assert cosine > 0.95, f"row {row} (ctx={ctx}) cosine vs bf16 is {cosine:.4f}"
        topk = min(64, ctx)
        overlap = (
            len(
                set(got.topk(topk).indices.tolist())
                & set(bf16.topk(topk).indices.tolist())
            )
            / topk
        )
        assert overlap > 0.75, f"row {row} top-{topk} overlap is {overlap:.3f}"


@pytest.mark.parametrize("batch,seq_len", [(1, 256), (2, 384), (4, 512)])
def test_decode_paged_mqa_logits(batch: int, seq_len: int) -> None:
    torch.manual_seed(batch * 100 + seq_len)
    case = _build_logits_case(batch, seq_len)
    workspace = prepare_fp4_decode_workspace(case["page_table"], case["c4_seq_lens"])

    logits = _run_logits(case, is_decode=True, decode_ws=workspace)

    _assert_logits_agree(logits, case)


@pytest.mark.parametrize("batch,seq_len", [(1, 256), (3, 512)])
def test_prefill_paged_mqa_logits(batch: int, seq_len: int) -> None:
    torch.manual_seed(batch * 200 + seq_len)
    case = _build_logits_case(batch, seq_len)
    workspace = prepare_fp4_prefill_workspace(case["page_table"], case["c4_seq_lens"])

    logits = _run_logits(case, is_decode=False, prefill_ws=workspace)

    _assert_logits_agree(logits, case)


@pytest.mark.parametrize("is_decode", [True, False])
def test_logits_with_ragged_context_lengths(is_decode: bool) -> None:
    """Sizing the persistent grid for uneven contexts is the scheduler's job.

    A uniform batch hides an unbalanced chunk assignment: every row gets the
    same number of KV chunks, so an off-by-one in the split still covers each
    row exactly once.
    """
    torch.manual_seed(21 if is_decode else 22)
    # Deliberately not multiples of the 64-token page or the 256-token chunk,
    # so the last block of each row is only partly in context.
    # A 0 stands for a padded row: ``match_num_queries`` pads c4_seq_lens with
    # 0 on the FP4 path, and such a row must not disturb its neighbours.
    ctx_lens = [17, 512, 1, 300, 0, 64, 129, 511]
    case = _build_logits_case(len(ctx_lens), 512, ctx_lens=ctx_lens)
    if is_decode:
        ws = prepare_fp4_decode_workspace(case["page_table"], case["c4_seq_lens"])
        logits = _run_logits(case, is_decode=True, decode_ws=ws)
    else:
        ws = prepare_fp4_prefill_workspace(case["page_table"], case["c4_seq_lens"])
        logits = _run_logits(case, is_decode=False, prefill_ws=ws)

    _assert_logits_agree(logits, case)


@pytest.mark.parametrize("is_decode", [True, False])
def test_logits_follow_shuffled_page_table(is_decode: bool) -> None:
    """Pages of a sequence are neither contiguous nor ordered under a radix cache."""
    torch.manual_seed(31 if is_decode else 32)
    case = _build_logits_case(4, 384, ctx_lens=[384, 300, 129, 384], shuffle_pages=True)
    if is_decode:
        ws = prepare_fp4_decode_workspace(case["page_table"], case["c4_seq_lens"])
        logits = _run_logits(case, is_decode=True, decode_ws=ws)
    else:
        ws = prepare_fp4_prefill_workspace(case["page_table"], case["c4_seq_lens"])
        logits = _run_logits(case, is_decode=False, prefill_ws=ws)

    _assert_logits_agree(logits, case)


@pytest.mark.parametrize("is_decode", [True, False])
def test_pinned_schedule_matches_unpinned_logits(is_decode: bool) -> None:
    """A pinned workspace only preplans the grid; the logits must not move."""
    torch.manual_seed(11 if is_decode else 12)
    case = _build_logits_case(2, 384)
    if is_decode:
        workspace = prepare_fp4_decode_workspace(
            case["page_table"], case["c4_seq_lens"]
        )
        pinned = _run_logits(case, is_decode=True, decode_ws=workspace)
    else:
        workspace = prepare_fp4_prefill_workspace(
            case["page_table"], case["c4_seq_lens"]
        )
        pinned = _run_logits(case, is_decode=False, prefill_ws=workspace)
    # Prefill scores are views of one pooled block, so the second call would
    # otherwise hand back the same memory and compare it against itself.
    pinned = pinned.clone()
    unpinned = _run_logits(case, is_decode=is_decode)

    seq_len = case["seq_len"]
    torch.testing.assert_close(pinned[:, :seq_len], unpinned[:, :seq_len])


def test_stale_workspace_row_count_falls_back_to_inline_schedule() -> None:
    """DP padding can leave a workspace sized for a different row count."""
    torch.manual_seed(13)
    case = _build_logits_case(2, 256)
    stale = prepare_fp4_decode_workspace(
        case["page_table"][:1], case["c4_seq_lens"][:1]
    )

    with_stale = _run_logits(case, is_decode=True, decode_ws=stale)
    without = _run_logits(case, is_decode=True)

    seq_len = case["seq_len"]
    torch.testing.assert_close(with_stale[:, :seq_len], without[:, :seq_len])


def test_prefill_logits_come_from_one_pooled_block() -> None:
    """Prefill must score into one constant-size block, not a per-call rectangle.

    The logits width tracks context length, so a fresh allocation per call feeds
    the caching allocator a growing size sequence: each request outgrows every
    cached block and strands a segment, until an allocator that bypasses torch
    (Triton kernel scratch) is refused. One pooled block keeps the request size
    constant, which is what makes the blocks reusable.
    """
    torch.manual_seed(14)
    narrow = _run_logits(_build_logits_case(2, 256), is_decode=False)
    wide = _run_logits(_build_logits_case(4, 512), is_decode=False)

    assert narrow.shape != wide.shape
    assert narrow.data_ptr() == wide.data_ptr()


def test_row_chunks_reproduce_the_unsplit_batch() -> None:
    """Rows are scored and reduced independently, which is what lets callers chunk.

    ``forward_c4_indexer`` splits prefill rows to whatever fits the pooled block,
    so a chunk must score its rows exactly as an unsplit call would.
    """
    torch.manual_seed(15)
    batch, chunk_rows = 6, 2
    case = _build_logits_case(batch, 512)
    # Cloned: the chunk calls below score into the same pooled block.
    full = _run_logits(case, is_decode=False).clone()

    for start in range(0, batch, chunk_rows):
        rows = slice(start, start + chunk_rows)
        chunk = aiter_fp4_paged_mqa_logits(
            q_fp4=case["q_fp4"][rows],
            q_scale=case["q_scale"][rows],
            k_payload=case["payload"],
            k_scale=case["scale"],
            weights=case["weights"][rows],
            page_table=case["page_table"][rows],
            c4_seq_lens=case["c4_seq_lens"][rows],
            weight_scale=case["weight_scale"],
            is_decode=False,
        )
        for row, ctx in enumerate(case["context"][rows].tolist()):
            torch.testing.assert_close(
                chunk[row, :ctx],
                full[start + row, :ctx],
                msg=f"row {start + row} (ctx={ctx})",
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
