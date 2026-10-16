"""Isolated numeric-first SM80 sparse MLA split-K candidate, NOT runtime routing.

Each program vectorizes 32 candidates for one query/head; a second pass merges
FP32 partials. No TensorCore/probability downcast yet: dot products and weighted
values use FP32 reductions, keeping the serial fallback's accuracy contract.
Only byte cache pointers are passed to kernels. The serial implementation stays
unchanged and is the production/reference route until GPU validation passes.
"""

import math
import numbers

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

from .dequant_k_cache_sm80 import (
    _e2m1_to_fp32,
    _e4m3_to_fp32,
    _scaled_bf16,
    _ue8m0_to_fp32,
)
from .kv_layout import KVLayout
from .sparse_attention_sm80 import _validate_tensor


# Fixed launch geometry keeps graph replay and workspace sizes deterministic.
_CANDIDATES = 32


@triton.jit
def _partial_kernel(
    q_ptr, cache_ptr, cache32_ptr, indices_ptr, length_ptr, partial_ptr, softmax_scale,
    H: tl.constexpr, K: tl.constexpr, HAS_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr, PITCH: tl.constexpr,
    DATA_BYTES: tl.constexpr, SCALE_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr, TILE_SIZE: tl.constexpr, IS_FP4: tl.constexpr,
    CHUNKS: tl.constexpr, CHUNK_OFFSET: tl.constexpr, BLOCK: tl.constexpr,
    WORD_LOADS: tl.constexpr = False,
):
    tl.static_assert(cache_ptr.dtype.element_ty == tl.uint8)
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    batch = row // H
    j = chunk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    d = tl.arange(0, 512).to(tl.int64)
    length = tl.full((), K, tl.int64)
    if HAS_LENGTH:
        length = tl.load(length_ptr + batch).to(tl.int64)
    loc = tl.load(indices_ptr + batch * K + j,
                  mask=(j < K) & (j < length), other=-1).to(tl.int64)
    valid = (j < K) & (j < length) & (loc >= 0)
    safe_loc = tl.where(valid, loc, 0)
    page_base = (safe_loc // PAGE_SIZE) * PITCH
    slot = safe_loc % PAGE_SIZE
    data_base = page_base + slot * DATA_BYTES
    scale_base = page_base + SCALE_OFFSET + slot * SCALE_BYTES
    if WORD_LOADS:
        # Load the payload as 4-byte words and unpack in registers. Both byte
        # forms below are narrow and partly redundant: the FP8 branch reads one
        # byte per dimension, the FP4 branch reads `d // 2` so each packed byte
        # is fetched twice, and both re-read the same TILE_SIZE-wide scale byte
        # 32 times per key. Measured on A100 at the live shapes this is 1.34x
        # (V41) and 1.29x (V41_FP4) faster with bitwise-identical partials.
        if IS_FP4:
            wd = tl.arange(0, DATA_BYTES // 4)
            w = tl.load(cache32_ptr + data_base[:, None] // 4 + wd[None, :],
                        mask=valid[:, None], other=0)
            nib = (w[:, :, None] >> ((tl.arange(0, 8) * 4)[None, None, :])) & 15
            value = _e2m1_to_fp32(tl.reshape(nib, (BLOCK, 512)))
            raw_scale = tl.load(cache_ptr + scale_base[:, None]
                                + tl.arange(0, 512 // TILE_SIZE)[None, :],
                                mask=valid[:, None], other=0)
            scales = tl.reshape(
                tl.broadcast_to(_e4m3_to_fp32(raw_scale)[:, :, None],
                                (BLOCK, 512 // TILE_SIZE, TILE_SIZE)),
                (BLOCK, 512),
            )
        else:
            wd = tl.arange(0, 128)
            w = tl.load(cache32_ptr + data_base[:, None] // 4 + wd[None, :],
                        mask=valid[:, None], other=0)
            by = (w[:, :, None] >> ((tl.arange(0, 4) * 8)[None, None, :])) & 0xFF
            value = _e4m3_to_fp32(tl.reshape(by, (BLOCK, 512)))
            raw_scale = tl.load(cache_ptr + scale_base[:, None]
                                + tl.arange(0, 512 // TILE_SIZE)[None, :],
                                mask=valid[:, None], other=0)
            scales = tl.reshape(
                tl.broadcast_to(_ue8m0_to_fp32(raw_scale)[:, :, None],
                                (BLOCK, 512 // TILE_SIZE, TILE_SIZE)),
                (BLOCK, 512),
            )
    elif IS_FP4:
        packed = tl.load(cache_ptr + data_base[:, None] + d[None, :] // 2,
                         mask=valid[:, None], other=0)
        code = (packed >> ((d[None, :] & 1) * 4)) & 15
        value = _e2m1_to_fp32(code)
        scale_code = tl.load(
            cache_ptr + scale_base[:, None] + d[None, :] // TILE_SIZE,
            mask=valid[:, None], other=0,
        )
        scales = _e4m3_to_fp32(scale_code)
    else:
        code = tl.load(cache_ptr + data_base[:, None] + d[None, :],
                       mask=valid[:, None], other=0)
        value = _e4m3_to_fp32(code)
        scale_code = tl.load(
            cache_ptr + scale_base[:, None] + d[None, :] // TILE_SIZE,
            mask=valid[:, None], other=0,
        )
        scales = _ue8m0_to_fp32(scale_code)
    # Round every decoded token ONCE to BF16, as in the standalone byte decoder.
    kv = _scaled_bf16(value, scales).to(tl.float32)
    q = tl.load(q_ptr + row * 512 + d).to(tl.float32)
    score = tl.sum(kv * q[None, :], 1) * softmax_scale
    score = tl.where(valid, score, float("-inf"))
    maximum = tl.max(score, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    probability = tl.exp(score - safe_maximum)
    denominator = tl.sum(probability, 0)
    numerator = tl.sum(kv * probability[:, None], 0)
    # Keep max and denominator SEPARATE: max+log(den) loses chunk cardinality
    # at large finite scores (e.g. 2**26), corrupting the merged output weights.
    base = (row * CHUNKS + CHUNK_OFFSET + chunk) * 514
    tl.store(partial_ptr + base + d, numerator)
    tl.store(partial_ptr + base + 512, maximum)
    tl.store(partial_ptr + base + 513, denominator)


@triton.jit
def _merge_kernel(
    partial_ptr, sink_ptr, out_ptr, lse_ptr,
    H: tl.constexpr, CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK_C).to(tl.int64)
    d = tl.arange(0, 512).to(tl.int64)
    base = (row * CHUNKS + c) * 514
    partial_maximum = tl.load(partial_ptr + base + 512,
                              mask=c < CHUNKS, other=float("-inf"))
    partial_denominator = tl.load(partial_ptr + base + 513,
                                  mask=c < CHUNKS, other=0.0)
    maximum = tl.max(partial_maximum, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp(partial_maximum - safe_maximum)
    denominator = tl.sum(partial_denominator * weights, 0)
    has_keys = denominator > 0.0
    safe_denominator = tl.where(has_keys, denominator, 1.0)
    key_lse = safe_maximum + tl.log(safe_denominator)
    tl.store(lse_ptr + row, tl.where(has_keys, key_lse, float("-inf")))
    partial = tl.load(partial_ptr + base[:, None] + d[None, :],
                      mask=c[:, None] < CHUNKS, other=0.0)
    numerator = tl.sum(partial * weights[:, None], 0)
    sink = tl.full((), float("-inf"), tl.float32)
    if HAS_SINK:
        sink = tl.load(sink_ptr + row % H)
    # Sink contributes only to the denominator, never returned LSE.
    output_maximum = tl.maximum(safe_maximum, sink)
    key_factor = tl.exp(safe_maximum - output_maximum)
    sink_factor = tl.exp(sink - output_maximum)
    output_denominator = denominator * key_factor + sink_factor
    safe_output_denominator = tl.where(output_denominator > 0.0,
                                        output_denominator, 1.0)
    result = numerator * key_factor / safe_output_denominator
    tl.store(out_ptr + row * 512 + d,
             tl.where(has_keys, result, 0.0).to(tl.bfloat16))


@triton.jit
def _merge_kernel_dtile(
    partial_ptr, sink_ptr, out_ptr, lse_ptr,
    H: tl.constexpr, CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr,
    HAS_SINK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Same arithmetic as ``_merge_kernel`` with the output axis tiled.

    ``_merge_kernel`` materialises a (BLOCK_C, 512) fp32 tile: with the 64+16
    chunk split the live decode has BLOCK_C = 128, i.e. 128 * 512 * 4 B = 256 KB
    across 128 threads = 512 registers per thread, far past the 255-register cap,
    so the production kernel spills (the profiler reports exactly 255 registers
    per thread) and costs 6.8-20.0 us per call. Splitting the 512-wide axis into
    BLOCK_D columns divides the register demand by 512/BLOCK_D while keeping the
    reduction over the chunk axis in the same order: measured on A100, BLOCK_D=64
    is 4.74x faster and BITWISE identical to the production kernel in both the
    bf16 output and the fp32 LSE.
    """
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    c = tl.arange(0, BLOCK_C).to(tl.int64)
    base = (row * CHUNKS + c) * 514
    partial_maximum = tl.load(partial_ptr + base + 512,
                              mask=c < CHUNKS, other=float("-inf"))
    partial_denominator = tl.load(partial_ptr + base + 513,
                                  mask=c < CHUNKS, other=0.0)
    maximum = tl.max(partial_maximum, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp(partial_maximum - safe_maximum)
    denominator = tl.sum(partial_denominator * weights, 0)
    has_keys = denominator > 0.0
    safe_denominator = tl.where(has_keys, denominator, 1.0)
    if dblk == 0:
        key_lse = safe_maximum + tl.log(safe_denominator)
        tl.store(lse_ptr + row, tl.where(has_keys, key_lse, float("-inf")))
    partial = tl.load(partial_ptr + base[:, None] + d[None, :],
                      mask=c[:, None] < CHUNKS, other=0.0)
    numerator = tl.sum(partial * weights[:, None], 0)
    sink = tl.full((), float("-inf"), tl.float32)
    if HAS_SINK:
        sink = tl.load(sink_ptr + row % H)
    # Sink contributes only to the denominator, never returned LSE.
    output_maximum = tl.maximum(safe_maximum, sink)
    key_factor = tl.exp(safe_maximum - output_maximum)
    sink_factor = tl.exp(sink - output_maximum)
    output_denominator = denominator * key_factor + sink_factor
    safe_output_denominator = tl.where(output_denominator > 0.0,
                                        output_denominator, 1.0)
    result = numerator * key_factor / safe_output_denominator
    tl.store(out_ptr + row * 512 + d,
             tl.where(has_keys, result, 0.0).to(tl.bfloat16))


def _validate_cache(name, cache, indices, length, page_size, layout, batch, device):
    if type(page_size) is not int or page_size <= 0:
        raise ValueError(f"{name} page_size must be a positive Python int")
    layout = KVLayout.parse(layout)
    if layout not in (KVLayout.V41, KVLayout.V41_FP4):
        raise ValueError(f"{name} supports only V41/V41_FP4, got {layout}")
    if not isinstance(cache, torch.Tensor) or cache.ndim != 2:
        raise ValueError(f"{name} must be a raw uint8 tensor [pages, pitch]")
    if cache.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype torch.uint8")
    if cache.device != device or not cache.is_cuda:
        raise ValueError(f"{name} must be on {device}")
    # Support padded physical page stride without making a contiguous copy.
    if cache.stride(1) != 1 or cache.stride(0) < cache.shape[1]:
        raise ValueError(f"{name} requires contiguous bytes in nonoverlapping pages")
    if cache.shape[1] < page_size * layout.bytes_per_token:
        raise ValueError(f"{name} pitch cannot hold page_size tokens and their scales")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise ValueError(f"{name} indices must have shape [B, K]")
    _validate_tensor(f"{name} indices", indices, (batch, indices.shape[1]),
                     (torch.int32, torch.int64), device)
    if length is not None:
        _validate_tensor(f"{name} topk_length", length, (batch,),
                         (torch.int32, torch.int64), device)
    return layout


def _partial_side_stream(device):
    """Lazily create (and cache) the side stream used to overlap the two pools.

    The two pool launches write disjoint halves of the same partial workspace and
    read different caches, so they are independent, yet in the captured decode
    graph they run strictly back to back (measured: the (8,16) launch ends at ts
    and the (8,64) launch starts at that same ts, every layer). Launching the
    smaller pool on a side stream hides it behind the larger one. The stream is
    created on first use - never during capture, where allocation is illegal - and
    the caller falls back to the sequential path if it is missing mid-capture.
    """
    from sglang.srt.runtime_context import get_resources

    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    streams = get_resources().streams
    name = f"dsv41_sm80_partial:{index}"
    stream = streams.get(name)
    if stream is None and not torch.cuda.is_current_stream_capturing():
        stream = torch.cuda.Stream(device=device)
        streams[name] = stream
    return stream


def _partial_side_stream_enabled() -> bool:
    return (
        envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() == "1"
        and envs.SGLANG_DSV41_SM80_PARTIAL_SIDE_STREAM.get() == "1"
    )


def sparse_attention_sm80_tiled(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
    layout: KVLayout | str,
    softmax_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    extra_layout: KVLayout | str | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    block: int = _CANDIDATES,
    warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same API/math as serial sparse_attention_sm80; isolated candidate only.

    Contiguous BF16 q[B,H,512], int32/int64 indices[B,K] and optional lengths[B]
    must be on one NVIDIA SM80 device. Cache bytes within a page are contiguous;
    padded stride(0) is supported. Nonnegative IDs must address real tokens
    (caller checked); negatives mask, duplicates count independently, negative
    lengths mask all entries, lengths > K clip to K. Main/extra namespaces,
    formats, page sizes and physical strides are independent.

    Output BF16[B,H,512]; LSE FP32[B,H] is natural logsumexp of unsunk keys.
    FP32 sink[H] is a zero-value denominator-only entry; finite or -inf. Empty
    rows are exact zero/-inf. Finite Q/decoded KV/FP32 scores are caller contract.
    No tensor value/host pointer checks, .item(), copies or synchronization.

    B=0 or H=0 launches no kernels. Workspaces are local deterministic FP32
    [B,H,ceil(K/32)+ceil(extra_K/32),514] (minimum one main chunk for K=0).
    For B64,H8,K512+512 peak partial storage is ~32 MiB. CUDA graph capture
    supports these allocations through PyTorch's graph-private memory pool;
    warm up compilation/device metadata first. Pass out/lse to retain output
    identities. Callers must prevent input/output aliasing. No H padding.
    """
    if type(block) is not int or block not in (8, 16, 32, 64):
        raise ValueError("block must be one of 8, 16, 32, 64")
    if type(warps) is not int or warps not in (4, 8):
        raise ValueError("warps must be 4 or 8")
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("q must have shape [B, H, 512]")
    if not q.is_cuda or torch.version.hip is not None:
        raise ValueError("sparse_attention_sm80_tiled requires NVIDIA CUDA SM80")
    device = q.device
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("sparse_attention_sm80_tiled requires SM80 (not SM86)")
    batch, heads, _ = q.shape
    _validate_tensor("q", q, (batch, heads, 512), (torch.bfloat16,), device)
    if (isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, numbers.Real)
        or not math.isfinite(softmax_scale)):
        raise ValueError("softmax_scale must be a finite Python real scalar")
    softmax_scale = float(softmax_scale)
    layout = _validate_cache("cache", cache, indices, topk_length, page_size,
                             layout, batch, device)
    if attn_sink is not None:
        _validate_tensor("attn_sink", attn_sink, (heads,), (torch.float32,), device)
    has_extra = extra_cache is not None
    extra_arguments = (extra_indices, extra_page_size, extra_layout)
    if has_extra:
        if any(argument is None for argument in extra_arguments):
            raise ValueError("extra_cache requires extra_indices/page_size/layout")
        extra_layout = _validate_cache(
            "extra_cache", extra_cache, extra_indices, extra_topk_length,
            extra_page_size, extra_layout, batch, device,
        )
    elif (any(argument is not None for argument in extra_arguments)
          or extra_topk_length is not None):
        raise ValueError("extra arguments require extra_cache")
    if out is None:
        out = torch.empty((batch, heads, 512), dtype=torch.bfloat16, device=device)
    else:
        _validate_tensor("out", out, (batch, heads, 512), (torch.bfloat16,), device)
    if lse is None:
        lse = torch.empty((batch, heads), dtype=torch.float32, device=device)
    else:
        _validate_tensor("lse", lse, (batch, heads), (torch.float32,), device)
    if batch == 0 or heads == 0:
        return out, lse

    main_chunks = max(1, triton.cdiv(indices.shape[1], block))
    extra_chunks = triton.cdiv(extra_indices.shape[1], block) if has_extra else 0
    chunks = main_chunks + extra_chunks
    partial = torch.empty((batch, heads, chunks, 514), dtype=torch.float32,
                          device=device)
    with torch.cuda.device(device):
        pools = [(cache, indices, topk_length, page_size, layout, main_chunks, 0)]
        if extra_chunks:
            pools.append((extra_cache, extra_indices, extra_topk_length,
                          extra_page_size, extra_layout, extra_chunks, main_chunks))
        word_loads = (
            envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() == "1"
            and envs.SGLANG_DSV41_SM80_PARTIAL_WORD_LOADS.get() == "1"
        )

        def _launch_partial(pool, ids, length, page, fmt, count, offset):
            _partial_kernel[(batch * heads, count)](
                q, pool, pool.view(torch.int32), ids,
                length if length is not None else ids,
                partial, softmax_scale,
                H=heads, K=ids.shape[1], HAS_LENGTH=length is not None,
                PAGE_SIZE=page, PITCH=pool.stride(0),
                DATA_BYTES=fmt.data_bytes, SCALE_BYTES=fmt.scale_bytes,
                SCALE_OFFSET=fmt.scale_offset(page), TILE_SIZE=fmt.tile_size,
                IS_FP4=fmt is KVLayout.V41_FP4,
                CHUNKS=chunks, CHUNK_OFFSET=offset, BLOCK=block,
                WORD_LOADS=word_loads,
                num_warps=warps, enable_fp_fusion=False,
            )

        # Overlap the two independent pool launches when opted in and the side
        # stream already exists (it must not be created during graph capture).
        side = None
        if len(pools) == 2 and _partial_side_stream_enabled():
            side = _partial_side_stream(device)
        if side is not None:
            current = torch.cuda.current_stream()
            side.wait_stream(current)
            _launch_partial(*pools[0])
            with torch.cuda.stream(side):
                _launch_partial(*pools[1])
            current.wait_stream(side)
        else:
            for pool_spec in pools:
                _launch_partial(*pool_spec)
        merge_kwargs = dict(
            H=heads, CHUNKS=chunks, BLOCK_C=triton.next_power_of_2(chunks),
            HAS_SINK=attn_sink is not None,
            num_warps=4, enable_fp_fusion=False,
        )
        # Bitwise-identical, register-tiled merge: opt-in so the default path is
        # untouched. 512 % BLOCK_D == 0 is required by the 2-D grid.
        if (
            envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() == "1"
            and envs.SGLANG_DSV41_SM80_MERGE_DTILE.get() == "1"
        ):
            _merge_kernel_dtile[(batch * heads, 512 // 64)](
                partial, attn_sink if attn_sink is not None else q, out, lse,
                BLOCK_D=64, **merge_kwargs,
            )
        else:
            _merge_kernel[(batch * heads,)](
                partial, attn_sink if attn_sink is not None else q, out, lse,
                **merge_kwargs,
            )
    return out, lse
