# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import logging
import os

import aiter
import torch
import triton
import triton.language as tl
from aiter.jit.core import ENABLE_CK as _AITER_ENABLE_CK

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.aiter import (
    USE_AITER_GFX95,
    USE_AITER_GFX1250,
)

logger = logging.getLogger(__name__)

_use_fp8_attn = os.environ.get("SGLANG_DIFFUSION_AITER_FP8_ATTN", "0") == "1"
_fp8_dtype = torch.float8_e4m3fn

# fmha_fwd_hd128_fp8_gfx950 ASM kernel. Support full MHA with q/k/v head_dim == 128 -- e.g., Wan 2.2 self- and cross-attention.
_FMHA_FP8_HEAD_DIM = 128

_FP8_MAX = torch.finfo(_fp8_dtype).max

# AITER FAv3 on gfx942 supports three different rounding modes:
#   0 = RTNE (round to nearest even)
#   1 = RTNA (round to nearest away)
#   2 = RTZ  (round to zero)
AITER_BF16_CVT_MODE: int = 2


if _use_fp8_attn:
    logger.info("DiT FP8 attention enabled via SGLANG_DIFFUSION_AITER_FP8_ATTN=1")


@triton.jit
def _per_tensor_quant_fp8_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    scratch_ptr,  # int32[3]: [0] = barrier counter, [1]/[2] = amax bits (ping-pong)
    n_elements,
    gen,  # launch generation, monotonically increasing
    INV_FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    MAX_SPIN: tl.constexpr,
):
    """Grid-wide amax reduction plus rescale-and-cast, in a single launch.

    The grid is persistent (one program per CU) and the two phases are separated
    by a grid-wide barrier, so all ``NUM_PROGRAMS`` programs must be co-resident.
    No host-side scratch reset is needed: the amax slot ping-pongs between two
    generations, and each launch clears the slot the next one reduces into.
    """
    pid = tl.program_id(0)
    cur = 1 + (gen % 2)
    nxt = 1 + ((gen + 1) % 2)
    stride = NUM_PROGRAMS * BLOCK

    # phase 1: per-program amax, folded into one global atomic_max
    local_amax = 0.0
    start = pid * BLOCK
    while start < n_elements:
        off = start + tl.arange(0, BLOCK)
        mask = off < n_elements
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        local_amax = tl.maximum(local_amax, tl.max(tl.abs(x)))
        start += stride

    # amax is non-negative, so int ordering of the bit pattern matches float order
    tl.atomic_max(scratch_ptr + cur, local_amax.to(tl.int32, bitcast=True))

    # Grid-wide barrier. The spin has to be a *bounded* loop: with the trip count
    # driven only by the atomic, the compiler hoists the load out of the loop and
    # the kernel hangs on any grid large enough that a program actually spins.
    tl.debug_barrier()
    tl.atomic_add(scratch_ptr, 1)
    target = (gen + 1) * NUM_PROGRAMS
    i = 0
    while i < MAX_SPIN:
        if tl.atomic_add(scratch_ptr, 0) >= target:
            i = MAX_SPIN
        else:
            i += 1

    amax = tl.atomic_add(scratch_ptr + cur, 0).to(tl.float32, bitcast=True)
    # Multiply by the reciprocal rather than divide: torch lowers ``amax / 448.0``
    # that way, and the two differ by 1 ULP -- enough to move the elements that
    # saturate and so to break bit-exactness with aiter.
    scale = amax * INV_FP8_MAX

    if pid == 0:
        tl.store(scale_ptr, scale)
        tl.store(scratch_ptr + nxt, 0)

    # phase 2: rescale + cast
    start = pid * BLOCK
    while start < n_elements:
        off = start + tl.arange(0, BLOCK)
        mask = off < n_elements
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        y = tl.fdiv(x, scale, ieee_rounding=True)
        tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=mask)
        start += stride


_QUANT_MAX_PROGRAMS = 256  # gfx950 CU count
_QUANT_BLOCK = 16384  # with num_warps=4, swept best on the Wan2.2 q/k/v shapes
_QUANT_NUM_WARPS = 4

# Per device: the persistent grid size and the int32[3] barrier/amax scratch.
_quant_grid: dict[torch.device, int] = {}
_quant_scratch: dict[torch.device, torch.Tensor] = {}
_quant_gen: dict[torch.device, int] = {}


def _per_tensor_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact drop-in for ``aiter.per_tensor_quant(x, quant_dtype=fp8)``, 6.6x
    faster on [1, 90000, 40, 128] bf16 (3.36 ms -> 0.50 ms on gfx950).

    aiter's reference upcasts to fp32 and runs abs/max/div/cast as four more eager
    kernels; its fused HIP/Triton paths decompose over rows of ``size(-1)``, which
    at head_dim 128 is one workgroup per 128 elements contending on one atomic.
    The reduction has to be decomposed over elements instead, which is what the
    persistent kernel above does -- in one launch, at the ~2.3 GB traffic floor.
    """
    xc = x.contiguous()
    y = torch.empty_like(xc, dtype=_fp8_dtype)
    scale = torch.empty(1, dtype=torch.float32, device=xc.device)

    device = xc.device
    grid = _quant_grid.get(device)
    if grid is None:
        # The barrier requires a co-resident grid, so never ask for more
        # programs than the device has CUs (partitioned GPUs report fewer).
        grid = min(
            _QUANT_MAX_PROGRAMS,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        _quant_grid[device] = grid
        _quant_scratch[device] = torch.zeros(3, dtype=torch.int32, device=device)
        _quant_gen[device] = 0
    gen = _quant_gen[device]
    _quant_gen[device] = gen + 1

    _per_tensor_quant_fp8_kernel[(grid,)](
        xc,
        y,
        scale,
        _quant_scratch[device],
        xc.numel(),
        gen,
        INV_FP8_MAX=1.0 / _FP8_MAX,
        BLOCK=_QUANT_BLOCK,
        NUM_PROGRAMS=grid,
        MAX_SPIN=100_000_000,
        num_warps=_QUANT_NUM_WARPS,
        num_stages=1,
    )
    return y.view(x.shape), scale


def _can_use_fmha_fp8_prefill(
    q_head_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> bool:
    """True if MHA q/k/v head_dim==128 on a gfx950-class arch."""
    if not USE_AITER_GFX95:
        return False
    if num_kv_heads != num_heads:
        return False
    return q_head_dim == k_head_dim == v_head_dim == _FMHA_FP8_HEAD_DIM


def _fmha_fp8_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 FMHA prefill via aiter.flash_attn_fp8_pertensor_func.

    Expects q, k, v as (batch, seqlen, nheads, 128) FP8, contiguous.
    """

    def _ensure_fp8_descale(scale: torch.Tensor) -> torch.Tensor:
        """Per-tensor descale as shape (1,) float32 for flash_attn_fp8_pertensor_func."""
        return scale.to(dtype=torch.float32).reshape(1).contiguous()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_descale = _ensure_fp8_descale(q_scale)
    k_descale = _ensure_fp8_descale(k_scale)
    v_descale = _ensure_fp8_descale(v_scale)

    return aiter.flash_attn_fp8_pertensor_func(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        causal=is_causal,
        softmax_scale=softmax_scale,
        window_size=(-1, -1, 0),
    )


def _load_mha_v4():
    """aiter's MHA v4 module when its gfx950 BF16 kernel is safe to use.

    MHA v4 is 8-9% faster than fmha v3 (flash_attn_func) at MiniMax-H3's MI350X
    shapes; both are BF16 flash attention with FP32 softmax and accumulation.
    Before ROCm/aiter#6841 it had RAW races: on 8 MI350X about 1 call in 800
    returned wrong rows (up to 0.25 off), so outputs varied run to run. That
    change also dropped ``sparse`` from ``_check_lse_capable``, which tells the
    builds apart.
    """
    if not USE_AITER_GFX95:
        return None
    try:
        from aiter.ops import mha_v4
    except ImportError:
        return None
    try:
        racy = "sparse" in inspect.signature(mha_v4._check_lse_capable).parameters
    except AttributeError:  # gone in later aiter, after the fix
        racy = False
    return None if racy else mha_v4


_MHA_V4 = _load_mha_v4()

# fmha v3 and MHA v4 run one 256-row query tile per CU per wave. When the last
# wave holds only a few tiles (Pika H3 Flex 5 s on 8 GPUs: 7 heads x 148 tiles
# = 4 waves + 12 tiles on 256 CUs), the tail rows finish sooner with their keys
# split over every CU and the partial results merged by LSE. Adapted from
# tail_plan in github.com/Pika-Labs/Pika-H3-Flex, branch contrib/amd-mi350x
# (src/pika_h3/sp_model.py, Apache-2.0).
_TILE_ROWS = 256
_TAIL_MAX_TILES = 96
# that branch reports no gain at 8+ waves: a nearly idle last wave runs at a
# higher clock under the power cap
_TAIL_MAX_WAVES = 6
_TAIL_KEY_CHUNKS = 8


@functools.lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _tail_split_rows(rows: int, heads: int, device: torch.device) -> int | None:
    """Query rows of the full waves when the last wave is mostly idle, else None."""
    cus = _num_cus(device.index)
    waves, rem = divmod(heads * -(-rows // _TILE_ROWS), cus)
    if not 0 < waves <= _TAIL_MAX_WAVES or rem == 0 or rem > _TAIL_MAX_TILES:
        return None
    return waves * cus // heads * _TILE_ROWS


def _flash(q, k, v, softmax_scale, return_lse=False, out=None):
    """Non-causal BSHD attention, written into ``out`` when given; with
    return_lse also the [B, H, S] softmax LSE. q/k/v need a unit-stride head
    dim only. MHA v4 for BF16 head dim 128, else fmha v3."""
    if _MHA_V4 is not None and q.dtype == torch.bfloat16 and q.shape[-1] == 128:
        bf16 = _MHA_V4.AttentionFormat.BF16
        none = _MHA_V4.AttentionScaleMode.NONE
        # the descale operands are unused for BF16; aiter takes any tensor there
        result = _MHA_V4.mha_v4_packed(
            q,
            k,
            v,
            q,
            k,
            v,
            bf16,
            bf16,
            bf16,
            none,
            none,
            none,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
        )
        return tuple(result) if return_lse else result
    result = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        return_lse=return_lse,
        how_v3_bf16_cvt=AITER_BF16_CVT_MODE,
    )
    o = result[0] if return_lse else result
    if out is not None:
        out.copy_(o)
        o = out
    return (o, result[1]) if return_lse else o


@triton.jit
def _lse_merge_kernel(
    o_ptr,
    lse_ptr,
    out_ptr,
    heads,
    n_chunks,
    o_stride_c,
    o_stride_r,
    o_stride_h,
    lse_stride_c,
    lse_stride_h,
    out_stride_r,
    out_stride_h,
    CHUNKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0) // heads
    head = tl.program_id(0) % heads
    c = tl.arange(0, CHUNKS)
    live = c < n_chunks
    lse = tl.load(
        lse_ptr + c * lse_stride_c + head * lse_stride_h + row,
        mask=live,
        other=float("-inf"),
    )
    w = tl.exp(lse - tl.max(lse, 0))
    w = w / tl.sum(w, 0)
    d = tl.arange(0, HEAD_DIM)
    o = tl.load(
        o_ptr + c[:, None] * o_stride_c + row * o_stride_r + head * o_stride_h + d,
        mask=live[:, None],
        other=0.0,
    )
    acc = tl.sum(o.to(tl.float32) * w[:, None], 0)
    tl.store(
        out_ptr + row * out_stride_r + head * out_stride_h + d,
        acc.to(out_ptr.dtype.element_ty),
    )


def _lse_merge(o, lse, out):
    """o [C, rows, H, D] partial outputs, lse [C, H, rows] -> out [rows, H, D]."""
    chunks, rows, heads, head_dim = o.shape
    _lse_merge_kernel[(rows * heads,)](
        o,
        lse,
        out,
        heads,
        chunks,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        out.stride(0),
        out.stride(1),
        CHUNKS=triton.next_power_of_2(chunks),
        HEAD_DIM=head_dim,
    )


def _tail_split_attention(q, k, v, softmax_scale, out, full_rows):
    """q/k/v [1, S, H, D] -> out [S, H, D]: rows [:full_rows] in one launch,
    the rest with keys split in _TAIL_KEY_CHUNKS ranges, merged by LSE."""
    _flash(q[:, :full_rows], k, v, softmax_scale, out=out[None, :full_rows])
    tail_rows, keys = q.shape[1] - full_rows, k.shape[1]
    chunks = _TAIL_KEY_CHUNKS
    size = keys // chunks
    split = chunks * size
    q_tail = q[:, full_rows:]
    o, lse = _flash(
        q_tail.expand(chunks, tail_rows, *q.shape[2:]),
        k[0, :split].unflatten(0, (chunks, size)),
        v[0, :split].unflatten(0, (chunks, size)),
        softmax_scale,
        return_lse=True,
    )
    if split < keys:
        o_rest, lse_rest = _flash(
            q_tail, k[:, split:], v[:, split:], softmax_scale, return_lse=True
        )
        o, lse = torch.cat((o, o_rest)), torch.cat((lse, lse_rest))
    _lse_merge(o, lse, out[full_rows:])


@functools.lru_cache(maxsize=64)
def _cu_seqlens(bounds: tuple[int, ...], device: torch.device) -> torch.Tensor:
    # one upload per bounds and device; callers must not mutate the result
    return torch.tensor(bounds, dtype=torch.int32, device=device)


class AITerBackend(AttentionBackend):
    """
    Backend for AITemplate attention implementation.
    """

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER

    @staticmethod
    def get_impl_cls() -> type["AITerImpl"]:
        return AITerImpl

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError("AITer backend does not have a metadata builder.")

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return True


class AITerImpl(AttentionImpl):
    """
    Implementation of attention using AITemplate.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        # aiter's mha entry points take GQA/MQA K/V directly (they broadcast
        # each KV head across its group of query heads), so the only
        # requirement is an even split. The FP8 ASM path is MHA-only and
        # already routes grouped shapes back to BF16 below.
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError(
                f"AITer backend requires num_heads ({num_heads}) to be a "
                f"multiple of num_kv_heads ({num_kv_heads})."
            )
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )

    @torch.compiler.disable
    def _forward_fp8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        FP8 FMHA prefill path. Kept out of the compiled graph (the aiter FP8
        quant + FMHA kernels are not torch.compile-safe); only invoked when
        SGLANG_DIFFUSION_AITER_FP8_ATTN=1.

        Returns the attention output, or None if the FP8 FMHA kernel does not
        support the current shapes, in which case the caller falls back to the
        (compilable) BF16 path.
        """
        if query.dtype != _fp8_dtype:
            q_fp8, q_scale = _per_tensor_quant_fp8(query)
            k_fp8, k_scale = _per_tensor_quant_fp8(key)
            v_fp8, v_scale = _per_tensor_quant_fp8(value)
        else:
            q_fp8, k_fp8, v_fp8 = query, key, value
            one = torch.tensor(1.0, dtype=torch.float32, device=query.device)
            q_scale = k_scale = v_scale = one

        d_q = q_fp8.shape[-1]
        d_k = k_fp8.shape[-1]
        d_v = v_fp8.shape[-1]
        h_q = q_fp8.shape[2]
        h_kv = k_fp8.shape[2]

        if _can_use_fmha_fp8_prefill(d_q, d_k, d_v, h_q, h_kv):
            return _fmha_fp8_prefill_attention(
                q_fp8,
                k_fp8,
                v_fp8,
                softmax_scale=self.softmax_scale,
                is_causal=self.causal,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        logger.warning_once(
            "FP8 FMHA prefill unsupported for this shape (need gfx950-class AITER, "
            "full MHA, q/k/v head_dim=%d; got q=%d, k=%d, v=%d, num_heads=%d, "
            "num_kv_heads=%d). Falling back to BF16.",
            _FMHA_FP8_HEAD_DIM,
            d_q,
            d_k,
            d_v,
            h_q,
            h_kv,
        )
        return None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Performs attention using one of:
          - _fmha_fp8_prefill_attention (FP8, SGLANG_DIFFUSION_AITER_FP8_ATTN=1 when eligible)
          - flash_attn_func (BF16, default or FP8 fallback for unsupported shapes)

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim],
            plus the [batch_size, num_heads, seq_len] softmax LSE when
            return_softmax_lse (always the BF16 kernel, for ring merging).
        """
        if return_softmax_lse:
            return self._forward_bf16(query, key, value, return_lse=True)
        if _use_fp8_attn:
            output = self._forward_fp8(query, key, value)
            if output is not None:
                return output

        if USE_AITER_GFX1250 or not _AITER_ENABLE_CK:
            return self._forward_bf16_eager(query, key, value)
        return self._forward_bf16(query, key, value)

    def _forward_bf16(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output = aiter.flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_p,
            causal=self.causal,
            return_attn_probs=False,
            return_lse=return_lse,
            how_v3_bf16_cvt=AITER_BF16_CVT_MODE,
        )
        return tuple(output[:2]) if return_lse else output

    @torch.compiler.disable
    def _forward_bf16_eager(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_bf16(query, key, value)

    @torch.compiler.disable
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        used = None
        # gfx950 only: the batch kernels and the key split were measured there;
        # other architectures keep the varlen kernel
        if USE_AITER_GFX95 and cu_seqlens_host is not None:
            if tuple(cu_seqlens_host) == (0, query.shape[0]):
                used = query.shape[0]
            elif self.packed_trailing_padding:
                used = trailing_padding_used_len(
                    total_tokens=query.shape[0],
                    max_seqlen=max_seqlen,
                    bounds=cu_seqlens_host,
                )
        if used is not None:
            # One live sequence (plus trailing padding): the batch kernels read
            # strided q/k/v. fmha v3 gives its rows the varlen kernel's bytes
            # and is 9-14% faster at MiniMax-H3's MI350X shapes. Padding rows
            # are zeroed, as the FA path leaves them; no live row attends to them.
            q, k, v = (t[None, :used] for t in (query, key, value))
            if self.causal or self.dropout_p != 0.0:
                live = aiter.flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_p,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                    how_v3_bf16_cvt=AITER_BF16_CVT_MODE,
                )[0]
            elif full_rows := _tail_split_rows(used, query.shape[1], query.device):
                output = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
                _tail_split_attention(
                    q, k, v, self.softmax_scale, output[:used], full_rows
                )
                output[used:].zero_()
                return output
            else:
                live = _flash(q, k, v, self.softmax_scale)[0]
            if used == query.shape[0]:
                return live
            output = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
            output[:used] = live
            output[used:].zero_()
            return output
        attention_func = aiter.flash_attn_varlen_func

        cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
        output = attention_func(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            how_v3_bf16_cvt=AITER_BF16_CVT_MODE,
        )
        return output[0] if isinstance(output, tuple) else output

    @torch.compiler.disable
    def forward_ring_kv_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend packed [T, H, D] queries to one rotated KV chunk; returns
        the output and the [H, Tq] softmax LSE."""
        output, lse = aiter.flash_attn_varlen_func(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            cu_seqlens_q=_cu_seqlens((0, query.shape[0]), query.device),
            cu_seqlens_k=_cu_seqlens((0, key.shape[0]), key.device),
            max_seqlen_q=query.shape[0],
            max_seqlen_k=key.shape[0],
            softmax_scale=self.softmax_scale,
            causal=False,
            how_v3_bf16_cvt=AITER_BF16_CVT_MODE,
            return_lse=True,
        )[:2]
        return output, lse
