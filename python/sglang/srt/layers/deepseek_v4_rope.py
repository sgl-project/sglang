import logging
import math
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# tilelang isn't shipped on every platform (e.g. Ascend NPU images) and the
# only tilelang artifacts in this file are pass_configs that downstream
# tilelang.jit decorators would consume — the kernels actually defined here
# are Triton. Keep the import optional so this module loads on NPU.
try:
    import tilelang

    tilelang.set_log_level("WARNING")

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
except ImportError:
    logger.info(
        "tilelang not installed; deepseek_v4_rope pass_configs unset. "
        "Triton kernels in this module still run; only downstream tilelang.jit "
        "consumers of pass_configs will need to handle the None."
    )
    tilelang = None
    pass_configs = None

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@lru_cache(2)
def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_freqs_cis(
    dim,
    seqlen,
    original_seq_len,
    base,
    factor,
    beta_fast,
    beta_slow,
) -> torch.Tensor:

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@triton.jit
def apply_rotary_emb_triton_kernel(
    x_ptr,
    freqs_ptr,
    positions_ptr,
    rope_dim,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_freq_pos,
    stride_freq_dim,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_dim = tl.program_id(2)

    if USE_POS:
        position = tl.load(positions_ptr + pid_batch)
    else:
        position = pid_batch

    if IS_3D:
        base_offset = pid_batch * stride_x_batch + pid_head * stride_x_head
    else:
        base_offset = pid_batch * stride_x_batch

    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_pair < (rope_dim // 2)

    offs_x_real = base_offset + offs_pair * 2 * stride_x_dim
    offs_x_imag = base_offset + (offs_pair * 2 + 1) * stride_x_dim

    x_real = tl.load(x_ptr + offs_x_real, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + offs_x_imag, mask=mask, other=0.0).to(tl.float32)

    offs_freq_real = position * stride_freq_pos + offs_pair * 2 * stride_freq_dim
    offs_freq_imag = position * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim

    freq_real = tl.load(freqs_ptr + offs_freq_real, mask=mask, other=0.0)
    freq_imag = tl.load(freqs_ptr + offs_freq_imag, mask=mask, other=0.0)

    if IS_INVERSE:
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + offs_x_real, out_real, mask=mask)
    tl.store(x_ptr + offs_x_imag, out_imag, mask=mask)


def apply_rotary_emb_triton(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> torch.Tensor:
    is_3d = x.ndim == 3

    if is_3d:
        batch_size, n_heads, rope_dim = x.shape
    else:
        batch_size, rope_dim = x.shape
        n_heads = 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    BLOCK_SIZE = 128

    num_blocks_dim = triton.cdiv(rope_dim // 2, BLOCK_SIZE)
    grid = (batch_size, n_heads if is_3d else 1, num_blocks_dim)

    if positions is not None:
        assert positions.shape == (
            batch_size,
        ), f"positions shape {positions.shape} != ({batch_size},)"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            positions,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=True,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        assert (
            freqs_real.shape[0] == batch_size
        ), f"freqs_cis batch size {freqs_real.shape[0]} != x batch size {batch_size}"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            None,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=False,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return x


# Cache for the contiguous real/imag halves of each freqs_cis tensor used in
# v4_rope_inplace_npu. complex freqs_cis.real / freqs_cis.imag are strided
# views (stride=2 on the underlying interleaved real layout);
_NPU_ROPE_CONTIG_CACHE: dict[int, tuple] = {}


def _get_contig_freqs_real_imag(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous (real, imag) halves of ``freqs_cis``, cached by id.

    Used by NPU rope paths to avoid the per-call StridedSlice materialization
    triggered by aclnnIndex over the strided ``.real`` / ``.imag`` views of
    the complex ``freqs_cis`` buffer. First call per freqs_cis pays the
    contiguous() once; later calls reuse the cached tensors.

    All callers within a single MQALayer (outer rope, indexer inner rope,
    compressor epilog rope) get the same freqs_cis instance, so each layer
    materializes at most one (real, imag) pair.
    """
    cache_key = id(freqs_cis)
    cached = _NPU_ROPE_CONTIG_CACHE.get(cache_key)
    if cached is None:
        cached = (freqs_cis.real.contiguous(), freqs_cis.imag.contiguous())
        _NPU_ROPE_CONTIG_CACHE[cache_key] = cached
    return cached


def get_fused_compressor_rope_cos_sin(
    freqs_cis: torch.Tensor,
    positions_cmp: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) tensors shaped ``[T, rope_head_dim]`` for the fused
    compressor op (``torch.ops.custom.compressor``).

    The op consumes ``rope_cos`` / ``rope_sin`` of shape
    ``[min(T, T//cmp_ratio + B), rope_head_dim]`` in bf16/fp16. We index
    the cached contig real/imag halves of the complex ``freqs_cis`` and
    interleave-double the last dim to match the kernel's expected layout
    (matches dsv4_release ``ComplexExpRotaryEmbedding.cos_cache``, which
    is built as ``complex_cache.real.repeat_interleave(2, dim=-1)``).

    Safe to call from inside a captured aclgraph: both ``index_select`` and
    ``repeat_interleave`` over a graph-input ``positions_cmp`` of fixed
    capture-time shape produce static-shape outputs. Identical to what the
    existing inplace_partial_rotary_mul fallback does at
    :func:`v4_rope_inplace_npu`, just without the inverse / 4D-view step.
    """
    real_contig, imag_contig = _get_contig_freqs_real_imag(freqs_cis)
    cos_half = real_contig.index_select(0, positions_cmp)
    sin_half = imag_contig.index_select(0, positions_cmp)
    cos = cos_half.repeat_interleave(2, dim=-1).to(dtype)
    sin = sin_half.repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def v4_rope_inplace_npu(
    q_rope: torch.Tensor,
    kv_rope: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """In-place interleaved RoPE for V4 — torch fallback used on NPU.

    Mirrors main's CUDA `fused_rope` kernel: consecutive (even, odd) pairs
    of x form complex pairs, with `freqs_cis` a complex tensor where
    `freqs_cis.real[t, k]` = cos(theta_{t,k}), `freqs_cis.imag` = sin(...)
    indexed by frequency pair k in [0, rope_dim/2).

    NOTE on V4-Flash YARN `mscale`: when the model was trained with the
    YARN magnitude-scale `mscale` ≠ 1.0, the cos/sin values stored in
    `freqs_cis` MUST already be pre-multiplied by `mscale` at precompute
    time — see `precompute_freqs_cis`. This function
    just reads what's stored; it does NOT apply mscale here.

    Prefer the NPU-native `torch.ops.custom.inplace_partial_rotary_mul`:
    the torch fallback differs by ~1 ULP per element vs the kernel because
    torch does bf16*bf16 muls with bf16 accumulation while the NPU kernel
    accumulates in fp32; 43 layers × (Q + K) = 86 rope calls compound that
    drift enough to flip argmax on marginal prompts.
    """
    # Build cos/sin caches in the layout the kernel expects:
    # (T, 1, 1, rope_dim) with each freq pair value repeated twice for
    # the interleaved pairing convention. Use contig real/imag views
    # cached by id(freqs_cis); see _get_contig_freqs_real_imag.
    freqs_real_contig, freqs_imag_contig = _get_contig_freqs_real_imag(freqs_cis)
    cos_half = freqs_real_contig[positions]  # (T, rope_dim/2)
    sin_half = freqs_imag_contig[positions]
    if inverse:
        sin_half = -sin_half
    cos_full = cos_half.repeat_interleave(2, dim=-1).to(q_rope.dtype)
    sin_full = sin_half.repeat_interleave(2, dim=-1).to(q_rope.dtype)
    rope_dim = cos_full.shape[-1]
    # repeat_interleave produces a contiguous tensor, so the .view()
    # below already returns a contiguous result — no .contiguous() needed.
    cos4 = cos_full.view(-1, 1, 1, rope_dim)
    sin4 = sin_full.view(-1, 1, 1, rope_dim)
    # q_rope: (T, n_heads, rope_dim) → (T, 1, n_heads, rope_dim) view
    # kv_rope: (T, 1, rope_dim) → (T, 1, 1, rope_dim) view
    q_view = q_rope.unsqueeze(1)
    torch.ops.custom.inplace_partial_rotary_mul(
        q_view,
        cos4,
        sin4,
        rotary_mode="interleave",
        partial_slice=[0, rope_dim],
    )
    if kv_rope is not None:
        if kv_rope.dim() == 3:
            kv_view = kv_rope.unsqueeze(1)
        else:
            kv_view = kv_rope.view(-1, 1, 1, rope_dim)
        torch.ops.custom.inplace_partial_rotary_mul(
            kv_view,
            cos4,
            sin4,
            rotary_mode="interleave",
            partial_slice=[0, rope_dim],
        )
