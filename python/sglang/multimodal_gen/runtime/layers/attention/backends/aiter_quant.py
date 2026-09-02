# SPDX-License-Identifier: Apache-2.0
#
# Quantized AITER attention family backend (ROCm / gfx950).
#
# One backend, several quant formats selected via --attention-backend-config
# (e.g. `format=mxfp4`).

import inspect
from collections.abc import Callable

import aiter
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import is_gfx95_supported

logger = init_logger(__name__)

# All aiter_quant kernels target full-MHA head_dim==128 models (e.g. Wan
# self-/cross-attention) on a gfx950-class arch. Selecting this backend is
# explicit, so unmet constraints raise rather than silently falling back.
_REQUIRED_HEAD_DIM = 128

# Hadamard block size for the fp8 path.
_HADAMARD_BLOCK_R = 128

_DEFAULT_FORMAT = "fp8"


# ---------------------------------------------------------------------------
# aiter.ops.mha_v4 imports (required by every format).
#
# Imported at module load so torch.compile sees stable symbols. If the installed
# aiter lacks mha_v4, the names resolve to None and construction raises a clear
# error.
# ---------------------------------------------------------------------------
try:
    from aiter.ops.mha_v4 import AttentionFormat as _AiterAttentionFormat
    from aiter.ops.mha_v4 import mha_v4_packed as _aiter_mha_v4_packed
    from aiter.ops.mha_v4 import mha_v4_q_multiplier as _aiter_mha_v4_q_multiplier
    from aiter.ops.mha_v4 import mxfp4_k_view as _aiter_mxfp4_k_view
    from aiter.ops.mha_v4 import mxfp4_v_view as _aiter_mxfp4_v_view
    from aiter.ops.mha_v4 import mxfp6_k_view as _aiter_mxfp6_k_view
    from aiter.ops.mha_v4 import native_fp8_format as _aiter_native_fp8_format
    from aiter.ops.mha_v4 import quantize_fp8 as _aiter_mha_v4_quantize_fp8
    from aiter.ops.mha_v4 import quantize_int8 as _aiter_mha_v4_quantize_int8
    from aiter.ops.mha_v4 import quantize_mxfp4_k as _aiter_quantize_mxfp4_k
    from aiter.ops.mha_v4 import quantize_mxfp4_q as _aiter_quantize_mxfp4_q
    from aiter.ops.mha_v4 import quantize_mxfp6_k as _aiter_quantize_mxfp6_k
    from aiter.ops.mha_v4 import quantize_mxfp6_q as _aiter_quantize_mxfp6_q
    from aiter.ops.mha_v4 import quantize_v_fp8 as _aiter_quantize_v_fp8
    from aiter.ops.mha_v4 import quantize_v_mxfp4 as _aiter_quantize_v_mxfp4
    from aiter.ops.mha_v4 import (
        scale_modes_for_formats as _aiter_scale_modes_for_formats,
    )

    _AITER_MHA_V4_AVAILABLE = True
except ImportError:
    # Keep the names defined (as None) so they remain patchable and referencing
    # them yields a clear message via the construction-time availability check.
    _AiterAttentionFormat = None
    _aiter_mha_v4_packed = None
    _aiter_mha_v4_q_multiplier = None
    _aiter_mxfp4_k_view = None
    _aiter_mxfp4_v_view = None
    _aiter_mxfp6_k_view = None
    _aiter_native_fp8_format = None
    _aiter_mha_v4_quantize_fp8 = None
    _aiter_mha_v4_quantize_int8 = None
    _aiter_quantize_mxfp4_k = None
    _aiter_quantize_mxfp4_q = None
    _aiter_quantize_mxfp6_k = None
    _aiter_quantize_mxfp6_q = None
    _aiter_quantize_v_fp8 = None
    _aiter_quantize_v_mxfp4 = None
    _aiter_scale_modes_for_formats = None

    _AITER_MHA_V4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hadamard rotation helpers (fp8 path only).
# ---------------------------------------------------------------------------
def _build_hadamard_matrix(
    block_r: int,
    dtype: torch.dtype = torch.bfloat16,
    allow_sylvester_fallback: bool = True,
) -> torch.Tensor | None:
    """Normalized Hadamard matrix (block_r x block_r, R @ R.T == I; block_r a
    power of two). Uses aiter's create_hadamard_matrix. If that's unavailable,
    falls back to a local Sylvester construction when allow_sylvester_fallback
    is set, otherwise returns None."""
    try:
        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
        except ImportError:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
                create_hadamard_matrix,
            )
        return create_hadamard_matrix(block_r, dtype=dtype) / (block_r**0.5)
    except ImportError:
        if not allow_sylvester_fallback:
            return None
        # Local Sylvester construction: H1=[[1]], H2n=[[Hn,Hn],[Hn,-Hn]].
        assert (
            block_r > 0 and (block_r & (block_r - 1)) == 0
        ), "Hadamard block_r must be a positive power of 2"
        H = torch.ones((1, 1), dtype=torch.float32)
        while H.shape[0] < block_r:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return (H / (block_r**0.5)).to(dtype)


def _replicate_hadamard_per_device(
    hadamard: torch.Tensor | None,
) -> dict[torch.device, torch.Tensor | None]:
    """Replicate a single Hadamard matrix on each available device, keyed by
    torch.device (all GPUs if CUDA is available, else CPU). A None matrix maps
    to None on every device."""
    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]
    return {
        device: (hadamard.to(device) if hadamard is not None else None)
        for device in devices
    }


def _aiter_hadamard_matrix(
    block_r: int, allow_sylvester_fallback: bool = True
) -> dict[torch.device, torch.Tensor | None]:
    """Build a normalized Hadamard matrix and replicate it across devices."""
    return _replicate_hadamard_per_device(
        _build_hadamard_matrix(
            block_r,
            dtype=torch.bfloat16,
            allow_sylvester_fallback=allow_sylvester_fallback,
        )
    )


FP8_HADAMARD_MATRIX = _aiter_hadamard_matrix(_HADAMARD_BLOCK_R)


def _fp8_hadamard_rotate(x: torch.Tensor, R: torch.Tensor | None) -> torch.Tensor:
    """Rotate the last (head_dim) axis by the Hadamard matrix R.

    Spreads outliers across dimensions to reduce quantization error while
    preserving attention scores (Q@K^T is invariant since R @ R.T == I).
    """
    if R is None:
        return x
    d = x.shape[-1]
    block_r = R.shape[-1]
    R = R.to(x.dtype)
    if block_r == d:
        return torch.matmul(x, R)
    return torch.matmul(x.unflatten(-1, (d // block_r, block_r)), R).flatten(-2)


def _aiter_fp8_has_descale() -> bool:
    """True if the installed aiter's flash_attn_fp8_pertensor_func accepts
    per-tensor descale vectors (q_descale/k_descale/v_descale)."""
    try:
        return (
            inspect.signature(aiter.flash_attn_fp8_pertensor_func).parameters.get(
                "q_descale"
            )
            is not None
        )
    except (AttributeError, TypeError, ValueError):
        return False


AITER_FP8_HAS_DESCALE = _aiter_fp8_has_descale()


# ---------------------------------------------------------------------------
# Quant ops + kernel launches, one torch.library.custom_op each.
# Separate Q/K/V quant ops let Ulysses overlap each preprocessing path with
# its independent all-to-all; the kernel launch stays a custom op so
# torch.compile observes the native mutation/aliasing contract.
# Quant-op fakes call the real aiter op since the packed shapes are
# format-specific and cheap to derive.
# ---------------------------------------------------------------------------


# --- fp8 -------------------------------------------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_fp8_attention", mutates_args=())
def _aiter_fp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    if is_causal:
        raise NotImplementedError(
            "MHA v4 FP8 attention does not support causal masking."
        )
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *_aiter_scale_modes_for_formats(fp8_format, fp8_format, fp8_format),
    )


@_aiter_fp8_attention.register_fake
def _aiter_fp8_attention_fake(
    query, key, value, q_descale, k_descale, v_descale, is_causal
):
    del key, q_descale, k_descale, v_descale, is_causal
    return query.new_empty(
        (query.shape[0], query.shape[1], query.shape[2], value.shape[-1]),
        dtype=torch.bfloat16,
    )


def _forward_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """fp8 per-tensor quantization + Hadamard-rotated Q/K via mha_v4_packed."""
    del softmax_scale  # fp8 mha_v4 uses the kernel's head_dim**-0.5 default.
    R = FP8_HADAMARD_MATRIX[query.device]
    # Rotate Q and K only; V is quantized but not rotated. Q@K^T is preserved
    # because R @ R.T == I.
    query = _fp8_hadamard_rotate(query, R).contiguous()
    key = _fp8_hadamard_rotate(key, R).contiguous()
    value = value.contiguous()

    quant_dtype = aiter.dtypes.fp8
    dtype_max = torch.finfo(quant_dtype).max
    # Dynamic per-tensor scale when descale vectors are supported, else a static
    # scale of 1.0 (no descale).
    scale = None
    if not AITER_FP8_HAS_DESCALE:
        scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

    quant_q, q_descale = aiter.per_tensor_quant(
        query, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
    )
    quant_k, k_descale = aiter.per_tensor_quant(
        key, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
    )
    quant_v, v_descale = aiter.per_tensor_quant(
        value, scale=scale, quant_dtype=quant_dtype, dtypeMax=dtype_max
    )

    # When descale vectors aren't supported the kernel still requires them, so
    # pass ones (the static scale of 1.0 already folded the quant scale in).
    if not AITER_FP8_HAS_DESCALE:
        q_descale = torch.ones((1,), dtype=torch.float32, device=query.device)
        k_descale = torch.ones((1,), dtype=torch.float32, device=query.device)
        v_descale = torch.ones((1,), dtype=torch.float32, device=query.device)

    return _aiter_fp8_attention(
        quant_q, quant_k, quant_v, q_descale, k_descale, v_descale, causal
    )


# --- i8fp8 -----------------------------------------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_i8fp8_quantize_q", mutates_args=())
def _aiter_i8fp8_quantize_q(
    query: torch.Tensor, clip: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_mha_v4_quantize_int8(query, clip)


@_aiter_i8fp8_quantize_q.register_fake
def _aiter_i8fp8_quantize_q_fake(query, clip=1.0):
    del clip
    return query.new_empty(query.shape, dtype=torch.int8), query.new_empty(
        (1,), dtype=torch.float32
    )


@torch.library.custom_op("sgl_diffusion::aiter_i8fp8_quantize_k", mutates_args=())
def _aiter_i8fp8_quantize_k(
    key: torch.Tensor, clip: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_mha_v4_quantize_int8(key, clip)


@_aiter_i8fp8_quantize_k.register_fake
def _aiter_i8fp8_quantize_k_fake(key, clip=1.0):
    del clip
    return key.new_empty(key.shape, dtype=torch.int8), key.new_empty(
        (1,), dtype=torch.float32
    )


@torch.library.custom_op("sgl_diffusion::aiter_i8fp8_quantize_v", mutates_args=())
def _aiter_i8fp8_quantize_v(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_mha_v4_quantize_fp8(value)


@_aiter_i8fp8_quantize_v.register_fake
def _aiter_i8fp8_quantize_v_fake(value):
    return value.new_empty(value.shape, dtype=aiter.dtypes.fp8), value.new_empty(
        (1,), dtype=torch.float32
    )


def _forward_i8fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """int8 Q/K + fp8 V via mha_v4_packed (no Hadamard rotation)."""
    del softmax_scale
    if causal:
        raise NotImplementedError(
            "MHA v4 I8FP8 attention does not support causal masking."
        )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    q_i8, q_descale = _aiter_i8fp8_quantize_q(query)
    k_i8, k_descale = _aiter_i8fp8_quantize_k(key)
    v_fp8, v_descale = _aiter_i8fp8_quantize_v(value)

    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_i8,
        k_i8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        _AiterAttentionFormat.INT8,
        _AiterAttentionFormat.INT8,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.INT8,
            _AiterAttentionFormat.INT8,
            fp8_format,
        ),
    )


# --- mxfp4 / f4f4 quant ops (Q/K shared) -----------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_mxfp4_quantize_q", mutates_args=())
def _aiter_mxfp4_quantize_q(
    query: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q-only rotate + fp4 (E2M1) pack for the hd128 mxfp4 kernel."""
    return _aiter_quantize_mxfp4_q(query, _aiter_mha_v4_q_multiplier(softmax_scale))


@_aiter_mxfp4_quantize_q.register_fake
def _aiter_mxfp4_quantize_q_fake(query, softmax_scale):
    return _aiter_quantize_mxfp4_q(query, _aiter_mha_v4_q_multiplier(softmax_scale))


@torch.library.custom_op("sgl_diffusion::aiter_mxfp4_quantize_k_raw", mutates_args=())
def _aiter_mxfp4_quantize_k_raw(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-only fused rotation + coalesced MXFP4 packing (contiguous raw buffers)."""
    return _aiter_quantize_mxfp4_k(key)


@_aiter_mxfp4_quantize_k_raw.register_fake
def _aiter_mxfp4_quantize_k_raw_fake(key):
    return _aiter_quantize_mxfp4_k(key)


@torch.library.custom_op("sgl_diffusion::aiter_mx_quantize_v", mutates_args=())
def _aiter_mx_quantize_v(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-only per-channel fp8 quantization shared by the MX backends."""
    return _aiter_quantize_v_fp8(value)


@_aiter_mx_quantize_v.register_fake
def _aiter_mx_quantize_v_fake(value):
    return _aiter_quantize_v_fp8(value)


@torch.library.custom_op("sgl_diffusion::aiter_f4_quantize_v_raw", mutates_args=())
def _aiter_f4_quantize_v_raw(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack true-MXFP4 V and return contiguous data + E8M0 scale buffers."""
    return _aiter_quantize_v_mxfp4(value)


@_aiter_f4_quantize_v_raw.register_fake
def _aiter_f4_quantize_v_raw_fake(value):
    return _aiter_quantize_v_mxfp4(value)


# --- mxfp4 kernel ----------------------------------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_mxfp4_kernel_raw", mutates_args=())
def _aiter_mxfp4_kernel_raw(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_buf: torch.Tensor,
    k_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Rebuild the K ABI view and invoke the ASM kernel behind a compile-safe
    custom-op boundary. fp4 (E2M1) Q/K + fp8 V."""
    k_fp4 = _aiter_mxfp4_k_view(k_buf, k_scale)
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_fp4,
        k_fp4,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
            fp8_format,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_mxfp4_kernel_raw.register_fake
def _aiter_mxfp4_kernel_raw_fake(
    q_fp4, q_scale, k_buf, k_scale, v_fp8, v_scale, softmax_scale
):
    # Attention output carries the QUERY seq_len (differs from V under cross-
    # attention), fp8-V head_dim, in bf16. q_fp4 is [B, S_q, H, packed].
    b, s, h, _ = q_fp4.shape
    return q_fp4.new_empty((b, s, h, v_fp8.shape[-1]), dtype=torch.bfloat16)


def _forward_mxfp4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """fp4 (E2M1) Q/K + fp8 V via mha_v4_packed. Hadamard rotation is fused into
    the fp4 quant op; softmax_scale is baked into the Q multiplier."""
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # The mxfp4 ASM kernel expects head_dim**-0.5 baked into the Q multiplier.
    softmax_scale = query.shape[-1] ** -0.5

    q_fp4, q_scale = _aiter_mxfp4_quantize_q(query, softmax_scale)
    k_fp4, k_scale = _aiter_mxfp4_quantize_k_raw(key)
    v_fp8, v_scale = _aiter_mx_quantize_v(value)
    return _aiter_mxfp4_kernel_raw(
        q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, softmax_scale
    )


# --- f4f4 kernel -----------------------------------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_f4f4_kernel_raw", mutates_args=())
def _aiter_f4f4_kernel_raw(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_buf: torch.Tensor,
    k_scale: torch.Tensor,
    v_buf: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    kv_len: int,
) -> torch.Tensor:
    """true-MXFP4 Q/K/V via mha_v4_packed."""
    v_fp4 = _aiter_mxfp4_v_view(v_buf, v_scale, kv_len)
    k_fp4 = _aiter_mxfp4_k_view(k_buf, k_scale)
    return _aiter_mha_v4_packed(
        q_fp4,
        k_fp4,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_f4f4_kernel_raw.register_fake
def _aiter_f4f4_kernel_raw_fake(
    q_fp4, q_scale, k_buf, k_scale, v_buf, v_scale, softmax_scale, kv_len
):
    b, s, h, _ = q_fp4.shape
    return q_fp4.new_empty((b, s, h, q_scale.shape[-1] * 32), dtype=torch.bfloat16)


def _forward_f4f4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """true-MXFP4 Q/K/V via mha_v4_packed."""
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    softmax_scale = query.shape[-1] ** -0.5
    q_fp4, q_scale = _aiter_mxfp4_quantize_q(query, softmax_scale)
    k_fp4, k_scale = _aiter_mxfp4_quantize_k_raw(key)
    v_buf, v_scale = _aiter_f4_quantize_v_raw(value)
    return _aiter_f4f4_kernel_raw(
        q_fp4, q_scale, k_fp4, k_scale, v_buf, v_scale, softmax_scale, value.shape[1]
    )


# --- mxfp6 / f6f4 quant ops (Q/K shared) -----------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_mxfp6_quantize_q", mutates_args=())
def _aiter_mxfp6_quantize_q(
    query: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q-only fused native hd128 rotate + fp6 pack."""
    return _aiter_quantize_mxfp6_q(query, _aiter_mha_v4_q_multiplier(softmax_scale))


@_aiter_mxfp6_quantize_q.register_fake
def _aiter_mxfp6_quantize_q_fake(query, softmax_scale):
    return _aiter_quantize_mxfp6_q(query, _aiter_mha_v4_q_multiplier(softmax_scale))


@torch.library.custom_op("sgl_diffusion::aiter_mxfp6_quantize_k_raw", mutates_args=())
def _aiter_mxfp6_quantize_k_raw(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-only fused Hadamard+fp6 pack, returned as contiguous raw ABI buffers."""
    return _aiter_quantize_mxfp6_k(key)


@_aiter_mxfp6_quantize_k_raw.register_fake
def _aiter_mxfp6_quantize_k_raw_fake(key):
    return _aiter_quantize_mxfp6_k(key)


# --- mxfp6 kernel ----------------------------------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_mxfp6_kernel_raw", mutates_args=())
def _aiter_mxfp6_kernel_raw(
    k_buf: torch.Tensor,
    k_scale_buf: torch.Tensor,
    q_fp6: torch.Tensor,
    q_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Rebuild the exotic K ABI view from contiguous buffers and invoke the asm
    kernel. fp6 (E2M3) Q/K + fp8 V."""
    b, s, h, _ = v_fp8.shape
    k_fp6, k_scale = _aiter_mxfp6_k_view(k_buf, k_scale_buf, b, s, h)
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_fp6,
        k_fp6,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP6,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP6,
            fp8_format,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_mxfp6_kernel_raw.register_fake
def _aiter_mxfp6_kernel_raw_fake(
    k_buf, k_scale_buf, q_fp6, q_scale, v_fp8, v_scale, softmax_scale
):
    # Attention output carries the QUERY seq_len (differs from V under cross-
    # attention), fp8-V head_dim, in bf16. q_fp6 is [B, S_q, H, packed].
    b, s, h, _ = q_fp6.shape
    return q_fp6.new_empty((b, s, h, v_fp8.shape[-1]), dtype=torch.bfloat16)


def _forward_mxfp6(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """fp6 (E2M3) Q/K + fp8 V via mha_v4_packed."""
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    softmax_scale = query.shape[-1] ** -0.5
    q_fp6, q_scale = _aiter_mxfp6_quantize_q(query, softmax_scale)
    k_buf, k_scale_buf = _aiter_mxfp6_quantize_k_raw(key)
    v_fp8, v_scale = _aiter_mx_quantize_v(value)
    return _aiter_mxfp6_kernel_raw(
        k_buf, k_scale_buf, q_fp6, q_scale, v_fp8, v_scale, softmax_scale
    )


# --- f6f4 kernel -----------------------------------------------------------
@torch.library.custom_op("sgl_diffusion::aiter_f6f4_kernel_raw", mutates_args=())
def _aiter_f6f4_kernel_raw(
    k_buf: torch.Tensor,
    k_scale_buf: torch.Tensor,
    q_fp6: torch.Tensor,
    q_scale: torch.Tensor,
    v_buf: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    kv_len: int,
) -> torch.Tensor:
    """MXFP6 Q/K + true-MXFP4 V via mha_v4_packed."""
    v_fp4 = _aiter_mxfp4_v_view(v_buf, v_scale, kv_len)
    b, _, h, _ = v_fp4.shape
    k_fp6, k_scale = _aiter_mxfp6_k_view(k_buf, k_scale_buf, b, kv_len, h)
    return _aiter_mha_v4_packed(
        q_fp6,
        k_fp6,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP4,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP4,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_f6f4_kernel_raw.register_fake
def _aiter_f6f4_kernel_raw_fake(
    k_buf, k_scale_buf, q_fp6, q_scale, v_buf, v_scale, softmax_scale, kv_len
):
    b, s, h, _ = q_fp6.shape
    return q_fp6.new_empty((b, s, h, q_scale.shape[-1] * 32), dtype=torch.bfloat16)


def _forward_f6f4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """MXFP6 Q/K + true-MXFP4 V via mha_v4_packed."""
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    softmax_scale = query.shape[-1] ** -0.5
    q_fp6, q_scale = _aiter_mxfp6_quantize_q(query, softmax_scale)
    k_buf, k_scale_buf = _aiter_mxfp6_quantize_k_raw(key)
    v_buf, v_scale = _aiter_f4_quantize_v_raw(value)
    return _aiter_f6f4_kernel_raw(
        k_buf,
        k_scale_buf,
        q_fp6,
        q_scale,
        v_buf,
        v_scale,
        softmax_scale,
        value.shape[1],
    )


# format name -> forward pipeline. Every format requires aiter.ops.mha_v4 to be
# importable (checked once at construction).
_FORMATS: dict[str, Callable[..., torch.Tensor]] = {
    "fp8": _forward_fp8,
    "i8fp8": _forward_i8fp8,
    "mxfp4": _forward_mxfp4,
    "mxfp6": _forward_mxfp6,
    "f4f4": _forward_f4f4,
    "f6f4": _forward_f6f4,
}


def _resolve_format() -> str:
    """Read the `format` key from --attention-backend-config (default fp8)."""
    cfg = get_global_server_args().attention_backend_config or {}
    return str(cfg.get("format", _DEFAULT_FORMAT)).lower()


class AITERQuantBackend(AttentionBackend):
    """AITER quantized attention family backend (ROCm)."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_QUANT

    @staticmethod
    def get_impl_cls() -> type["AITERQuantImpl"]:
        return AITERQuantImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITER quant backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER quant backend does not have a metadata builder."
        )


class AITERQuantImpl(AttentionImpl):
    """Quantized attention via aiter, with the variant selected by the `format`
    key of --attention-backend-config (fp8, i8fp8, mxfp4, mxfp6, f4f4, f6f4)."""

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
        fmt = _resolve_format()
        forward = _FORMATS.get(fmt)
        if forward is None:
            raise ValueError(
                f"Unknown aiter_quant format {fmt!r}. Set "
                "--attention-backend-config format=<name> to one of: "
                f"{', '.join(sorted(_FORMATS))}."
            )

        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITER quant backend does not support Grouped Query Attention "
                f"(num_heads={num_heads}, num_kv_heads={num_kv_heads})."
            )
        if head_size != _REQUIRED_HEAD_DIM:
            raise NotImplementedError(
                f"AITER quant backend requires head_dim == {_REQUIRED_HEAD_DIM}, "
                f"got {head_size}."
            )
        if not is_gfx95_supported():
            raise RuntimeError("AITER quant backend requires a gfx950-class arch.")
        if not _AITER_MHA_V4_AVAILABLE:
            raise RuntimeError(
                "AITER quant backend requires aiter.ops.mha_v4, which is not "
                "available in the installed aiter build."
            )

        self.format = fmt
        self._forward = forward
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

        # Deduped per message, so this logs once per format for the whole run.
        logger.info_once(f"aiter_quant attention backend using format={fmt}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs quantized attention using the configured format.

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        return self._forward(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )

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
        raise NotImplementedError(
            "AITER quant backend does not support varlen attention."
        )
