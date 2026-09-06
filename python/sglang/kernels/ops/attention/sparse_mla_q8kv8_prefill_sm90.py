"""JIT-compiled Q8KV8 sparse prefill attention kernel for SM90 (Hopper/H200).

Uses native FP8 GMMA instructions via CUTLASS/CUTE for MLA attention
with FP8 quantized Q and KV tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# ---------------------------------------------------------------------------
# Build flags
# ---------------------------------------------------------------------------


def _q8kv8_cuda_flags() -> list[str]:
    # Minimal flag set, verified by per-flag ablation on SM90/H200 (CUDA 12.9).
    # The original list was lifted from DeepSeek FlashMLA's AOT setup.py; under
    # this tvm_ffi JIT build only --use_fast_math has any measurable effect, so
    # the rest are dropped.
    #
    # --use_fast_math maps the softmax exp2f to the ex2.approx.f32 MUFU op. Cost
    # of removing it: ~+4.3% at short-context / large-topk (s_kv=8192,
    # topk=2048), ~+1-2% mid, ~0% at long context -- with no accuracy change
    # (its ~2^-22 relative error is far below the fp8-e4m3 quantization noise).
    #
    # Dropped, all confirmed to leave perf and accuracy bit-identical here:
    #   * -U__CUDA_NO_HALF*/__CUDA_NO_BFLOAT16_CONVERSIONS__: these only matter
    #     when the toolchain pre-defines the matching -D__CUDA_NO_* macros, as
    #     torch.utils.cpp_extension's AOT path does (COMMON_NVCC_FLAGS). The JIT
    #     toolchain never defines them, so undefining is a no-op.
    #   * --expt-relaxed-constexpr and -O3: already supplied by the JIT default
    #     target flags (see utils.arch.get_default_target_flags).
    #   * --expt-extended-lambda, -lineinfo, -D_USE_MATH_DEFINES: not required
    #     by this single-translation-unit kernel.
    return [
        "-O3",
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "--use_fast_math",
    ]


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


@cache_once
def _jit_sparse_mla_q8kv8_prefill_module() -> Module:
    return load_jit(
        "sparse_mla_q8kv8_prefill_sm90",
        cuda_files=[
            "sparse_mla_q8kv8_prefill_sm90/entry.cuh",
        ],
        cuda_wrappers=[
            ("dispatch", "sparse_prefill_q8kv8_dispatch"),
            ("dispatch_full", "sparse_prefill_q8kv8_dispatch_full"),
            ("dispatch_topk_length", "sparse_prefill_q8kv8_dispatch_topk_length"),
        ],
        extra_cuda_cflags=_q8kv8_cuda_flags(),
        extra_dependencies=["cutlass"],
    )


# Pre-resolve entry-point callables on first use to avoid per-call module
# dictionary lookups.
_resolved_entries: Optional[tuple] = None


def _get_entries() -> tuple:
    global _resolved_entries
    if _resolved_entries is None:
        m = _jit_sparse_mla_q8kv8_prefill_module()
        _resolved_entries = (
            m["dispatch"],
            m["dispatch_full"],
            m["dispatch_topk_length"],
        )
    return _resolved_entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# torch._C._cuda_getCurrentRawStream returns the cudaStream_t pointer expected
# by the JIT wrapper. torch._C._cuda_getCurrentStream returns a packed stream
# id and must not be used here.
_get_current_stream_raw = torch._C._cuda_getCurrentRawStream


# Module-level cache for kernel-write-only output tensors. The active s_q rows
# are overwritten every call; buffers grow monotonically by device/head shape.
def _check_out_buffer(
    t: torch.Tensor,
    name: str,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    if t.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {t.dtype}")
    if t.device != device:
        raise ValueError(f"{name} must be on device {device}, got {t.device}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


# Internal custom-op wrappers so the JIT kernel calls participate in
# torch.library / torch.compile tracing and kernel-API debug logging.
# The dispatch_full variant carries the optional attn_sink / topk_length
# tensors as required args; the public API chooses which op to call.
@register_custom_op(
    op_name="sparse_mla_q8kv8_prefill",
    mutates_args=["out", "max_logits", "lse"],
)
def _sparse_mla_q8kv8_prefill_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    s_q: int,
    s_kv: int,
    h_q: int,
    h_kv: int,
    d_qk: int,
    d_v: int,
    topk: int,
    sm_scale: float,
    cuda_stream: int,
) -> None:
    dispatch_fn, _, _ = _get_entries()
    dispatch_fn(
        q,
        kv,
        indices,
        q_scale,
        kv_scale,
        out,
        max_logits,
        lse,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        d_v,
        topk,
        sm_scale,
        cuda_stream,
    )


@register_custom_op(
    op_name="sparse_mla_q8kv8_prefill_full",
    mutates_args=["out", "max_logits", "lse"],
)
def _sparse_mla_q8kv8_prefill_full_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    s_q: int,
    s_kv: int,
    h_q: int,
    h_kv: int,
    d_qk: int,
    d_v: int,
    topk: int,
    sm_scale: float,
    cuda_stream: int,
) -> None:
    _, dispatch_full_fn, _ = _get_entries()
    dispatch_full_fn(
        q,
        kv,
        indices,
        q_scale,
        kv_scale,
        attn_sink,
        topk_length,
        out,
        max_logits,
        lse,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        d_v,
        topk,
        sm_scale,
        cuda_stream,
    )


@register_custom_op(
    op_name="sparse_mla_q8kv8_prefill_topk_length",
    mutates_args=["out", "max_logits", "lse"],
)
def _sparse_mla_q8kv8_prefill_topk_length_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    topk_length: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    s_q: int,
    s_kv: int,
    h_q: int,
    h_kv: int,
    d_qk: int,
    d_v: int,
    topk: int,
    sm_scale: float,
    cuda_stream: int,
) -> None:
    _, _, dispatch_topk_length_fn = _get_entries()
    dispatch_topk_length_fn(
        q,
        kv,
        indices,
        q_scale,
        kv_scale,
        topk_length,
        out,
        max_logits,
        lse,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        d_v,
        topk,
        sm_scale,
        cuda_stream,
    )


@debug_kernel_api
def sparse_mla_q8kv8_prefill_fwd(
    q: torch.Tensor,  # [s_q, h_q, d_qk], float8_e4m3fn
    kv: torch.Tensor,  # [s_kv, h_kv, d_qk], float8_e4m3fn
    indices: torch.Tensor,  # [s_q, h_kv, topk], int32
    sm_scale: float,
    q_scale: torch.Tensor,  # scalar tensor on GPU, float32
    kv_scale: torch.Tensor,  # scalar tensor on GPU, float32
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,  # [h_q], float32
    topk_length: Optional[torch.Tensor] = None,  # [s_q], int32
    *,
    out: Optional[torch.Tensor] = None,  # [s_q, h_q, d_v], bfloat16
    max_logits: Optional[torch.Tensor] = None,  # [s_q, h_q], float32
    lse: Optional[torch.Tensor] = None,  # [s_q, h_q], float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Q8KV8 (FP8) sparse prefill attention on SM90.

    The kernel writes into three output tensors. By default fresh tensors
    are allocated and returned; callers that want to reuse buffers may pass
    pre-allocated ``out`` / ``max_logits`` / ``lse`` tensors of the expected
    shape/dtype/device. The three output tensors must not alias each other.

    Returns:
        out:        [s_q, h_q, d_v], bfloat16
        max_logits: [s_q, h_q], float32
        lse:        [s_q, h_q], float32
    """
    # Validate ranks before unpacking shapes so malformed callers fail with a
    # clear error instead of a Python unpacking/indexing exception.
    if q.ndim != 3:
        raise ValueError(f"q must have shape (s_q, h_q, d_qk), got {tuple(q.shape)}")
    if kv.ndim != 3:
        raise ValueError(
            f"kv must have shape (s_kv, h_kv, d_qk), got {tuple(kv.shape)}"
        )
    if indices.ndim != 3:
        raise ValueError(
            f"indices must have shape (s_q, h_kv, topk), got {tuple(indices.shape)}"
        )

    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, kv_d_qk = kv.shape
    topk = indices.shape[2]
    device = q.device

    # entry.cuh interprets q/kv as contiguous FP8 buffers and launches all
    # accesses on q's CUDA device. Reject contract violations before launch.
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if not kv.is_cuda:
        raise ValueError("kv must be a CUDA tensor")
    if not indices.is_cuda:
        raise ValueError("indices must be a CUDA tensor")

    if kv.device != device:
        raise ValueError(f"kv must be on q's device {device}, got {kv.device}")
    if indices.device != device:
        raise ValueError(
            f"indices must be on q's device {device}, got {indices.device}"
        )

    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q must be torch.float8_e4m3fn, got {q.dtype}")
    if kv.dtype != torch.float8_e4m3fn:
        raise ValueError(f"kv must be torch.float8_e4m3fn, got {kv.dtype}")

    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if not kv.is_contiguous():
        raise ValueError("kv must be contiguous")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")

    if kv_d_qk != d_qk:
        raise ValueError(f"kv d_qk must match q d_qk={d_qk}, got {kv_d_qk}")

    # The CUDA implementation uses B_H=64 and launches h_q / B_H CTAs.
    # Reject unpadded TP-local head counts instead of launching zero CTAs and
    # returning uninitialized outputs, which can appear to callers as a hang or
    # a later collective failure.
    if h_q == 0 or h_q % 64 != 0:
        raise ValueError(
            "sparse_mla_q8kv8_prefill_fwd requires h_q padded to a positive "
            f"multiple of 64, got {h_q}"
        )

    if h_kv != 1:
        raise ValueError(f"sparse_mla_q8kv8_prefill_fwd requires h_kv=1, got {h_kv}")

    if d_qk not in (512, 576):
        raise ValueError(
            f"sparse_mla_q8kv8_prefill_fwd supports d_qk=512/576, got {d_qk}"
        )

    if indices.shape[:2] != (s_q, h_kv):
        raise ValueError(
            f"indices must have shape ({s_q}, {h_kv}, topk), got {tuple(indices.shape)}"
        )

    if indices.dtype != torch.int32:
        raise ValueError(f"indices must be int32, got {indices.dtype}")

    if topk == 0 or topk % 128 != 0:
        raise ValueError(
            "Q8KV8 sparse-prefill topk width must be a positive multiple of 128, "
            f"got {topk}"
        )

    if topk_length is not None:
        if topk_length.shape != (s_q,) or topk_length.dtype != torch.int32:
            raise ValueError(
                f"topk_length must be int32 with shape ({s_q},), got "
                f"{tuple(topk_length.shape)}/{topk_length.dtype}"
            )
        if not topk_length.is_cuda:
            raise ValueError("topk_length must be a CUDA tensor")
        if topk_length.device != device:
            raise ValueError(
                f"topk_length must be on q's device {device}, got {topk_length.device}"
            )
        if not topk_length.is_contiguous():
            raise ValueError("topk_length must be contiguous")
        if torch.any(topk_length < 0).item() or torch.any(topk_length > topk).item():
            raise ValueError(
                f"topk_length values must satisfy 0 <= topk_length <= topk ({topk})"
            )

    if d_v != 512:
        raise ValueError(
            f"sparse_mla_q8kv8_prefill_fwd only supports d_v=512, got {d_v}"
        )

    if attn_sink is not None and topk_length is None:
        raise ValueError("attn_sink requires topk_length to be provided as well")

    if attn_sink is not None:
        if attn_sink.shape != (h_q,) or attn_sink.dtype != torch.float32:
            raise ValueError(
                f"attn_sink must be float32 with shape ({h_q},), got "
                f"{tuple(attn_sink.shape)}/{attn_sink.dtype}"
            )
        if not attn_sink.is_cuda:
            raise ValueError("attn_sink must be a CUDA tensor")
        if attn_sink.device != device:
            raise ValueError(
                f"attn_sink must be on q's device {device}, got {attn_sink.device}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous")

    for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
        if not isinstance(scale, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if not scale.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if scale.device != device:
            raise ValueError(
                f"{name} must be on q's device {device}, got {scale.device}"
            )
        if scale.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {scale.dtype}")
        if scale.numel() != 1:
            raise ValueError(
                f"{name} must be a scalar tensor, got shape {tuple(scale.shape)}"
            )
        if not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if out is None:
        out = torch.empty(s_q, h_q, d_v, dtype=torch.bfloat16, device=device)
    else:
        _check_out_buffer(out, "out", (s_q, h_q, d_v), torch.bfloat16, device)

    if max_logits is None:
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        _check_out_buffer(max_logits, "max_logits", (s_q, h_q), torch.float32, device)

    if lse is None:
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        _check_out_buffer(lse, "lse", (s_q, h_q), torch.float32, device)

    # The three output tensors are written independently by the kernel; any
    # aliasing among them would corrupt results, so reject it explicitly.
    out_ptr = out.data_ptr()
    ml_ptr = max_logits.data_ptr()
    lse_ptr = lse.data_ptr()
    if out_ptr == ml_ptr or out_ptr == lse_ptr or ml_ptr == lse_ptr:
        raise ValueError("out, max_logits and lse must not alias each other")

    cuda_stream = _get_current_stream_raw(q.device.index)

    if attn_sink is not None and topk_length is not None:
        _sparse_mla_q8kv8_prefill_full_op(
            q,
            kv,
            indices,
            q_scale,
            kv_scale,
            attn_sink,
            topk_length,
            out,
            max_logits,
            lse,
            s_q,
            s_kv,
            h_q,
            h_kv,
            d_qk,
            d_v,
            topk,
            sm_scale,
            cuda_stream,
        )
    elif topk_length is not None:
        _sparse_mla_q8kv8_prefill_topk_length_op(
            q,
            kv,
            indices,
            q_scale,
            kv_scale,
            topk_length,
            out,
            max_logits,
            lse,
            s_q,
            s_kv,
            h_q,
            h_kv,
            d_qk,
            d_v,
            topk,
            sm_scale,
            cuda_stream,
        )
    else:
        _sparse_mla_q8kv8_prefill_op(
            q,
            kv,
            indices,
            q_scale,
            kv_scale,
            out,
            max_logits,
            lse,
            s_q,
            s_kv,
            h_q,
            h_kv,
            d_qk,
            d_v,
            topk,
            sm_scale,
            cuda_stream,
        )

    return out, max_logits, lse
