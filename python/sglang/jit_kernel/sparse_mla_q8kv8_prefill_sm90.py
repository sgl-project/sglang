"""JIT-compiled Q8KV8 sparse prefill attention kernel for SM90 (Hopper/H200).

Uses native FP8 GMMA instructions via CUTLASS/CUTE for MLA attention
with FP8 quantized Q and KV tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels._jit import cache_once, load_jit, override_jit_cuda_arch
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
    with override_jit_cuda_arch(9, 0, "a"):
        return load_jit(
            "sparse_mla_q8kv8_prefill_sm90",
            cuda_files=[
                "sparse_mla_q8kv8_prefill_sm90/entry.cuh",
            ],
            cuda_wrappers=[
                ("dispatch", "sparse_prefill_q8kv8_dispatch"),
                ("dispatch_full", "sparse_prefill_q8kv8_dispatch_full"),
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
    dispatch_fn, _ = _get_entries()
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
    _, dispatch_full_fn = _get_entries()
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
    are allocated and returned; callers that want to reuse buffers (e.g.
    for CUDA graph capture) may pass pre-allocated ``out`` / ``max_logits``
    / ``lse`` tensors of the expected shape/dtype/device. The three output
    tensors must not alias each other.

    Returns:
        out:        [s_q, h_q, d_v], bfloat16
        max_logits: [s_q, h_q], float32
        lse:        [s_q, h_q], float32
    """
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    h_kv = kv.shape[1]
    topk = indices.shape[2]

    if d_v != 512:
        raise ValueError(
            f"sparse_mla_q8kv8_prefill_fwd only supports d_v=512, got {d_v}"
        )

    if (attn_sink is None) != (topk_length is None):
        raise ValueError("attn_sink and topk_length must be provided together")

    device = q.device
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
