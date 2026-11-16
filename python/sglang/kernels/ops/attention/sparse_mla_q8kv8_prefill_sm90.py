"""JIT-compiled Q8KV8 sparse prefill attention kernel for SM90 (Hopper/H200).

Uses native FP8 GMMA instructions via CUTLASS/CUTE for MLA attention
with FP8 quantized Q and KV tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    fp8_dtype,
)
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
    sm_scale: float,
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
        sm_scale,
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
    sm_scale: float,
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
        sm_scale,
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
    sm_scale: float,
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
        sm_scale,
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
    # Shapes, dtypes, devices and contiguity are checked by the C++ launcher (entry.cuh).
    s_q, h_q = q.shape[0], q.shape[1]
    device = q.device
    # d_v only sizes a freshly allocated out; the launcher checks a caller's out against 512.
    if d_v != 512:
        raise ValueError(
            f"sparse_mla_q8kv8_prefill_fwd only supports d_v=512, got {d_v}"
        )
    if attn_sink is not None and topk_length is None:
        raise ValueError("attn_sink requires topk_length to be provided as well")
    if topk_length is not None:
        topk = indices.shape[-1]
        if torch.any(topk_length < 0).item() or torch.any(topk_length > topk).item():
            raise ValueError(
                f"topk_length values must satisfy 0 <= topk_length <= topk ({topk})"
            )

    if out is None:
        out = torch.empty(s_q, h_q, d_v, dtype=torch.bfloat16, device=device)

    if max_logits is None:
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)

    if lse is None:
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)

    # The three output tensors are written independently by the kernel; any
    # aliasing among them would corrupt results, so reject it explicitly.
    out_ptr = out.data_ptr()
    ml_ptr = max_logits.data_ptr()
    lse_ptr = lse.data_ptr()
    if out_ptr == ml_ptr or out_ptr == lse_ptr or ml_ptr == lse_ptr:
        raise ValueError("out, max_logits and lse must not alias each other")

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
            sm_scale,
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
            sm_scale,
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
            sm_scale,
        )

    return out, max_logits, lse


def q8kv8_padded_num_heads(num_heads: int) -> int:
    """Return a Q-head count supported by the SM90 Q8KV8 kernel."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if num_heads <= 64:
        return 64
    if num_heads <= 128:
        return 128
    raise ValueError(
        "DeepSeek-V4 Q8KV8 sparse prefill supports at most 128 local "
        f"query heads, got {num_heads}"
    )


def cast_q_fp8_for_q8kv8_prefill(
    q: torch.Tensor,
    padded_num_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast DeepSeek-V4 sparse-prefill Q to the Q8KV8 kernel format.

    The incoming Q is the model-produced BF16/FP16 tensor already shaped as
    ``(num_tokens, num_heads, 512)`` after removing the singleton MQA axis.

    The SM90 kernel processes query heads in 64-head blocks. Tensor parallelism
    commonly leaves fewer than 64 local heads, so the active heads are copied
    into a zero-padded 64/128-head FP8 tensor.
    """
    assert q.ndim == 3
    assert q.shape[-1] == DIM_NOPE + DIM_ROPE

    num_tokens, num_heads, head_dim = q.shape
    if padded_num_heads is None:
        padded_num_heads = q8kv8_padded_num_heads(num_heads)

    if padded_num_heads not in (64, 128) or padded_num_heads < num_heads:
        raise ValueError(
            f"invalid padded_num_heads={padded_num_heads} for num_heads={num_heads}"
        )

    expected_shape = (num_tokens, padded_num_heads, head_dim)

    if out is None:
        q_fp8 = torch.zeros(
            expected_shape,
            dtype=fp8_dtype,
            device=q.device,
        )
    else:
        if (
            out.shape != expected_shape
            or out.dtype != fp8_dtype
            or out.device != q.device
        ):
            raise ValueError(
                "Q8KV8 Q output must have shape/dtype/device "
                f"{expected_shape}/{fp8_dtype}/{q.device}, got "
                f"{tuple(out.shape)}/{out.dtype}/{out.device}"
            )
        q_fp8 = out
        if padded_num_heads > num_heads:
            q_fp8[:, num_heads:].zero_()

    q_fp8[:, :num_heads].copy_(q)
    q_scale = torch.ones((), dtype=torch.float32, device=q.device)
    return q_fp8, q_scale
