"""JIT-compiled Q8KV8 sparse prefill attention kernel for SM90 (Hopper/H200).

Uses native FP8 GMMA instructions via CUTLASS/CUTE for MLA attention
with FP8 quantized Q and KV tensors.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# ---------------------------------------------------------------------------
# Helpers for CUTLASS include resolution (shared pattern with nvfp4.py)
# ---------------------------------------------------------------------------


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


def _resolve_cutlass_include_paths() -> list[str]:
    include_paths: list[str] = []

    # Prefer FlashMLA's bundled CUTLASS (vendored inside sgl-kernel build deps)
    # over flashinfer/deep_gemm's CUTLASS. The Q8KV8 kernel was developed
    # against FlashMLA's CUTLASS version; mismatch between flashinfer's 2025
    # CUTLASS and FlashMLA's 2026 CUTLASS produces correct LSE/max_logits but
    # catastrophically wrong P*V output (~1e30 vs correct ~1e-3). Using the
    # same CUTLASS as the upstream prebuilt kernel resolves this.
    _here = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    flashmla_cutlass_candidates = [
        # When sgl-kernel has been built (sibling dir)
        _here.parent
        / "sgl-kernel"
        / "build"
        / "_deps"
        / "repo-flashmla-src"
        / "csrc"
        / "cutlass",
    ]
    for base in flashmla_cutlass_candidates:
        inc = base / "include"
        tu_inc = base / "tools" / "util" / "include"
        if inc.exists():
            include_paths.append(str(inc))
            if tu_inc.exists():
                include_paths.append(str(tu_inc))
            break

    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is not None:
        candidates = [
            flashinfer_root / "data" / "cutlass" / "include",
            flashinfer_root / "data" / "cutlass" / "tools" / "util" / "include",
        ]
        for path in candidates:
            if path.exists():
                include_paths.append(str(path))

    deep_gemm_root = _find_package_root("deep_gemm")
    if deep_gemm_root is not None:
        candidate = deep_gemm_root / "include"
        if candidate.exists():
            include_paths.append(str(candidate))

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in include_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _q8kv8_cuda_flags() -> list[str]:
    # These flags must match the prebuilt FlashMLA setup.py nvcc flags.
    # Missing `-U__CUDA_NO_BFLOAT16_CONVERSIONS__` (and the other -U flags)
    # causes the kernel to miscompile: attention output (P*V writeback)
    # becomes garbage while LSE/max_logits are still correct.
    return [
        "-O3",
        "-lineinfo",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


@cache_once
def _jit_sparse_mla_q8kv8_prefill_module() -> Module:
    extra_include_paths = _resolve_cutlass_include_paths()
    if not extra_include_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for Q8KV8 FlashMLA JIT kernel. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )

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
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=_q8kv8_cuda_flags(),
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
_q8kv8_outbuf_cache: dict = {}


def _q8kv8_get_outbufs(s_q: int, h_q: int, d_v: int, device: torch.device):
    key = (device, h_q, d_v)
    entry = _q8kv8_outbuf_cache.get(key)
    if entry is None or entry[0].shape[0] < s_q:
        out = torch.empty(s_q, h_q, d_v, dtype=torch.bfloat16, device=device)
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
        _q8kv8_outbuf_cache[key] = (out, max_logits, lse)
    else:
        out, max_logits, lse = entry
    return out[:s_q], max_logits[:s_q], lse[:s_q]


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Q8KV8 (FP8) sparse prefill attention on SM90.

    Returns:
        out:        [s_q, h_q, d_v], bfloat16
        max_logits: [s_q, h_q], float32
        lse:        [s_q, h_q], float32
    """
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    h_kv = kv.shape[1]
    topk = indices.shape[2]

    if (attn_sink is None) != (topk_length is None):
        raise ValueError("attn_sink and topk_length must be provided together")

    out, max_logits, lse = _q8kv8_get_outbufs(s_q, h_q, d_v, q.device)

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
