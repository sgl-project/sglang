"""Declarative DSA backend compatibility table (step 1 of RFC #31774).

Single source for DSA backend defaulting, validation, and the per-cell test.
Undeclared cells are UNKNOWN: never defaulted, never rejected. Torch-free:
device facts are passed in by the caller.
"""

from typing import Dict, List, Optional, Tuple

import msgspec

# sm90 = CUDA below SM100; hip = any ROCm device.
PLATFORM_SM90 = "sm90"
PLATFORM_SM100 = "sm100"
PLATFORM_HIP = "hip"
PLATFORMS = (PLATFORM_SM90, PLATFORM_SM100, PLATFORM_HIP)

# The resolver only distinguishes fp8_e4m3 from everything else.
KV_FP8 = "fp8_e4m3"
KV_BF16 = "bfloat16"
KV_BUCKETS = (KV_BF16, KV_FP8)

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASES = (PHASE_PREFILL, PHASE_DECODE)

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_UNKNOWN = "unknown"


class DSACell(msgspec.Struct, frozen=True, kw_only=True):
    backend: str
    platform: str
    kv_dtype: str
    phase: str
    status: str
    # highest supported priority wins defaulting; None = never a default
    default_priority: Optional[int] = None
    # unsupported cells: user-facing why
    reason: str = ""
    # PR/issue or code reference backing the status
    evidence: str = ""


def _supported(
    backend: str,
    platform: str,
    kv_dtype: str,
    phase: str,
    *,
    priority: Optional[int] = None,
    evidence: str = "",
) -> DSACell:
    return DSACell(
        backend=backend,
        platform=platform,
        kv_dtype=kv_dtype,
        phase=phase,
        status=STATUS_SUPPORTED,
        default_priority=priority,
        evidence=evidence,
    )


def _unsupported(
    backend: str,
    platform: str,
    kv_dtype: str,
    phase: str,
    *,
    reason: str,
    evidence: str,
) -> DSACell:
    return DSACell(
        backend=backend,
        platform=platform,
        kv_dtype=kv_dtype,
        phase=phase,
        status=STATUS_UNSUPPORTED,
        reason=reason,
        evidence=evidence,
    )


_TILELANG_FP8_CUDA_REASON = (
    "the CUDA tilelang sparse kernels hardcode a bfloat16 KV cache; the fp8 "
    "path exists on ROCm only"
)
_Q8_DECODE_REASON = (
    "flashmla_sparse_q8 is a prefill-only backend; for FP8 decode use "
    "flashmla_kv (SM90) or trtllm (SM100)"
)

# fmt: off
DSA_COMPAT_TABLE: Tuple[DSACell, ...] = (
    # tilelang: ROCm default; on CUDA bf16 only (#31346)
    _supported("tilelang", PLATFORM_HIP, KV_BF16, PHASE_PREFILL, priority=100, evidence="ROCm resolver default"),
    _supported("tilelang", PLATFORM_HIP, KV_BF16, PHASE_DECODE, priority=100, evidence="ROCm resolver default"),
    _supported("tilelang", PLATFORM_HIP, KV_FP8, PHASE_PREFILL, priority=100, evidence="ROCm fp8 sparse-MLA kernels, see #31346"),
    _supported("tilelang", PLATFORM_HIP, KV_FP8, PHASE_DECODE, priority=100, evidence="ROCm fp8 sparse-MLA kernels, see #31346"),
    _supported("tilelang", PLATFORM_SM90, KV_BF16, PHASE_PREFILL, evidence="bf16 CUDA path verified on H100, see #31346"),
    _supported("tilelang", PLATFORM_SM90, KV_BF16, PHASE_DECODE, evidence="bf16 CUDA path verified on H100, see #31346"),
    _unsupported("tilelang", PLATFORM_SM90, KV_FP8, PHASE_PREFILL, reason=_TILELANG_FP8_CUDA_REASON, evidence="#31346"),
    _unsupported("tilelang", PLATFORM_SM90, KV_FP8, PHASE_DECODE, reason=_TILELANG_FP8_CUDA_REASON, evidence="#31346"),
    _unsupported("tilelang", PLATFORM_SM100, KV_FP8, PHASE_PREFILL, reason=_TILELANG_FP8_CUDA_REASON, evidence="#31346"),
    _unsupported("tilelang", PLATFORM_SM100, KV_FP8, PHASE_DECODE, reason=_TILELANG_FP8_CUDA_REASON, evidence="#31346"),
    # flashmla_sparse: bf16 prefill default on CUDA
    _supported("flashmla_sparse", PLATFORM_SM90, KV_BF16, PHASE_PREFILL, priority=100, evidence="resolver default"),
    _supported("flashmla_sparse", PLATFORM_SM100, KV_BF16, PHASE_PREFILL, priority=100, evidence="resolver default"),
    # fa3: bf16 decode default below SM100
    _supported("fa3", PLATFORM_SM90, KV_BF16, PHASE_DECODE, priority=100, evidence="resolver default"),
    # trtllm: SM100 defaults
    _supported("trtllm", PLATFORM_SM100, KV_BF16, PHASE_DECODE, priority=100, evidence="resolver default"),
    _supported("trtllm", PLATFORM_SM100, KV_FP8, PHASE_PREFILL, priority=100, evidence="resolver default"),
    _supported("trtllm", PLATFORM_SM100, KV_FP8, PHASE_DECODE, priority=100, evidence="resolver default"),
    # flashmla_kv: fp8 default below SM100
    _supported("flashmla_kv", PLATFORM_SM90, KV_FP8, PHASE_PREFILL, priority=100, evidence="resolver default"),
    _supported("flashmla_kv", PLATFORM_SM90, KV_FP8, PHASE_DECODE, priority=100, evidence="resolver default"),
    # flashmla_sparse_q8: mirrors the dsa_backend.py construction checks
    _supported("flashmla_sparse_q8", PLATFORM_SM90, KV_FP8, PHASE_PREFILL, evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM90, KV_BF16, PHASE_PREFILL, reason="flashmla_sparse_q8 is native FP8 and requires an fp8_e4m3 KV cache; use flashmla_sparse for the bf16 path", evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM100, KV_BF16, PHASE_PREFILL, reason="flashmla_sparse_q8 is SM90-only", evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM100, KV_FP8, PHASE_PREFILL, reason="flashmla_sparse_q8 is SM90-only", evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM90, KV_BF16, PHASE_DECODE, reason=_Q8_DECODE_REASON, evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM90, KV_FP8, PHASE_DECODE, reason=_Q8_DECODE_REASON, evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM100, KV_BF16, PHASE_DECODE, reason=_Q8_DECODE_REASON, evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_SM100, KV_FP8, PHASE_DECODE, reason=_Q8_DECODE_REASON, evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_HIP, KV_BF16, PHASE_DECODE, reason=_Q8_DECODE_REASON, evidence="dsa_backend.py construction checks"),
    _unsupported("flashmla_sparse_q8", PLATFORM_HIP, KV_FP8, PHASE_DECODE, reason=_Q8_DECODE_REASON, evidence="dsa_backend.py construction checks"),
    # q8 requires fp8 on every platform; q8+HIP+fp8 prefill stays UNKNOWN
    # because dsa_backend.py's sm-major-9 check passes on gfx9 ROCm.
    _unsupported("flashmla_sparse_q8", PLATFORM_HIP, KV_BF16, PHASE_PREFILL, reason="flashmla_sparse_q8 is native FP8 and requires an fp8_e4m3 KV cache; use flashmla_sparse for the bf16 path", evidence="dsa_backend.py construction checks"),
    # aiter: ROCm decode path
    _supported("aiter", PLATFORM_HIP, KV_BF16, PHASE_DECODE, evidence="aiter mla_decode_fwd path in dsa_backend.py"),
    _supported("aiter", PLATFORM_HIP, KV_FP8, PHASE_DECODE, evidence="aiter fp8 metadata path in dsa_backend.py"),
    # flashmla_auto: prefill-only; decode rejection lands separately in #31344
    _supported("flashmla_auto", PLATFORM_SM90, KV_BF16, PHASE_PREFILL, evidence="auto-select in dsa_backend.py"),
    _supported("flashmla_auto", PLATFORM_SM90, KV_FP8, PHASE_PREFILL, evidence="auto-select in dsa_backend.py"),
    _supported("flashmla_auto", PLATFORM_SM100, KV_BF16, PHASE_PREFILL, evidence="auto-select in dsa_backend.py"),
    _supported("flashmla_auto", PLATFORM_SM100, KV_FP8, PHASE_PREFILL, evidence="auto-select in dsa_backend.py"),
)
# fmt: on

_CellKey = Tuple[str, str, str, str]


def _build_index() -> Dict[_CellKey, DSACell]:
    index: Dict[_CellKey, DSACell] = {}
    for cell in DSA_COMPAT_TABLE:
        key = (cell.backend, cell.platform, cell.kv_dtype, cell.phase)
        if key in index:
            raise AssertionError(f"duplicate DSA compat cell: {key}")
        index[key] = cell
    return index


_CELL_INDEX: Dict[_CellKey, DSACell] = _build_index()


def platform_bucket(sm_major: int, *, hip: bool) -> str:
    if hip:
        return PLATFORM_HIP
    return PLATFORM_SM100 if sm_major >= 10 else PLATFORM_SM90


def kv_dtype_bucket(kv_cache_dtype: str) -> str:
    return KV_FP8 if kv_cache_dtype == KV_FP8 else KV_BF16


def lookup_cell(*, backend: str, platform: str, kv_dtype: str, phase: str) -> DSACell:
    """Declared cell, or an implicit UNKNOWN."""
    cell = _CELL_INDEX.get((backend, platform, kv_dtype, phase))
    if cell is not None:
        return cell
    return DSACell(
        backend=backend,
        platform=platform,
        kv_dtype=kv_dtype,
        phase=phase,
        status=STATUS_UNKNOWN,
    )


def supported_backends(*, platform: str, kv_dtype: str, phase: str) -> List[str]:
    return [
        cell.backend
        for cell in DSA_COMPAT_TABLE
        if cell.platform == platform
        and cell.kv_dtype == kv_dtype
        and cell.phase == phase
        and cell.status == STATUS_SUPPORTED
    ]


def _default_backend(*, platform: str, kv_dtype: str, phase: str) -> Optional[str]:
    candidates = [
        cell
        for cell in DSA_COMPAT_TABLE
        if cell.platform == platform
        and cell.kv_dtype == kv_dtype
        and cell.phase == phase
        and cell.status == STATUS_SUPPORTED
        and cell.default_priority is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda cell: cell.default_priority).backend


def resolve_dsa_default_backends(
    *,
    sm_major: int,
    hip: bool,
    kv_cache_dtype: str,
    user_set_prefill: bool,
    user_set_decode: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Table-driven defaults, bit-identical to the historical branches.
    Quirk preserved: HIP tilelang default needs both sides unset; a
    partially-set HIP config falls through to the CUDA-bucket defaults."""
    kv_dtype = kv_dtype_bucket(kv_cache_dtype)
    if hip and not user_set_prefill and not user_set_decode:
        platform = PLATFORM_HIP
    else:
        platform = platform_bucket(sm_major, hip=False)
    prefill = None
    decode = None
    if not user_set_prefill:
        prefill = _default_backend(
            platform=platform, kv_dtype=kv_dtype, phase=PHASE_PREFILL
        )
    if not user_set_decode:
        decode = _default_backend(
            platform=platform, kv_dtype=kv_dtype, phase=PHASE_DECODE
        )
    return prefill, decode


def check_dsa_backend_compat(
    *,
    kv_cache_dtype: str,
    prefill_backend: Optional[str],
    decode_backend: Optional[str],
    sm_major: int,
    hip: bool,
) -> None:
    """Reject combinations declared unsupported; unknown cells pass."""
    platform = platform_bucket(sm_major, hip=hip)
    kv_dtype = kv_dtype_bucket(kv_cache_dtype)
    for phase, backend in (
        (PHASE_PREFILL, prefill_backend),
        (PHASE_DECODE, decode_backend),
    ):
        if backend is None:
            continue
        cell = lookup_cell(
            backend=backend, platform=platform, kv_dtype=kv_dtype, phase=phase
        )
        if cell.status != STATUS_UNSUPPORTED:
            continue
        alternatives = supported_backends(
            platform=platform, kv_dtype=kv_dtype, phase=phase
        )
        # dtypes under which THIS backend is supported here (the #31346 hint:
        # keep the backend, switch the KV dtype)
        keep_backend_dtypes = [
            c.kv_dtype
            for c in DSA_COMPAT_TABLE
            if c.backend == backend
            and c.platform == platform
            and c.phase == phase
            and c.status == STATUS_SUPPORTED
        ]
        switch_hint = (
            f" Or keep --dsa-{phase}-backend {backend} and set "
            f"--kv-cache-dtype {keep_backend_dtypes[0]}."
            if keep_backend_dtypes
            else ""
        )
        raise ValueError(
            f"--dsa-{phase}-backend {backend} is not supported with "
            f"--kv-cache-dtype {kv_cache_dtype} on {platform}: {cell.reason} "
            f"(see {cell.evidence}). Supported {phase} backends here: "
            f"{', '.join(alternatives) or 'none declared'}.{switch_hint}"
        )
