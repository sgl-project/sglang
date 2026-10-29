"""Separate HC gate preparation and combine, with optional next-branch norm.

The decode specialization accepts four branches of 2560 BF16 values. Gate
partials belong to each invocation, so overlapping sublayers never reuse them.
"""

import os
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

_HC_CLUSTER_M4ONLY = envs.SGLANG_Q38C_HC_CLUSTER_M4ONLY.get()
_HC_DISABLE_CLUSTER = os.environ.get("SGLANG_Q38C_HC_DISABLE_CLUSTER", "0") == "1"
_HC_PARTS_OVERRIDE = int(os.environ.get("SGLANG_Q38C_HC_PARTS", "0"))
_HC_P2_M16_GATE = os.environ.get("SGLANG_Q38C_HC_P2_M16_GATE", "0") == "1"
_HC_P2_M4_ONLY = os.environ.get("SGLANG_Q38C_HC_P2_M4_ONLY", "0") == "1"


@cache_once
def _jit_module(parts: int = 2, use_override: bool = True, force_cluster: bool = False):
    if use_override and _HC_PARTS_OVERRIDE:
        parts = _HC_PARTS_OVERRIDE
    if parts not in (1, 2, 4):
        raise ValueError("HC combine parts must be 1, 2 or 4")
    cluster_m4only = _HC_CLUSTER_M4ONLY and not force_cluster
    m4only = str(cluster_m4only).lower()
    disable_cluster = use_override and _HC_DISABLE_CLUSTER and not force_cluster
    no_cluster = str(disable_cluster).lower()
    return load_jit(
        f"hc_combine_decode_p{parts}_m4only{int(cluster_m4only)}_nocluster{int(disable_cluster)}",
        cuda_files=["elementwise/hc_combine_decode.cuh"],
        cuda_wrappers=[
            ("gate", "HcCombineDecode::gate"),
            ("apply", "HcCombineDecode::apply"),
            ("apply_norm", f"HcCombineNormDecode<{parts}, {m4only}, {no_cluster}>::run"),
            ("shared_apply_norm", f"HcSharedCombineNormDecode<{parts}, {m4only}, {no_cluster}>::run"),
        ],
    )


def _module_for_rows(parts: int, rows: int):
    if _HC_P2_M16_GATE:
        # M32 verify explicitly uses the existing two-CTA cluster kernel.
        return _jit_module(2, rows <= 16, force_cluster=rows > 16)
    return _jit_module(parts, not _HC_P2_M4_ONLY or rows == 4)


def hc_combine_gate(normed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partials = torch.empty(
        (normed.shape[0], 8, 4), device=normed.device, dtype=torch.float32
    )
    _jit_module().gate(normed, weight, partials)
    return partials


def hc_combine_apply(
    block: torch.Tensor, residual: torch.Tensor, partials: torch.Tensor
) -> torch.Tensor:
    combined = torch.empty_like(residual)
    _jit_module().apply(block, residual, partials, combined)
    return combined


def hc_combine_apply_norm(
    block: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    parts: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _module_for_rows(parts, block.shape[0]).apply_norm(
        block, residual, partials, weight, combined, normalized, eps
    )
    return combined, normalized


def hc_shared_combine_apply_norm(hidden, shared, routed, residual, partials,
                                 shared_weight, norm_weight, eps, parts=2):
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _module_for_rows(parts, hidden.shape[0]).shared_apply_norm(
        hidden, shared, routed, residual, partials, shared_weight,
        norm_weight, combined, normalized, eps
    )
    return combined, normalized



# Rows the fused combine + next norm + next gate kernel accepts.
HC_COMBINE_NORM_GATE_MAX_ROWS = 32


def _norm_gate_source_path() -> str:
    """Write the kernel source to a content-addressed file once; return it."""
    import hashlib
    import os

    from sglang.kernels.jit.utils.compile.cache import cache_root
    from sglang.kernels.ops.elementwise.hc_combine_norm_gate_src import CUDA_SOURCE

    data = CUDA_SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = cache_root() / "inline-src"
    path = directory / f"hc_combine_norm_gate-{digest}.cuh"
    if not path.is_file() or path.read_bytes() != data:
        directory.mkdir(parents=True, exist_ok=True)
        staged = directory / f".{path.name}.{os.getpid()}.tmp"
        staged.write_bytes(data)
        os.replace(staged, path)
    return str(path)


@cache_once
def _jit_norm_gate_module():
    return load_jit(
        "hc_combine_norm_gate",
        cuda_files=[_norm_gate_source_path()],
        cuda_wrappers=[
            ("gate", "HcCombineNormGateDecode::gate"),
            ("norm", "HcCombineNormGateDecode::norm"),
        ],
    )


def hc_combine_norm_gate(
    block: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    inject_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """One HC boundary for 1..32 rows of 4x2560 BF16 streams.

    Applies the gate from ``partials`` (as ``hc_combine_apply``), normalizes
    the BF16 combined streams with the next read's per-branch norm (as
    ``grouped_gemma_rmsnorm``) and, given the next read's ``inject_weight``,
    also returns that read's gate partials ([rows, 8, 4] fp32, a different
    split of the same dot than ``hc_combine_gate``; only their sum over the
    8 splits is meaningful). Returns ``(combined, normalized, next_partials)``;
    ``next_partials`` is None without ``inject_weight``.
    """
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    module = _jit_norm_gate_module()
    if inject_weight is None:
        module.norm(block, residual, partials, norm_weight, combined, normalized, eps)
        return combined, normalized, None
    next_partials = torch.empty(
        (residual.shape[0], 8, 4), device=residual.device, dtype=torch.float32
    )
    module.gate(
        block,
        residual,
        partials,
        norm_weight,
        inject_weight,
        combined,
        normalized,
        next_partials,
        eps,
    )
    return combined, normalized, next_partials
