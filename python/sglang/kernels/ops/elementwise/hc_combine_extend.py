"""Extend-size HyperConnection boundary: combine + next norm + next gate.

One kernel per HC boundary for prefill-size row counts, bit-identical to the
stock extend chain ``hc_combine`` -> ``grouped_gemma_rmsnorm`` -> (next)
``hc_combine`` gate dot. The CUDA source lives in ``hc_combine_extend_src``.

Gate partials here are the next read's raw gate dot totals, ``[rows, hc]``
fp32 (a different format from the decode ``[rows, 8, hc]`` partials; the two
never meet because they are produced and consumed at disjoint row counts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _source_path() -> str:
    """Write the kernel source to a content-addressed file once; return it."""
    import hashlib
    import os

    from sglang.kernels.jit.utils.compile.cache import cache_root
    from sglang.kernels.ops.elementwise.hc_combine_extend_src import CUDA_SOURCE

    data = CUDA_SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = cache_root() / "inline-src"
    path = directory / f"hc_combine_extend-{digest}.cuh"
    if not path.is_file() or path.read_bytes() != data:
        directory.mkdir(parents=True, exist_ok=True)
        staged = directory / f".{path.name}.{os.getpid()}.tmp"
        staged.write_bytes(data)
        os.replace(staged, path)
    return str(path)


# (has_partials, do_norm, next_gate) -> (wrapper name, min CTAs/SM, prefetch
# norm weight). Tuned at 1601..16384 rows: 64 registers
# (4 CTAs/SM) with the norm weight read at use is fastest for every flavour.
_VARIANTS = {
    (True, True, True): ("pnn", 4, False),
    (False, True, True): ("snn", 4, False),
    (True, True, False): ("pn", 4, False),
    (False, True, False): ("sn", 4, False),
    (True, False, False): ("p", 4, False),
    (False, False, False): ("s", 4, False),
}


def _b(x: bool) -> str:
    return "true" if x else "false"


@cache_once
def _jit_module(hc_count: int, hidden_size: int, dtype: torch.dtype) -> Module:
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}")
    if hidden_size % 512 != 0 or (hc_count * hidden_size) % 2048 != 0:
        raise RuntimeError(f"Unsupported HC shape {hc_count}x{hidden_size}")
    if hc_count > 8:
        raise RuntimeError(f"Unsupported hc_count {hc_count}")
    args = make_cpp_args(hc_count, hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "hc_combine_extend",
        *args,
        cuda_files=[_source_path()],
        cuda_wrappers=[
            (
                name,
                f"HcCombineExtendKernel<{args}, {_b(p)}, {_b(n)}, {_b(g)}, "
                f"{min_blocks}, {_b(prefetch)}>::run",
            )
            for (p, n, g), (name, min_blocks, prefetch) in _VARIANTS.items()
        ],
    )


def warmup(hc_count: int, hidden_size: int, dtype: torch.dtype) -> None:
    """Compile (or load from the JIT cache) ahead of the first extend."""
    _jit_module(hc_count, hidden_size, dtype)


def hc_combine_extend(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    *,
    partials: Optional[torch.Tensor] = None,
    normed_residual: Optional[torch.Tensor] = None,
    inject_weight: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    next_inject_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(combined, normalized, next_partials)``.

    The current gate comes from ``partials`` ([rows, hc] fp32 dot totals made
    by the previous boundary) or, without them, from ``normed_residual`` and
    ``inject_weight`` exactly as ``hc_combine``. With ``norm_weight`` the
    combined streams are normalized per branch as ``grouped_gemma_rmsnorm``
    (group = hidden_size); with ``next_inject_weight`` too, the next read's
    gate dot totals are returned.
    """
    has_partials = partials is not None
    do_norm = norm_weight is not None
    next_gate = do_norm and next_inject_weight is not None
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    rows = r.shape[0]
    out = torch.empty_like(r)
    normed = torch.empty_like(r) if do_norm else r
    next_partials = (
        torch.empty((rows, hc_count), dtype=torch.float32, device=r.device)
        if next_gate
        else None
    )
    if has_partials:
        n = r
        w = r
        p = partials
    else:
        n = normed_residual.reshape(-1, hc_count * hidden_size)
        w = inject_weight
        p = r
    module = _jit_module(hc_count, hidden_size, r.dtype)
    getattr(module, _VARIANTS[(has_partials, do_norm, next_gate)][0])(
        y,
        r,
        n,
        w,
        p,
        norm_weight if do_norm else r,
        next_inject_weight if next_gate else r,
        out,
        normed,
        next_partials if next_gate else r,
        float(eps),
    )
    return (
        out.reshape(residual.shape),
        normed.reshape(residual.shape) if do_norm else None,
        next_partials,
    )
