"""One-launch HyperConnection boundary for decode verify rows, K-sliced (TP1, hc 4, hidden 2560, rank 320).

A write-back (combine + the next read's per-branch norm and gate dots) and the next read's mix run as one launch
of 40 CTAs per 64-row tile (hc_boundary_ks_src). Each CTA combines a 256-column slice of its rows and runs the
mix down of that slice on tcgen05; the inverse RMS, a per-(row, branch) scalar, is applied after the K sum, so no
CTA waits for the norm before its GEMM (the reassociation upstream #40552 and #46881 use for this boundary). The
40 partials of a row tile are reduced in a fixed order through an L2 scratch buffer, then each CTA runs the mix up
and gated mean for 64 hidden columns. The combined streams are hc_combine_extend's bits; the normalized streams,
next gate totals and mixed input are not bitwise to the separate kernels (rounding points and K order differ).
Every CTA must be resident at once: rows <= 320 (200 CTAs), checked against the device at launch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, is_arch_support_pdl, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

HC, HS, LOWRANK = 4, 2560, 320
MIN_ROWS = 33
MAX_ROWS = 320
_ROW_TILE = 64
_SLICES = 40
_MAX_ROW_TILES = MAX_ROWS // _ROW_TILE
_P_BLOCK = _ROW_TILE * (LOWRANK // _SLICES)
_CTR_SIZE = 32
DBG_SLOTS = 16
# Kernel flags: 1 drops the consumed scratch lines from L2 (no DRAM write-back); 2 runs the gate sigmoid as
# 0.5 tanh(x / 2) + 0.5 (one MUFU op) in place of Triton's ex2 + div. 4 is the caller's: always write the
# normalized streams (GatedResidual otherwise skips them where only the gate totals are read).
FLAG_DISCARD = 1
FLAG_TANH = 2
FLAG_WRITE_NORMED = 4
FLAGS_DEFAULT = FLAG_DISCARD | FLAG_TANH
_logged = False


def _source_path() -> str:
    import hashlib
    import os

    from sglang.kernels.jit.utils.compile.cache import cache_root
    from sglang.kernels.ops.gemm.hc_boundary_ks_src import CUDA_SOURCE

    data = CUDA_SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = cache_root() / "inline-src"
    path = directory / f"hc_boundary_ks-{digest}.cuh"
    if not path.is_file() or path.read_bytes() != data:
        directory.mkdir(parents=True, exist_ok=True)
        staged = directory / f".{path.name}.{os.getpid()}.tmp"
        staged.write_bytes(data)
        os.replace(staged, path)
    return str(path)


def _b(x: bool) -> str:
    return "true" if x else "false"


def _name(next_gate: bool, write_normed: bool) -> str:
    return f"run_{'g' if next_gate else 'x'}{'n' if write_normed else 'x'}"


@cache_once
def _jit_module() -> Module:
    pdl = is_arch_support_pdl()
    wrappers = [
        (_name(g, n), f"HcBoundaryKsKernel<{_b(pdl)}, {_b(g)}, {_b(n)}>::run")
        for g in (True, False)
        for n in (True, False)
    ]
    wrappers.append(("capacity", f"HcBoundaryKsKernel<{_b(pdl)}, true, true>::capacity"))
    return load_jit(
        "hc_boundary_ks",
        str(pdl),
        cuda_files=[_source_path()],
        cuda_wrappers=wrappers,
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


_workspaces: dict = {}


def _workspace(device: torch.device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _workspaces.get(key)
    if ws is None:
        dev = torch.device("cuda", key)
        ws = (
            torch.empty(_MAX_ROW_TILES * _SLICES * _SLICES * _P_BLOCK, dtype=torch.float32, device=dev),
            torch.empty(_MAX_ROW_TILES * _SLICES * _ROW_TILE, dtype=torch.float32, device=dev),
            torch.empty(_MAX_ROW_TILES * _SLICES * _ROW_TILE * HC, dtype=torch.float32, device=dev),
            torch.zeros((MAX_ROWS, LOWRANK), dtype=torch.bfloat16, device=dev),
            torch.zeros(_CTR_SIZE, dtype=torch.int32, device=dev),
        )
        _workspaces[key] = ws
    return ws


def warmup(device: Optional[torch.device] = None) -> None:
    """Compile (or load from the JIT cache) and allocate the scratch before graph capture."""
    _jit_module()
    _workspace(device if device is not None else torch.device("cuda", torch.cuda.current_device()))


def grid_for(rows: int) -> int:
    return -(-rows // _ROW_TILE) * _SLICES


@cache_once
def capacity(device_index: int) -> int:
    probe = torch.empty(1, device=f"cuda:{device_index}")
    return int(_jit_module().capacity(probe))


def supported(rows: int, device_index: int) -> bool:
    return MIN_ROWS <= rows <= MAX_ROWS and grid_for(rows) <= capacity(device_index)


def hc_boundary_ks(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    *,
    partials: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    next_inject_weight: Optional[torch.Tensor],
    w_down: torch.Tensor,
    w_up_permuted: torch.Tensor,
    write_normed: bool = True,
    flags: Optional[int] = None,
    trace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Return ``(combined, normalized, next_partials, mixed)`` for MIN_ROWS..MAX_ROWS rows.

    ``partials`` are this read's [rows, 4] fp32 gate dot totals (the extend boundary's); ``next_partials`` is
    None without ``next_inject_weight`` and ``normalized`` None without ``write_normed``. ``w_up_permuted`` is
    hc_mix_splitk.permute_pad_up_weight's layout (rows j*4 + g, the first 320 columns used).
    """
    global _logged
    rows = residual.shape[0]
    device = residual.device
    dev_index = device.index if device.index is not None else torch.cuda.current_device()
    if not supported(rows, dev_index):
        raise ValueError(f"hc_boundary_ks serves {MIN_ROWS}..{MAX_ROWS} rows within the resident grid, got {rows}")
    next_gate = next_inject_weight is not None
    ws_p, ws_ss, ws_q, ws_t, ctr = _workspace(device)
    out = torch.empty_like(residual)
    normed = torch.empty_like(residual) if write_normed else None
    mixed = torch.empty((rows, HS), dtype=residual.dtype, device=device)
    next_partials = torch.empty((rows, HC), dtype=torch.float32, device=device) if next_gate else None
    if not _logged:
        _logged = True
        logger.info(
            "HC boundary KS on: rows %d..%d in one launch (K-sliced combine + tcgen05 down, fixed-order L2 "
            "reduction, tcgen05 up/gate); first rows=%d grid=%d next_gate=%s normed=%s",
            MIN_ROWS, MAX_ROWS, rows, grid_for(rows), next_gate, write_normed,
        )
    getattr(_jit_module(), _name(next_gate, write_normed))(
        block_output,
        residual,
        partials,
        norm_weight,
        next_inject_weight if next_gate else w_down,
        w_down,
        w_up_permuted,
        out,
        normed if write_normed else out,
        next_partials if next_gate else partials,
        mixed,
        ws_p,
        ws_ss,
        ws_q,
        ws_t,
        ctr,
        trace if trace is not None else ctr,
        int(trace is not None),
        FLAGS_DEFAULT if flags is None else int(flags) & (FLAG_DISCARD | FLAG_TANH),
        float(eps),
    )
    return out, normed, next_partials, mixed
