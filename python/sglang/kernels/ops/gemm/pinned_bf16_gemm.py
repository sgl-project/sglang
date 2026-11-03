"""BF16 ``x @ w.T`` through pinned GEMM kernels that are bitwise equal to torch.mm.

At decode-sized M, cuBLAS's heuristic can pick a kernel whose grid leaves most SMs
idle. When a GEMM runs without split-K, every output element is one fp32
accumulator over K in ascending order followed by one rounding to BF16, so other
split-K-free cuBLASLt algorithms of the same build often give the same bits while
using the GPU better. ``benchmark/kernels/pinned_bf16_gemm/tune_pinned_bf16_gemm.py``
searches them per (N, K, M) and writes the fastest one that matches torch.mm bit for
bit into a JSON table for one GPU model and cuBLASLt version.

``SGLANG_OPT_BF16_GEMM_TABLE=<table.json>`` enables the table. On the first eager
call, a table recorded for another GPU or cuBLASLt version is ignored, and every
remaining entry is re-checked bitwise against torch.mm in this process; an entry
that differs is dropped with a warning. Shapes outside the table return ``None``
and the caller keeps its own call.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.toolchain import cuda_home
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_ALGO_WORDS = 8
# One row of `enumerate_algos`: the algo words, workspace bytes, split-K count.
_ROW_WORDS = _ALGO_WORDS + 2
_SELF_CHECK_SEEDS = (1, 2, 3)

# A table entry as a callable (x, w, out) -> out; None until the table is loaded.
_Runner = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
_table: Optional[Dict[Tuple[int, int, int], _Runner]] = None
_workspaces: Dict[int, torch.Tensor] = {}


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "cublaslt_pinned_gemm",
        cuda_files=["gemm/cublaslt_pinned_gemm.cuh"],
        cuda_wrappers=[
            ("run", "sglang::CublasLtPinnedGemm::run"),
            ("version", "sglang::CublasLtPinnedGemm::version"),
            ("enumerate", "sglang::CublasLtPinnedGemm::enumerate"),
        ],
        # Resolves to the libcublasLt torch has already loaded (same soname).
        extra_ldflags=[f"-L{os.path.join(cuda_home(), 'lib64')}", "-lcublasLt"],
    )


def cublaslt_version() -> int:
    return int(_jit_module().version())


def table_device_name() -> str:
    return torch.cuda.get_device_name().replace(" ", "_")


def enumerate_algos(m: int, n: int, k: int, max_workspace_bytes: int) -> torch.Tensor:
    """Candidate cuBLASLt algorithms for the packed (M, N, K) problem as int64 rows
    of (8 algo words, workspace bytes, split-K count); for the tuner and tests."""
    rows = torch.empty(4096, _ROW_WORDS, dtype=torch.int64)
    found = int(_jit_module().enumerate(m, n, k, max_workspace_bytes, rows))
    if found > rows.shape[0]:
        rows = torch.empty(found, _ROW_WORDS, dtype=torch.int64)
        found = int(_jit_module().enumerate(m, n, k, max_workspace_bytes, rows))
    return rows[:found]


def _workspace(device: torch.device, nbytes: int) -> torch.Tensor:
    ws = _workspaces.get(device.index)
    if ws is None or ws.numel() < nbytes:
        ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _workspaces[device.index] = ws
    return ws[:nbytes]


def run_algo(
    x: torch.Tensor,
    w: torch.Tensor,
    algo: torch.Tensor,
    workspace_bytes: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ w.T`` with one explicit cuBLASLt algorithm (``algo``: CPU int64 [8])."""
    if out is None:
        out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
    _jit_module().run(x, w, out, algo, _workspace(x.device, workspace_bytes))
    return out


def _runner_for(entry: dict) -> Optional[_Runner]:
    algo = torch.tensor(entry.get("cublaslt_algo", []), dtype=torch.int64)
    if algo.numel() != _ALGO_WORDS:
        return None
    ws = int(entry.get("workspace_bytes", 0))
    return lambda x, w, out: run_algo(x, w, algo, ws, out=out)


def _matches_torch(w: torch.Tensor, m: int, runner: _Runner) -> bool:
    for seed in _SELF_CHECK_SEEDS:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
        x = torch.randn(
            m, w.shape[1], device=w.device, dtype=torch.bfloat16, generator=g
        )
        ref = torch.mm(x, w.t())
        got = runner(x, w, None)
        if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
            return False
    return True


def _load_table(path: str) -> Dict[Tuple[int, int, int], _Runner]:
    with open(path) as f:
        data = json.load(f)
    device_name, version = table_device_name(), cublaslt_version()
    if (
        data.get("device_name") != device_name
        or data.get("cublaslt_version") != version
    ):
        logger.warning(
            "BF16 GEMM table %s was recorded for %s / cuBLASLt %s, this process "
            "runs %s / cuBLASLt %s; the table is ignored",
            path,
            data.get("device_name"),
            data.get("cublaslt_version"),
            device_name,
            version,
        )
        return {}
    # cuBLAS picks kernels by shape, layout, alignment and dtype, never by values,
    # so one random weight per (N, K) checks every layer of that shape. Every entry
    # is checked now because CUDA graphs capture batch sizes that never ran
    # eagerly, and capture cannot synchronize to check.
    device = torch.device("cuda", torch.cuda.current_device())
    table = {}
    w = None
    for entry in sorted(data["entries"], key=lambda e: (e["n"], e["k"], e["m"])):
        n, k, m = int(entry["n"]), int(entry["k"]), int(entry["m"])
        runner = _runner_for(entry)
        if w is None or w.shape != (n, k):
            w = None
            g = torch.Generator(device=device)
            g.manual_seed(0)
            w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g)
        if runner is not None and _matches_torch(w, m, runner):
            table[(n, k, m)] = runner
        else:
            logger.warning(
                "BF16 GEMM table entry N=%d K=%d M=%d does not match torch.mm; "
                "keeping torch.mm for it",
                n,
                k,
                m,
            )
    del w
    torch.cuda.empty_cache()
    logger.info(
        "BF16 GEMM table: %d of %d entries active", len(table), len(data["entries"])
    )
    return table


def pinned_mm(
    x: torch.Tensor, w: torch.Tensor, out: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """``x @ w.T`` with torch.mm's bits through the table's kernel, or ``None``
    when the table has no entry for this (N, K, M) or the operands do not qualify."""
    global _table
    if _table is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        _table = _load_table(envs.SGLANG_OPT_BF16_GEMM_TABLE.get())
    if not _table or x.dim() != 2 or w.dim() != 2:
        return None
    runner = _table.get((w.shape[0], w.shape[1], x.shape[0]))
    if runner is None:
        return None
    # The entries were checked on packed, 16-byte-aligned operands; other
    # layouts can make cuBLAS choose differently.
    if not (
        x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 16 == 0
        and w.data_ptr() % 16 == 0
    ):
        return None
    if out is not None and not (
        out.dtype == torch.bfloat16 and out.is_contiguous() and out.data_ptr() % 16 == 0
    ):
        return None
    return runner(x, w, out)
