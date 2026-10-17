"""Opt-in Hopper LiteTopK backend for single-request ragged prefill.

Set ``SGLANG_USE_SM90_LITETOPK=1`` to enable it. Unsupported calls return
``None`` so the caller can use the regular dense logits path.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_HEAD = 12288
_TOPK = 2048
_MAX_KEYS = 1 << 20
_MAX_CARRY_ROWS = 1536
_CARRY_STATE_INTS = 135
_ENV = "SGLANG_USE_SM90_LITETOPK"

CandidateSelector = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _deep_gemm_include() -> Path:
    configured = os.getenv("SM90_LITETOPK_DEEP_GEMM_INCLUDE")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    else:
        candidates = []
        spec = importlib.util.find_spec("deep_gemm")
        if spec is not None and spec.origin is not None:
            candidates.append(Path(spec.origin).resolve().parent / "include")
        candidates.append(
            Path(__file__).resolve().parents[3] / "third_party/deep_gemm/include"
        )

    for include in candidates:
        required = (
            include / "deep_gemm/common/utils.cuh",
            include / "cutlass/cutlass.h",
        )
        if all(path.is_file() for path in required):
            return include
    raise RuntimeError(
        "DeepGEMM/CUTLASS headers were not found; set "
        "SM90_LITETOPK_DEEP_GEMM_INCLUDE to their include directory"
    )


def _source_digest(paths: tuple[Path, ...], flags: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(repr(flags).encode())
    return digest.hexdigest()[:12]


@functools.cache
def _load_extensions() -> tuple[Any, Any]:
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / "csrc"
    include = _deep_gemm_include()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    flags = (
        "-DLT_BQ=4",
        "-DLT_BK=256",
        "-DLT_STAGES=2",
        "-DLT_MATH=512",
        "-DLT_REGS=112",
        "-DLT_UNROLL=1",
        "-DLT_PRODUCER_REGS=32",
        "-DLT_UNIFORM_WARP=1",
        "-DLT_COMBINED_EMITTER=1",
        "-DLT_OVERLAP_EMIT=0",
        "-DLT_DOUBLE_MMA=0",
        "-DLT_WARP_BUFFER=2",
        "-DLT_MERGE_REDUCE=2",
        "-DLT_INTERIOR=1",
        "-DLT_ONLINE=1",
        "-DLT_REFRESH=0",
        "-DLT_PENDING=128",
        "-DLT_GLOBAL_GATE=1",
    )
    common_cuda_flags = (
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_90a,code=sm_90a",
    )
    verbose = os.getenv("SM90_LITETOPK_BUILD_VERBOSE", "0") == "1"

    core_sources = (
        source_dir / "litetopk_sm90.cu",
        source_dir / "sm90_litetopk.cuh",
    )
    core = load(
        name=f"sm90_litetopk_core_{_source_digest(core_sources, flags)}",
        sources=[str(core_sources[0])],
        extra_include_paths=[str(source_dir), str(include)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[*common_cuda_flags, *flags],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )

    aux_sources = (source_dir / "online_aux.cu", source_dir / "pr_hot.cuh")
    aux = load(
        name=f"sm90_litetopk_aux_{_source_digest(aux_sources, ())}",
        sources=[str(aux_sources[0])],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=list(common_cuda_flags),
        verbose=verbose,
    )
    return core, aux


@dataclass
class _Carry:
    indices: torch.Tensor
    extent: int


class _Workspace:
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        capacity: int,
        core: Any,
        aux: Any,
    ) -> None:
        rows, keys = q.shape[0], k.shape[0]
        device = q.device
        self.shape = (rows, keys)
        self.capacity = capacity
        self.core = core
        self.aux = aux

        self.k = torch.empty_like(k)
        self.scales = torch.empty(keys, dtype=torch.float32, device=device)
        self.starts = torch.zeros(rows, dtype=torch.int32, device=device)
        self.seed_ends = torch.full_like(self.starts, _HEAD)
        self.threshold = torch.empty(rows, dtype=torch.float32, device=device)
        self.values = torch.empty((rows, capacity), dtype=torch.float32, device=device)
        self.indices = torch.empty((rows, capacity), dtype=torch.int32, device=device)
        self.counts = torch.empty(rows, dtype=torch.int32, device=device)
        self.selected = torch.empty((rows, _TOPK), dtype=torch.int32, device=device)
        self.output = torch.empty_like(self.selected)
        self.seed_values = torch.empty(
            (rows, _HEAD), dtype=torch.float32, device=device
        )
        self.seed_hist = torch.empty((rows, 256), dtype=torch.int32, device=device)
        self.origin = torch.empty(rows, dtype=torch.float32, device=device)
        self.inv_delta = torch.empty_like(self.origin)
        self.buckets = torch.empty(rows, dtype=torch.int32, device=device)

        self.permutation = torch.arange(keys, dtype=torch.int32, device=device)
        self.epochs = torch.zeros(keys, dtype=torch.int32, device=device)
        self.swap_a = torch.empty(_HEAD, dtype=torch.int32, device=device)
        self.swap_b = torch.empty_like(self.swap_a)
        self.plan_counts = torch.zeros(4, dtype=torch.int32, device=device)

        self.votes = torch.zeros(keys, dtype=torch.int32, device=device)
        max_vote = min(rows, _MAX_CARRY_ROWS)
        self.partial = torch.empty(
            ((keys + 8191) // 8192, max_vote + 1),
            dtype=torch.int16,
            device=device,
        )
        self.carry_state = torch.zeros(
            _CARRY_STATE_INTS, dtype=torch.int32, device=device
        )
        self.next_carry = torch.empty(_HEAD, dtype=torch.int64, device=device)

    def run(
        self,
        q: torch.Tensor,
        source_k: torch.Tensor,
        source_scales: torch.Tensor,
        weights: torch.Tensor,
        ends: torch.Tensor,
        carry: torch.Tensor,
        common_end: int,
        select: CandidateSelector,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        splits = 1 if source_k.shape[0] <= 131072 else 8
        self.aux.prepare(
            carry,
            self.epochs,
            self.permutation,
            self.swap_a,
            self.swap_b,
            self.plan_counts,
            common_end,
            source_k,
            source_scales,
            self.k,
            self.scales,
        )
        self.core.score_dense(
            q,
            self.k[:_HEAD],
            self.scales[:_HEAD],
            weights,
            self.starts,
            self.seed_ends,
            self.threshold,
            self.seed_values,
            self.indices,
            self.counts,
            1,
        )
        self.aux.seed(
            self.seed_values,
            self.origin,
            self.inv_delta,
            self.buckets,
            self.threshold,
            self.values,
            self.indices,
            self.counts,
            self.seed_hist,
        )
        self.core.score_hot(
            q,
            self.k,
            self.scales,
            weights,
            self.starts,
            ends,
            self.threshold,
            self.values,
            self.indices,
            self.counts,
            splits,
            self.origin,
            self.inv_delta,
            self.seed_hist,
        )

        counts = self.counts.cpu()
        if bool(((counts < _TOPK) | (counts > self.capacity)).any()):
            return None, None, int(ends.max().item())

        selected = select(self.values, self.counts)
        if (
            selected.shape != self.selected.shape
            or selected.dtype != torch.int32
            or not selected.is_cuda
            or selected.device != self.selected.device
        ):
            raise RuntimeError(
                "SM90 LiteTopK candidate selector must return CUDA int32 "
                f"{self.selected.shape}"
            )
        self.selected.copy_(selected)
        self.core.map_indices(self.selected, self.indices, self.counts, self.output)
        self.aux.map_vote(self.output, self.permutation, self.votes)

        last_end = int(ends.max().item())
        self.aux.carry(
            self.votes[:last_end],
            self.next_carry,
            self.partial,
            self.carry_state,
            _HEAD,
            min(q.shape[0], _MAX_CARRY_ROWS),
            0,
        )
        return self.output, self.next_carry, last_end


class SM90LiteTopK:
    """Stateful, fail-closed adapter around the validated HOT12288 kernels."""

    def __init__(self) -> None:
        self.enabled = os.getenv(_ENV, "0") == "1"
        self._build_failed = False
        self._modules: tuple[Any, Any] | None = None
        self._carry: dict[Hashable, _Carry] = {}
        self._workspaces: dict[int, _Workspace] = {}
        capacity = (
            int(os.getenv("SM90_LITETOPK_CAPACITY", "49152")) if self.enabled else 49152
        )
        if capacity < _HEAD or capacity % 256:
            raise ValueError(
                "SM90_LITETOPK_CAPACITY must be >= 12288 and divisible by 256"
            )
        self.capacity = capacity

    def _get_modules(self) -> tuple[Any, Any] | None:
        if not self.enabled or self._build_failed:
            return None
        if self._modules is None:
            try:
                self._modules = _load_extensions()
            except Exception:
                self._build_failed = True
                logger.warning(
                    "SM90 LiteTopK JIT build failed; using the dense indexer",
                    exc_info=True,
                )
        return self._modules

    def _can_run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> bool:
        rows, keys = q.shape[0], k.shape[0]
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if (
            not self.enabled
            or not q.is_cuda
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability(q.device) != (9, 0)
            or q.dtype != fp8
            or k.dtype != fp8
            or rows < 1
            or q.shape != (rows, 32, 128)
            or k.shape != (keys, 128)
            or not (_HEAD <= keys <= _MAX_KEYS)
            or keys % 4
            or scales.shape != (keys,)
            or scales.dtype != torch.float32
            or weights.shape != (rows, 32)
            or weights.dtype != torch.float32
            or starts.shape != (rows,)
            or ends.shape != (rows,)
            or starts.dtype != torch.int32
            or ends.dtype != torch.int32
        ):
            return False
        tensors = (q, k, scales, weights, starts, ends)
        return all(
            tensor.is_contiguous() and tensor.device == q.device for tensor in tensors
        )

    def try_run(
        self,
        state_key: Hashable,
        q: torch.Tensor,
        k: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        select: CandidateSelector,
    ) -> torch.Tensor | None:
        if not self._can_run(q, k, scales, weights, starts, ends):
            return None
        if bool((starts != 0).any()):
            return None
        common_end = int(ends.min().item())
        if common_end < _HEAD or int(ends.max().item()) > k.shape[0]:
            return None

        state = self._carry.get(state_key)
        if state is None or state.extent > common_end:
            return None
        modules = self._get_modules()
        if modules is None:
            return None

        device_index = q.device.index
        assert device_index is not None
        workspace = self._workspaces.get(device_index)
        if (
            workspace is None
            or workspace.shape != (q.shape[0], k.shape[0])
            or workspace.capacity != self.capacity
        ):
            workspace = _Workspace(q, k, self.capacity, *modules)
            self._workspaces[device_index] = workspace

        with torch.cuda.device(q.device):
            output, next_carry, extent = workspace.run(
                q,
                k,
                scales,
                weights,
                ends,
                state.indices,
                common_end,
                select,
            )
        if output is None:
            return None
        assert next_carry is not None
        self._carry[state_key] = _Carry(next_carry.clone(), extent)
        return output

    def observe(
        self, state_key: Hashable, topk_indices: torch.Tensor, extent: int
    ) -> None:
        if (
            not self.enabled
            or extent < _HEAD
            or extent > _MAX_KEYS
            or not topk_indices.is_cuda
            or topk_indices.dtype != torch.int32
            or topk_indices.ndim != 2
            or topk_indices.shape[1] != _TOPK
        ):
            self._carry.pop(state_key, None)
            return

        rows = min(topk_indices.shape[0], _MAX_CARRY_ROWS)
        recent = topk_indices[-rows:].reshape(-1).to(torch.int64)
        valid = recent[(recent >= 0) & (recent < extent)]
        votes = torch.zeros(extent, dtype=torch.int32, device=topk_indices.device)
        votes.scatter_add_(
            0,
            valid,
            torch.ones(valid.shape, dtype=torch.int32, device=valid.device),
        )
        carry = torch.topk(votes, _HEAD, sorted=False).indices.contiguous()
        self._carry[state_key] = _Carry(carry, extent)


_INSTANCE = SM90LiteTopK()


def get_sm90_litetopk() -> SM90LiteTopK:
    return _INSTANCE
