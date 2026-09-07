# SPDX-License-Identifier: Apache-2.0
"""Precomputed AdaLN plan cache for the MiniMax H3 DiT."""

from __future__ import annotations

import os
import struct
from collections import OrderedDict
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Callable

import msgspec
import torch
import torch.nn as nn
from safetensors.torch import safe_open

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tp_rank,
    get_world_group,
    world_group_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HOST_RESERVE_FRACTION,
    MIN_HOST_RESERVE_BYTES,
    host_memory_available_bytes,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32

# The native adaln_proj tensor names the online rebuild streams; a checkpoint
# without them (Diffusers layout, quantized export) cannot serve as a rebuild
# source.
_NATIVE_ADALN_PROBE_KEY = "blocks.0.adaln_proj.linear.weight"


# A ref2va request carrying both a visual and an audio reference reaches four
# distinct timesteps in one step: video, audio, the imgvid condition and the
# audio reference. That is the widest case, so it is the default; a deployment
# serving only narrower tasks (t2va reaches 2, fl2va 3) can shrink the slab
# proportionally via --minimax-h3-adaln-plan-width.
MINIMAX_H3_ADALN_MAX_PLAN_WIDTH = 4


def _plan_key(timesteps: torch.Tensor) -> tuple[int, ...]:
    """One denoise step's unique timesteps as their exact fp32 bit patterns."""
    return tuple(
        struct.unpack("<I", struct.pack("<f", float(value)))[0]
        for value in timesteps.tolist()
    )


def native_adaln_weight_files(weights_path: str) -> list[str]:
    """Safetensors under ``weights_path`` usable as an online rebuild source.

    Returns [] when the directory holds no native-layout adaln tensors, so a
    caller retargeting the rebuild source can fail closed instead of deferring
    a KeyError to the next request.
    """
    import glob as _glob

    files = sorted(_glob.glob(os.path.join(weights_path, "*.safetensors")))
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            if _NATIVE_ADALN_PROBE_KEY in handle.keys():
                return files
    return []


@contextmanager
def _fp32_gemm_guard(*, enabled: bool, device: torch.device):
    """Neighboring stages flip the process-global TF32 switches; without this
    an 'fp32' build silently becomes tf32-once."""
    if not enabled or device.type != "cuda":
        yield
        return
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision(matmul_precision)


class MiniMaxH3AdalnCacheStats(msgspec.Struct):
    gpu_hit_plans: int = 0
    host_hit_plans: int = 0
    built_plans: int = 0
    host_evicted_groups: int = 0
    host_pressure_skips: int = 0


class _HostPlan(msgspec.Struct):
    pages: list[int]
    refcount: int = 0


class MiniMaxH3AdalnHostTier:
    """Bounded pinned-host store of built AdaLN plans, group-LRU evicted.

    One page holds one (plan, timestep) entry: the 50-layer block params and
    the final-layer params packed back to back (~9.25 MiB at full size). A
    plan spans ``length`` pages that need not be contiguous; H2D swap-in is
    two copies per page. Groups (one request's full plan set) are the LRU and
    refcount unit because a rebuild pass streams every adaln_proj layer no
    matter how few plans it misses -- a partial host hit saves nothing, so
    admission and eviction are all-or-nothing per group.

    Rank determinism: capacity is MIN-reduced across ranks at construction and
    every state transition happens synchronously inside prepare, as a pure
    function of the (replica-broadcast) request stream. Any timing-based
    state flip here would desynchronize build()'s collectives across ranks.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        block_width: int,
        final_width: int,
        capacity_bytes: int,
        device: torch.device,
    ) -> None:
        self.block_numel = num_layers * block_width
        self.num_layers = num_layers
        self.block_width = block_width
        self.page_numel = self.block_numel + final_width
        page_bytes = self.page_numel * 2
        budget = _host_cache_budget_bytes()
        num_pages = max(0, min(capacity_bytes, budget)) // page_bytes
        self._slab, self.pinned, num_pages = _allocate_host_slab(
            num_pages=num_pages, page_numel=self.page_numel
        )
        num_pages = _all_ranks_min(num_pages, device=device)
        self.num_pages = num_pages
        self._free_pages = list(range(num_pages))
        self._plans: dict[tuple[int, ...], _HostPlan] = {}
        self._groups: OrderedDict[tuple[tuple[int, ...], ...], tuple] = OrderedDict()
        self._stream = (
            torch.cuda.Stream(device=device)
            if device.type == "cuda" and self.pinned
            else None
        )
        self.stats: MiniMaxH3AdalnCacheStats | None = None
        logger.info(
            "MiniMax H3 AdaLN host tier: %d pages x %.2f MiB (%s)",
            num_pages,
            page_bytes / 2**20,
            "pinned" if self.pinned else "pageable",
        )

    def fence_prepare_start(self) -> None:
        if self._stream is None:
            return
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        current.wait_stream(self._stream)

    def fence_gpu_reads(self) -> None:
        if self._stream is None:
            return
        torch.cuda.current_stream().wait_stream(self._stream)

    def synchronize(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    def has_all(self, keys) -> bool:
        return all(key in self._plans for key in keys)

    def register_group(self, *, group_key, keys) -> None:
        """Record (or refresh) a group whose plans are all resident already."""
        if group_key in self._groups:
            self._groups.move_to_end(group_key)
            return
        keys = tuple(keys)
        if not self.has_all(keys):
            # Some plans were never stored (host_pressure skip); nothing to pin.
            return
        for key in keys:
            self._plans[key].refcount += 1
        self._groups[group_key] = keys

    def copy_to_gpu(
        self,
        assignments: dict[tuple[int, ...], int],
        *,
        block_params: torch.Tensor,
        final_params: torch.Tensor,
    ) -> None:
        """H2D-copy each assigned plan's pages into its reserved slab slot.

        Runs on the tier's copy stream; the caller fences the compute stream
        afterwards (fence_gpu_reads) and only then flips plan_lengths.
        """
        with self._stream_ctx():
            for key, slot in assignments.items():
                for index, page in enumerate(self._plans[key].pages):
                    row = self._slab[page]
                    block_params[slot, index].copy_(
                        row[: self.block_numel].view(self.num_layers, self.block_width),
                        non_blocking=True,
                    )
                    final_params[slot, index].copy_(
                        row[self.block_numel :], non_blocking=True
                    )

    def store_group(
        self,
        *,
        group_key,
        entries: dict[tuple[int, ...], tuple[int, int]],
        block_params: torch.Tensor,
        final_params: torch.Tensor,
    ) -> None:
        """D2H-copy a freshly built group into host pages and commit it.

        ``entries`` maps each plan key to its (GPU slot, timestep count).
        Commits synchronously (the copies are waited on before the group
        becomes visible) so the resident set stays a deterministic function of
        the request stream on every rank. Over capacity the group is simply
        not cached; the next occurrence recomputes.
        """
        if group_key in self._groups:
            self._groups.move_to_end(group_key)
            return
        new_plans = {
            key: value for key, value in entries.items() if key not in self._plans
        }
        needed = sum(length for _, length in new_plans.values())
        if needed > self.num_pages:
            self._skip_for_pressure(group_key)
            return
        # Pin the shared plans first so evicting other groups cannot free them
        # (and then double-count them as new).
        for key in entries:
            if key in self._plans:
                self._plans[key].refcount += 1
        while needed > len(self._free_pages) and self._groups:
            self._evict_oldest_group()
        if needed > len(self._free_pages):
            for key in entries:
                if key in self._plans:
                    self._unpin(key)
            self._skip_for_pressure(group_key)
            return

        if self._stream is not None:
            # Order the D2H reads after the build's slab writes.
            self._stream.wait_stream(torch.cuda.current_stream())
        staged: dict[tuple[int, ...], _HostPlan] = {}
        with self._stream_ctx():
            for key, (slot, length) in new_plans.items():
                pages = [self._free_pages.pop() for _ in range(length)]
                for index, page in enumerate(pages):
                    row = self._slab[page]
                    row[: self.block_numel].view(
                        self.num_layers, self.block_width
                    ).copy_(block_params[slot, index], non_blocking=True)
                    row[self.block_numel :].copy_(
                        final_params[slot, index], non_blocking=True
                    )
                staged[key] = _HostPlan(pages=pages, refcount=1)
        self.synchronize()
        self._plans.update(staged)
        self._groups[group_key] = tuple(entries)

    def clear(self) -> None:
        self._plans.clear()
        self._groups.clear()
        self._free_pages = list(range(self.num_pages))

    def _stream_ctx(self):
        if self._stream is None:
            return nullcontext()
        return torch.cuda.stream(self._stream)

    def _unpin(self, key) -> None:
        plan = self._plans[key]
        plan.refcount -= 1
        if plan.refcount == 0:
            self._free_pages.extend(plan.pages)
            del self._plans[key]

    def _evict_oldest_group(self) -> None:
        _, keys = self._groups.popitem(last=False)
        for key in keys:
            self._unpin(key)
        if self.stats is not None:
            self.stats.host_evicted_groups += 1

    def _skip_for_pressure(self, group_key) -> None:
        if self.stats is not None:
            self.stats.host_pressure_skips += 1
        logger.info(
            "MiniMax H3 AdaLN host tier: group of %d plan(s) skipped under "
            "memory pressure; it will recompute on its next occurrence",
            len(group_key),
        )


def _host_cache_budget_bytes() -> int:
    """Per-rank share of the host memory actually available right now.

    Co-located ranks each see the same free memory; without the split every
    rank would size its tier against all of it (the HiCache lesson), and the
    reserve mirrors HostPinBudget so this pinner honors the same headroom as
    every other one. Assumes single-node deployments when LOCAL_WORLD_SIZE is
    unset.
    """
    ranks = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE",
            (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            ),
        )
    )
    available = host_memory_available_bytes()
    reserve = max(int(available * HOST_RESERVE_FRACTION), MIN_HOST_RESERVE_BYTES)
    return max(0, available - reserve) // max(1, ranks)


def _allocate_host_slab(
    *, num_pages: int, page_numel: int
) -> tuple[torch.Tensor, bool, int]:
    """Pinned slab, halving the page count on failure; pageable as last resort."""
    pages = num_pages
    while pages > 0:
        try:
            slab = torch.empty((pages, page_numel), dtype=_BF16_DTYPE, pin_memory=True)
            if pages < num_pages:
                logger.warning(
                    "MiniMax H3 AdaLN host tier shrank to %d of %d pages "
                    "(pinned allocation failures)",
                    pages,
                    num_pages,
                )
            return slab, True, pages
        except RuntimeError:
            pages //= 2
    if num_pages > 0:
        logger.warning(
            "MiniMax H3 AdaLN host tier could not pin any memory; falling "
            "back to pageable host memory (H2D copies run synchronously)"
        )
        return torch.empty((num_pages, page_numel), dtype=_BF16_DTYPE), False, num_pages
    return torch.empty((0, page_numel), dtype=_BF16_DTYPE), False, 0


def _all_ranks_min(value: int, *, device: torch.device) -> int:
    """Every rank must run the same tier capacity or build()'s collectives
    desynchronize; MIN over the world group is the conservative agreement."""
    if not torch.distributed.is_initialized() or not world_group_is_initialized():
        return value
    if torch.distributed.get_world_size() == 1:
        return value
    probe = torch.tensor(
        [value],
        dtype=torch.int64,
        device=device if device.type == "cuda" else "cpu",
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)
    return int(probe.item())


class MiniMaxH3AdalnCache(nn.Module):
    """Precomputed AdaLN outputs for fixed FP32 timestep plans."""

    _FORMAT_VERSION = "2"
    plan_timesteps: torch.Tensor
    plan_lengths: torch.Tensor
    block_params: torch.Tensor
    final_params: torch.Tensor

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        path: str | None = None,
        model_variant: str | None = None,
        weight_files: list[str] | None = None,
        max_plans: int = 64,
        max_plan_width: int = MINIMAX_H3_ADALN_MAX_PLAN_WIDTH,
        host_cache_bytes: int = 0,
        precision: str = "match",
    ) -> None:
        super().__init__()
        if (path is None) == (weight_files is None):
            raise ValueError(
                "MiniMax H3 AdaLN cache takes exactly one of path (prebuilt "
                "sidecar) or weight_files (rebuild from the checkpoint)"
            )
        if max_plans < 1:
            raise ValueError("MiniMax H3 AdaLN cache max_plans must be positive")
        if max_plan_width < 1:
            raise ValueError(
                "MiniMax H3 AdaLN cache max_plan_width must be positive; "
                "set --minimax-h3-adaln-plan-width to at least 1"
            )
        if precision not in ("match", "fp32"):
            raise ValueError(
                "MiniMax H3 AdaLN cache precision must be 'match' (bit-exact "
                f"with resident adaln_proj weights) or 'fp32', got {precision!r}"
            )
        self.path = path
        self.model_variant = model_variant
        self.weight_files = weight_files
        self.max_plans = max_plans
        self.max_plan_width = max_plan_width
        self.num_layers = arch.num_layers
        self.hidden_size = arch.hidden_size
        self.block_width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * arch.hidden_size
        self.final_width = 2 * arch.hidden_size
        # Plan bit pattern -> slot, tracked on the host in LRU order (oldest
        # first). The rebuild path evicts per plan; a sidecar never evicts.
        self._slots: OrderedDict[tuple[int, ...], int] = OrderedDict()
        self._free_slots: list[int] = list(range(max_plans))
        self.host_cache_bytes = host_cache_bytes
        self.precision = precision
        # Constructed in load(): needs the device and the distributed runtime.
        self._host_tier: MiniMaxH3AdalnHostTier | None = None
        self.stats = MiniMaxH3AdalnCacheStats()
        self.rebuilds = 0

    def load(self, device: torch.device) -> None:
        if self.path is None:
            self._allocate(device)
            if self.host_cache_bytes > 0:
                self._host_tier = MiniMaxH3AdalnHostTier(
                    num_layers=self.num_layers,
                    block_width=self.block_width,
                    final_width=self.final_width,
                    capacity_bytes=self.host_cache_bytes,
                    device=device,
                )
                self._host_tier.stats = self.stats
            return
        if not os.path.isfile(self.path):
            raise ValueError(f"MiniMax H3 AdaLN cache does not exist: {self.path}")

        with safe_open(self.path, framework="pt", device="cpu") as cache_file:
            metadata = cache_file.metadata() or {}
            if metadata.get("format_version") != self._FORMAT_VERSION:
                raise ValueError(
                    "MiniMax H3 AdaLN cache has an unsupported or missing format_version"
                )
            cache_variant = metadata.get("model_variant")
            if self.model_variant is not None and cache_variant != self.model_variant:
                raise ValueError(
                    "MiniMax H3 AdaLN cache model_variant does not match the loaded "
                    f"variant ({cache_variant!r} != {self.model_variant!r})"
                )
            plan_timesteps = cache_file.get_tensor("plan_timesteps")
            plan_lengths = cache_file.get_tensor("plan_lengths")
            block_params = cache_file.get_tensor("block_params")
            final_params = cache_file.get_tensor("final_params")

        if (
            plan_timesteps.dtype != _FP32_DTYPE
            or plan_timesteps.ndim != 2
            or plan_lengths.dtype != torch.int64
            or plan_lengths.shape != (plan_timesteps.shape[0],)
            or (plan_lengths < 1).any()
            or (plan_lengths > plan_timesteps.shape[1]).any()
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid timestep plans")
        if block_params.dtype != _BF16_DTYPE or block_params.shape != (
            plan_timesteps.shape[0],
            plan_timesteps.shape[1],
            self.num_layers,
            self.block_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid block_params")
        if final_params.dtype != _BF16_DTYPE or final_params.shape != (
            plan_timesteps.shape[0],
            plan_timesteps.shape[1],
            self.final_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid final_params")

        for slot in range(plan_timesteps.shape[0]):
            length = int(plan_lengths[slot])
            self._slots[_plan_key(plan_timesteps[slot, :length])] = slot

        # The 0.9-2.3 GiB slabs are derived data; keep them out of state_dict.
        self.register_buffer(
            "plan_timesteps", plan_timesteps.to(device), persistent=False
        )
        self.register_buffer("plan_lengths", plan_lengths.to(device), persistent=False)
        self.register_buffer("block_params", block_params.to(device), persistent=False)
        self.register_buffer("final_params", final_params.to(device), persistent=False)

    def _allocate(self, device: torch.device) -> None:
        """Empty slab for the rebuild path; its pointers must never move.

        ``plan_lengths`` starts at zero and that is what keeps unused slots out
        of ``lookup``: a real plan always has at least one timestep, so a zero
        length can never match. Breakable CUDA graph keys its replay signature
        on tensor pointers, so this is allocated once and only written in place.
        """
        width = self.max_plan_width
        self.register_buffer(
            "plan_timesteps",
            torch.zeros((self.max_plans, width), dtype=_FP32_DTYPE, device=device),
            persistent=False,
        )
        self.register_buffer(
            "plan_lengths",
            torch.zeros((self.max_plans,), dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "block_params",
            torch.zeros(
                (self.max_plans, width, self.num_layers, self.block_width),
                dtype=_BF16_DTYPE,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "final_params",
            torch.zeros(
                (self.max_plans, width, self.final_width),
                dtype=_BF16_DTYPE,
                device=device,
            ),
            persistent=False,
        )
        logger.info(
            "MiniMax H3 AdaLN rebuild slab: %d plans x %d timesteps = %.2f GiB",
            self.max_plans,
            width,
            self.block_params.numel() * 2 / 2**30,
        )

    def build(
        self,
        step_timesteps: list[torch.Tensor],
        *,
        embed: Callable[[torch.Tensor], torch.Tensor],
        keys: list[tuple[int, ...]] | None = None,
    ) -> None:
        """Fill every plan this request will look up, in one streaming pass.

        Each plan keeps its own timestep count as the GEMM batch size, because
        cuBLAS selects kernels by shape and the selection is not monotonic in M:
        against the runtime's M == 2, results at M == 4/8/16/64/96 are
        bit-identical while M == 32 differs in 11760 of 96768 elements and
        M == 1 (the GEMV path the first denoise step takes) differs in 69.
        Rebuilding a plan at any other batch size silently perturbs the output.

        The pass reads all 50 adaln_proj layers regardless of how many plans are
        missing, so a request builds everything it needs before denoising rather
        than filling in step by step.
        """
        if keys is None:
            keys = [_plan_key(timesteps) for timesteps in step_timesteps]
        wanted: dict[tuple[int, ...], torch.Tensor] = {}
        for key, timesteps in zip(keys, step_timesteps):
            wanted.setdefault(key, timesteps)
        missing = {k: v for k, v in wanted.items() if k not in self._slots}
        group_key = tuple(wanted)
        if not missing:
            # A pure hit is still a use: without the touch, a hot schedule
            # keeps its build-time LRU stamp and gets evicted first.
            for key in wanted:
                self._slots.move_to_end(key)
            self.stats.gpu_hit_plans += len(wanted)
            if self._host_tier is not None:
                self._host_tier.register_group(group_key=group_key, keys=wanted)
            return
        if len(wanted) > self.max_plans:
            raise ValueError(
                f"MiniMax H3 AdaLN rebuild needs {len(wanted)} plans but the "
                "slab holds "
                f"{self.max_plans}; raise SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS"
            )
        widest = max(timesteps.numel() for timesteps in wanted.values())
        if widest > self.max_plan_width:
            raise ValueError(
                f"MiniMax H3 AdaLN rebuild hit a {widest}-timestep plan but the "
                f"slab was allocated for {self.max_plan_width}; raise "
                "--minimax-h3-adaln-plan-width (t2va needs 2, fl2va 3, ref2va 4)"
            )
        self.stats.gpu_hit_plans += len(wanted) - len(missing)

        if self._host_tier is not None:
            # Order the tier's copy stream against everything the compute
            # stream still has queued (a failed request never drains it) and
            # vice versa, once per request, before any slot is touched.
            self._host_tier.fence_prepare_start()
            if self._host_tier.has_all(missing):
                self._swap_in_from_host(wanted=wanted, missing=missing)
                self._host_tier.register_group(group_key=group_key, keys=wanted)
                return

        device = self.block_params.device
        pending_slots = self._allocate_slots(wanted=wanted, missing=missing)
        slots = []
        for key, timesteps in missing.items():
            slot = pending_slots[key]
            adaln_input = embed(timesteps.to(device))
            if self.precision == "fp32":
                # Compute the projections in fp32 once; the slab stays bf16.
                adaln_input = adaln_input.float()
            slots.append((slot, timesteps.numel(), adaln_input))
            self.plan_timesteps[slot, : timesteps.numel()] = timesteps.to(device)

        # adaln_proj is a ColumnParallelLinear: each rank owns a slice of the
        # output features and all-gathers afterwards. The rebuild has to do the
        # same rather than read the full width in one go -- a sharded GEMM has a
        # different N, so cuBLAS picks a different kernel and the outputs stop
        # matching. It also cuts per-rank checkpoint reads to 1/tp.
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank() if tp_size > 1 else 0

        try:
            with _fp32_gemm_guard(enabled=self.precision == "fp32", device=device):
                self._project_plans_from_checkpoint(
                    slots, tp_size=tp_size, tp_rank=tp_rank, device=device
                )
        except BaseException:
            # The pending slots never became visible (their lengths stayed
            # zero); hand them back so a retry does not leak capacity.
            self._free_slots.extend(pending_slots.values())
            raise

        for slot, length, _ in slots:
            self.plan_lengths[slot] = length
        # Commit host metadata only after every layer has been written. If a
        # checkpoint read or projection raises, the zero-length slots remain
        # invisible and a later request can retry the rebuild.
        self._slots.update(pending_slots)
        self.rebuilds += 1
        self.stats.built_plans += len(missing)
        logger.info(
            "MiniMax H3 AdaLN: rebuilt %d plan(s), %d/%d resident, pass #%d",
            len(missing),
            len(self._slots),
            self.max_plans,
            self.rebuilds,
        )
        if self._host_tier is not None:
            self._host_tier.store_group(
                group_key=group_key,
                entries={
                    key: (self._slots[key], wanted[key].numel()) for key in wanted
                },
                block_params=self.block_params,
                final_params=self.final_params,
            )

    def _allocate_slots(
        self,
        *,
        wanted: dict[tuple[int, ...], torch.Tensor],
        missing: dict[tuple[int, ...], torch.Tensor],
    ) -> dict[tuple[int, ...], int]:
        """Reserve one slab slot per missing plan, evicting LRU plans if needed.

        Reserved slots stay invisible (length 0) until the caller commits them
        into ``self._slots``.
        """
        for key in wanted:
            if key in self._slots:
                self._slots.move_to_end(key)
        shortfall = len(missing) - len(self._free_slots)
        if shortfall > 0:
            # The wanted-count check in build() guarantees enough resident
            # plans outside this request to cover the shortfall, in LRU order.
            victims = [key for key in self._slots if key not in wanted][:shortfall]
            for key in victims:
                slot = self._slots.pop(key)
                # Zero the length first: the slot must be invisible to lookup
                # and resolve_slots before its contents are overwritten.
                self.plan_lengths[slot] = 0
                self._free_slots.append(slot)
        return {key: self._free_slots.pop() for key in missing}

    def _swap_in_from_host(
        self,
        *,
        wanted: dict[tuple[int, ...], torch.Tensor],
        missing: dict[tuple[int, ...], torch.Tensor],
    ) -> None:
        """Fill the missing GPU slots from pinned host pages, no weight pass."""
        assert self._host_tier is not None
        device = self.block_params.device
        assignments = self._allocate_slots(wanted=wanted, missing=missing)
        try:
            for key, timesteps in missing.items():
                slot = assignments[key]
                self.plan_timesteps[slot, : timesteps.numel()] = timesteps.to(device)
            self._host_tier.copy_to_gpu(
                assignments,
                block_params=self.block_params,
                final_params=self.final_params,
            )
        except BaseException:
            # Lengths never flipped, so the slots stayed invisible; hand them
            # back or the free-list invariant breaks and a later admission-
            # accepted request pops from an empty list.
            self._free_slots.extend(assignments.values())
            raise
        # The compute stream must observe the H2D copies before any forward
        # reads the slots; lengths flip last so half-filled slots stay hidden.
        self._host_tier.fence_gpu_reads()
        for key, timesteps in missing.items():
            self.plan_lengths[assignments[key]] = timesteps.numel()
        self._slots.update(assignments)
        self.stats.host_hit_plans += len(missing)
        logger.info(
            "MiniMax H3 AdaLN: %d plan(s) from the host cache, %d/%d resident",
            len(missing),
            len(self._slots),
            self.max_plans,
        )

    def invalidate(self) -> None:
        """Drop every cached plan; call between requests after a weight swap."""
        if self._host_tier is not None:
            self._host_tier.synchronize()
            self._host_tier.clear()
        self._slots.clear()
        self._free_slots = list(range(self.max_plans))
        self.plan_lengths.zero_()

    def _project_plans_from_checkpoint(
        self,
        slots: list[tuple[int, int, torch.Tensor]],
        *,
        tp_size: int,
        tp_rank: int,
        device: torch.device,
    ) -> None:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(safe_open(f, framework="pt", device=str(device)))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor

            def project(adaln_input: torch.Tensor, weight, bias) -> torch.Tensor:
                out = nn.functional.linear(adaln_input, weight, bias)
                return tensor_model_parallel_all_gather(out) if tp_size > 1 else out

            for layer in range(self.num_layers):
                prefix = f"blocks.{layer}.adaln_proj.linear"
                weight = read_shard(f"{prefix}.weight", self.block_width)
                bias = read_shard(f"{prefix}.bias", self.block_width)
                for slot, length, adaln_input in slots:
                    self.block_params[slot, :length, layer] = project(
                        adaln_input, weight, bias
                    )
                del weight, bias
            prefix = "final_layer.adaln_proj.linear"
            weight = read_shard(f"{prefix}.weight", self.final_width)
            bias = read_shard(f"{prefix}.bias", self.final_width)
            for slot, length, adaln_input in slots:
                self.final_params[slot, :length] = project(adaln_input, weight, bias)
            del weight, bias

    def resolve_slots(
        self,
        step_timesteps: list[torch.Tensor],
        *,
        keys: list[tuple[int, ...]] | None = None,
    ) -> torch.Tensor:
        """Per-step slab slots as one device tensor, resolved on the host.

        Forward receives one scalar view per step. Keeping it a device tensor
        matters twice over: a Python int would enter the breakable-CUDA-graph
        replay signature (one graph per slot value), and an int baked into a
        captured gather would read the wrong slab row after slot reuse.
        """
        if keys is None:
            keys = [_plan_key(timesteps) for timesteps in step_timesteps]
        slots = []
        for key in keys:
            slot = self._slots.get(key)
            if slot is None:
                raise ValueError(
                    "MiniMax H3 AdaLN cache does not cover the request timestep plan"
                )
            slots.append(slot)
        return torch.tensor(slots, dtype=torch.int64, device=self.block_params.device)

    def lookup(self, unique_timesteps: torch.Tensor) -> torch.Tensor:
        num_timesteps = unique_timesteps.shape[0]
        if num_timesteps > self.plan_timesteps.shape[1]:
            # Without this the slice below silently clamps and the comparison
            # dies on a shape mismatch instead of the real reason.
            raise ValueError(
                "MiniMax H3 AdaLN cache does not cover the request timestep plan"
            )
        matches = self.plan_lengths.eq(num_timesteps) & self.plan_timesteps[
            :, :num_timesteps
        ].eq(unique_timesteps).all(dim=-1)
        if not bool(matches.any()):
            raise ValueError(
                "MiniMax H3 AdaLN cache does not cover the request timestep plan"
            )
        return matches.to(torch.int64).argmax()

    def block_all(
        self,
        *,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every block's AdaLN tuple via one layer-major slab gather."""
        stacked = self.block_params.permute(2, 0, 1, 3)[
            :, cache_plan_index, :num_timesteps
        ]
        stacked = stacked.reshape(self.num_layers, -1, 6, self.hidden_size)
        return tuple(tuple(layer.unbind(dim=1)) for layer in stacked)

    def final(
        self,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, ...]:
        params = self.final_params[cache_plan_index, :num_timesteps]
        return tuple(params.reshape(-1, 2, self.hidden_size).unbind(dim=1))
