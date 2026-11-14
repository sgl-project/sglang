"""The disk tier: where a layer's experts live when the host budget cannot hold them all.

When the host budget (``--paged-experts-host-memory-gb``) cannot hold every expert of every
MoE layer and ``--paged-experts-enable-disk`` allows it, each layer keeps all E experts in its
expert cache file (``expert_cache``) and only R of them in pinned host rows
(``TieredExpertStore``); the K GPU slots are filled from the rows, which keep their experts. The
file is built (and checked) as for a layer that fits, and written once: a repack replays K
experts at a time from the mapped checkpoint and writes its output straight into the file
being built; a format that pages the loaded rows loads them straight into it.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import math
import os
import platform
import shutil
import threading
import time
from types import SimpleNamespace
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.layers.moe.paged_experts import expert_cache as _expert_cache
from sglang.srt.layers.moe.paged_experts.expert_cache import (
    ExpertCache,
    ExpertReader,
    ObjectLayout,
    _directory,
    _outcomes,
    _prune_other_keys,
    check_groups,
    checked_groups,
    read_many,
)
from sglang.srt.layers.moe.paged_experts.formats import build_store
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore, RowsLike

_GIB = 1 << 30
# The plan kernel's bound on K (``tiered_plan`` in paged_experts.cuh).
_MAX_TIER_SLOTS = 1024
# How long a captured step waits at a layer for its disk reads before it runs on and is
# replayed (a safety net: a read takes milliseconds).
WAIT_TIMEOUT_MS = 2000
logger = logging.getLogger(__name__)


class ModelContext:
    """The disk tier's state for the model being built and served: the host budget (measured
    once, before the first layer allocates), whether the disk space was checked, and the
    device state of its tiered layers."""

    def __init__(self):
        self.budget_measured = False
        # None: every expert fits (pinned, as without the tier).
        self.budget_bytes: Optional[int] = None
        # When every expert fits: the memory that was compared with the first layer's
        # estimate, and what the layers pinned so far (layers of other formats can be larger).
        self.fits_bytes: Optional[int] = None
        self.pinned_bytes = 0
        self.token_routed = 0  # leading layers that keep every expert in host memory
        self.space_checked = False
        self.device_state = None


_MODEL = ModelContext()


def current_model() -> ModelContext:
    return _MODEL


def host_memory_budget_bytes(need_bytes: int) -> Optional[int]:
    """Pinned host memory for all expert stores, which need ``need_bytes`` to hold every
    expert: ``--paged-experts-host-memory-gb``; unset, None (every expert pinned, as without
    the disk tier) while they fit what is available now within the container's memory limit,
    else that less the reserve HiCache also leaves."""
    from sglang.srt.mem_cache.pool_host.base import (
        HICACHE_HOST_MEMORY_RESERVE_BYTES,
        host_memory_budget_plain_hugetlb,
    )
    from sglang.srt.mem_cache.storage.mmap.mmap_allocator import HUGEPAGE_MODE_OFF
    from sglang.srt.runtime_context import get_exec

    gb = get_exec().moe.paged_experts_host_memory_gb
    if gb is not None:
        return int(gb * _GIB)
    try:
        # Plain RAM: pinned host rows are not hugetlb pages.
        budget, _ = host_memory_budget_plain_hugetlb(
            auto_size=True, page_size_and_mode=(0, HUGEPAGE_MODE_OFF)
        )
    except (RuntimeError, ValueError, KeyError) as e:
        raise RuntimeError(
            "Paged experts: cannot read the container's memory limit to size the host "
            "budget for the disk tier; set --paged-experts-host-memory-gb"
        ) from e
    if need_bytes <= budget + HICACHE_HOST_MEMORY_RESERVE_BYTES:
        current_model().fits_bytes = budget + HICACHE_HOST_MEMORY_RESERVE_BYTES
        return None
    return max(0, budget)


def get_exec_moe():
    from sglang.srt.runtime_context import get_exec

    return get_exec().moe


def token_routed_layers() -> int:
    """How many leading layers route by token id rather than by hidden state (DeepSeek-V4's
    hash-routed layers, ``num_hash_layers``)."""
    from sglang.srt.runtime_context import process_model_config

    config = getattr(process_model_config(), "hf_text_config", None)
    return int(getattr(config, "num_hash_layers", 0) or 0)


def host_rows(
    layout: RowsLike,
    num_experts: int,
    num_slots: int,
    enabled: bool,
    layer_id: Optional[int] = None,
) -> int:
    """Experts per layer the host budget holds in pinned rows: E when all fit (no disk tier).
    Called only with ``--paged-experts-enable-disk`` or ``--paged-experts-host-memory-gb``:
    without them every expert is pinned. The budget is measured once, before the first layer
    allocates, and split evenly over all hidden layers; fewer than E need
    ``--paged-experts-enable-disk`` and, once the checkpoint is loaded, the layer's cache
    file (``tier_cache``). Layers routed by token id
    (``token_routed_layers``) get every expert when they take at most half of it: their
    experts follow the token sequence, not the hidden state, so a share of host rows misses
    about as often as random access (DeepSeek-V4-Flash: 3 reads per token in each of these
    layers, 0.3-1 in the others)."""
    from sglang.srt.runtime_context import process_model_config

    model = current_model()
    num_layers = process_model_config().num_hidden_layers
    per_expert = sum(
        math.prod(shape) * dtype.itemsize for shape, dtype in layout.values()
    )
    need = num_experts * per_expert * num_layers
    if not model.budget_measured:
        model.budget_measured = True
        model.budget_bytes = host_memory_budget_bytes(need)
        full = token_routed_layers() * num_experts * per_expert
        if model.budget_bytes is not None and 0 < full <= model.budget_bytes // 2:
            model.token_routed = token_routed_layers()
    if model.budget_bytes is None:
        # Every expert fits, judged from the first layer's size: a later layer of another
        # format (e.g. bf16 experts a quantization leaves out) must not pin past it.
        model.pinned_bytes += num_experts * per_expert
        if model.fits_bytes is not None and model.pinned_bytes > model.fits_bytes:
            raise RuntimeError(
                f"Paged experts: the experts need more pinned host memory than the first MoE "
                f"layer suggested (their layers differ in size): "
                f"{model.pinned_bytes / _GIB:.1f} GiB by layer {layer_id}, "
                f"{model.fits_bytes / _GIB:.1f} GiB available; set "
                "--paged-experts-host-memory-gb to what the experts may pin"
            )
        return num_experts
    if layer_id is not None and layer_id < model.token_routed:
        return num_experts
    shared = model.budget_bytes - model.token_routed * num_experts * per_expert
    rows = min(
        num_experts, shared // (num_layers - model.token_routed) // max(1, per_expert)
    )
    if rows == num_experts:
        return rows
    budget = model.budget_bytes / _GIB
    if not enabled:
        raise RuntimeError(
            f"Paged experts: the experts need about {need / _GIB:.1f} GiB of pinned host "
            f"memory, --paged-experts-host-memory-gb is {budget:.1f} GiB; set "
            "--paged-experts-enable-disk to keep the rest on disk, or raise "
            "--paged-experts-host-memory-gb if the host has the memory"
        )
    if platform.machine() not in ("x86_64", "AMD64"):
        # The host answers a waiting step with plain stores (payload, then ticket) and reads
        # a published row after its flag the same way: ordered on x86, not on Arm.
        raise RuntimeError(
            f"Paged experts: the disk tier runs on x86-64 hosts for now, this host is "
            f"{platform.machine()}; set --paged-experts-host-memory-gb so that every expert "
            "fits in host memory"
        )
    if rows < num_slots:
        # A wave can read up to K experts from disk, each into a host row of its own.
        raise RuntimeError(
            f"Paged experts: the host budget ({budget:.1f} GiB) holds {rows} experts per "
            f"layer, fewer than the {num_slots} GPU slots; "
            + (
                "set --paged-experts-host-memory-gb to a larger budget"
                if get_exec_moe().paged_experts_host_memory_gb is None
                else "raise --paged-experts-host-memory-gb"
            )
            + " or set --paged-experts-num-resident to fewer slots"
        )
    if num_slots > _MAX_TIER_SLOTS:
        raise RuntimeError(
            f"Paged experts: the disk tier plans at most {_MAX_TIER_SLOTS} GPU slots per "
            f"layer, --paged-experts-num-resident gives {num_slots}; set it to at most "
            f"{_MAX_TIER_SLOTS}"
        )
    _check_replayable()
    return rows


def tier_cache(
    cache: Optional[ExpertCache], layout: RowsLike, num_experts: int, rows: int
) -> ExpertCache:
    """The cache file of a layer on the disk tier, once the checkpoint is loaded (its key
    follows the files the loader read): the tier cannot do without it. The first such layer
    checks the space a first build needs and says where the files are."""
    from sglang.srt.runtime_context import process_model_config

    if cache is None:
        raise RuntimeError(
            "Paged experts: the experts do not fit the host budget and the disk tier needs "
            "the layer's expert cache, which this checkpoint or layer does not get (the "
            "'Paged experts: expert cache' warning above says why); raise "
            "--paged-experts-host-memory-gb so that every expert fits"
        )
    model = current_model()
    if not model.space_checked:
        model.space_checked = True
        # Every tiered layer's file, at this layer's size (layers routed by token id keep
        # every expert in host memory and build no file).
        num_layers = process_model_config().num_hidden_layers - model.token_routed
        if not cache.exists(num_experts):
            _check_space(
                os.path.dirname(cache.path),
                ObjectLayout.of(layout).file_bytes(num_experts),
                num_layers,
            )
        logger.info(
            "Paged experts: %d of %d experts per layer in host memory (host budget %.1f GiB), "
            "the rest read from the expert cache under %s%s",
            rows,
            num_experts,
            (model.budget_bytes or 0) / _GIB,
            os.path.dirname(cache.path),
            f"; all of them in the {model.token_routed} layers routed by token id"
            if model.token_routed
            else "",
        )
    return cache


def _check_space(directory: str, layer_bytes: int, num_layers: int) -> None:
    """A first build writes each layer's file and leaves the expert cache's free-space margin;
    the model's entries under other keys go first, so that they can make the space. Each file
    is checked again when it is created (``ExpertCache.create``)."""
    need = layer_bytes * num_layers
    margin = _expert_cache._FREE_MARGIN_BYTES
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Paged experts: cannot create the disk tier's directory {directory}: {e}; set "
            "SGLANG_CACHE_DIR to a writable directory"
        ) from e
    _prune_other_keys(directory)
    free = shutil.disk_usage(directory).free
    if need + margin > free:
        raise RuntimeError(
            f"Paged experts: the disk tier needs about {need / _GIB:.0f} GiB under "
            f"{directory} to build its files and keeps {margin / _GIB:.0f} GiB free, "
            f"{free / _GIB:.0f} GiB are free; set SGLANG_CACHE_DIR to a larger disk or raise "
            "--paged-experts-host-memory-gb"
        )


def _check_replayable() -> None:
    """With experts on disk, a captured decode step is replayed after a disk read that did not
    land in time: nothing the step accumulates may count twice. The recurrent (mamba /
    linear-attention) state of the request pool is restored before a replay
    (``save_recurrent_state``); a ReplaySSM ring, the state of the unified memory pool and
    DeepSeek-V4's online c128 compressor are not."""
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
        cuda_graph_fully_disabled,
    )
    from sglang.srt.runtime_context import get_exec, get_memory

    if cuda_graph_fully_disabled() or not check_cuda_graph_backend(
        Phase.DECODE, Backend.FULL
    ):
        return
    problems = []
    mamba = get_exec().mamba
    if mamba.enable_linear_replayssm:
        problems.append("--enable-linear-replayssm would replay its ring twice")
    if mamba.enable_linear_replayssm_spec:
        problems.append("--enable-linear-replayssm-spec would replay its ring twice")
    if get_exec().moe.expert_distribution_recorder_mode is not None:
        problems.append(
            "--expert-distribution-recorder-mode would count the step twice"
        )
    if get_memory().enable_unified_memory:
        problems.append(
            "--enable-unified-memory keeps the recurrent state where a replay cannot "
            "restore it"
        )
    if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        problems.append(
            "SGLANG_OPT_USE_ONLINE_COMPRESS would fold the step's tokens into the "
            "compressed KV twice"
        )
    if problems:
        raise RuntimeError(
            "Paged experts: with experts on disk, a captured decode step is replayed when a "
            f"disk read does not land in time, and {'; '.join(problems)}; drop that, use "
            "--cuda-graph-backend-decode disabled, or raise --paged-experts-host-memory-gb "
            "so that every expert fits"
        )


_PR_SET_TIMERSLACK = 29
_fine_slack = threading.local()


def _fine_timer_slack() -> None:
    """Give the calling thread a 1 ns timer slack, once: the resolve's 50 us poll sleep takes
    ~107 us under Linux's default 50 us slack (a layer's row is seen that much later)."""
    if getattr(_fine_slack, "set", False):
        return
    _fine_slack.set = True
    try:
        ctypes.CDLL(None).prctl(_PR_SET_TIMERSLACK, ctypes.c_ulong(1), 0, 0, 0)
    except (OSError, AttributeError):  # not Linux: sleeps keep their slack
        pass


def fill_file(step, cache: ExpertCache, width: int) -> Dict[str, torch.Tensor]:
    """The layer's experts in its cache file, as ``[E, ...]`` views of each tensor, for a layer
    on the disk tier: the file is checked as ``fill_store`` checks it, else built from every
    expert, K at a time, written straight into the new file (the only write)."""
    loads = step.loads
    num_experts, num_slots = loads.num_experts, loads.num_slots
    if not _directory:
        _directory.append(os.path.dirname(cache.path))
    groups = checked_groups(num_experts, min(width, num_slots))
    # The checked experts' checkpoint pages are read ahead at once and kept until the check
    # ends (a group shares experts with another). A file that passes then gives them back; a
    # build leaves them to the chunk that replays them.
    checked = [e for e in sorted(set().union(*groups)) if e not in loads.keep]
    loads.read_ahead(checked)
    loads.keep.update(checked)
    restored = ran_check = False
    try:
        first = step.run(groups[0])  # also gives the paged layout
        views = cache.map(step.rows_like, num_experts)
        if views is not None:
            ran_check = True
            # Compared on the host: the step's rows are views of the layer's whole K-slot
            # table on the GPU, which must not stay alive next to the later groups'.
            first = {name: rows.to("cpu", copy=True) for name, rows in first.items()}
            stale = check_groups(step, SimpleNamespace(host=views), groups, first)
            if stale is None:
                restored = True
                _outcomes["read"] += 1
                return views
            cache.warn_stale(stale)
            _outcomes["rebuilt"] += 1
    finally:
        loads.keep.difference_update(checked)
        if restored:
            loads.page_out(checked)
    reuse = not ran_check and groups[0] == list(range(min(num_slots, num_experts)))
    views = cache.create(step.rows_like, num_experts)
    if reuse:  # the first group is chunk 0: copied, not run again
        for name, rows in first.items():
            views[name][: len(groups[0])].copy_(rows)
    first = rows = None  # its K-slot table must not stay alive next to the build's
    try:
        build_store(step, SimpleNamespace(host=views), first_done=reuse)
    except BaseException:
        cache.abandon()
        raise
    cache.publish()
    return views


class TieredExpertStore(HostExpertStore):
    """All E experts in the layer's expert cache file, and R of them in ``num_rows`` pinned
    host rows (``host``). The rows are inclusive: an expert moved into a GPU slot is copied from
    its row and keeps it, the slot's old expert is dropped (it keeps its row if it has one,
    the disk has it anyway), and an expert only on disk is first read into the least recently
    used row, whose expert is dropped. Nothing is ever written back from a slot.

    With steps decided on the GPU, the row maps live in the layer's row of the model's device
    state (``_DeviceState``), in which such a step also flags the experts it found only on
    disk; the host then reads them while a captured step waits for them, or after the step.
    """

    def __init__(
        self, rows: RowsLike, num_experts: int, num_rows: int, cache: ExpertCache
    ):
        super().__init__(rows, num_rows)
        self._num_experts = num_experts
        self.cache = cache
        self.num_rows = num_rows
        self._reader = None  # opened at the first read
        # Which expert each host row holds (-1: none), and when it was last written; with steps
        # decided on the GPU, the device copies in ``state`` are authoritative between
        # host-planned steps.
        self.row_expert: List[int] = [-1] * num_rows
        self.row_stamp: List[int] = [0] * num_rows
        self.clock = 0
        self.device = None  # the transfer kernels' device, once bound
        self.state = None  # this layer's row of the model's ``_DeviceState``
        # n, rows [K], slots [K] of the reads a captured step waited for
        self.await_plan = None
        # [K] host rows of a device-planned step's entries (``paged_experts_tiered_plan``)
        self.rows_out = None

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def reader(self) -> ExpertReader:
        if self._reader is None:
            self._reader = ExpertReader(self.cache, self.host)
        return self._reader

    def fill_slots(self, layer, num_slots: int) -> None:
        """Read experts 0..K-1 into the layer's K slots."""
        bufs = {
            name: torch.empty((num_slots, *t.shape[1:]), dtype=t.dtype)
            for name, t in self.host.items()
        }
        requests = [
            (e, {n: b[e].data_ptr() for n, b in bufs.items() if b[e].numel()})
            for e in range(num_slots)
        ]
        read_many([(self.reader, requests)])
        for name, buf in bufs.items():
            getattr(layer, name).data.copy_(buf)

    def bind(self, layer, device_planned: bool) -> None:
        """Fill the host rows with the experts after the GPU's 0..K-1, and prepare the
        transfers for the final K."""
        self.num_slots = getattr(layer, next(iter(self.host))).shape[0]
        # The experts after the slots' first, then the slots' own if there are rows left.
        first = [
            e % self._num_experts
            for e in range(
                self.num_slots, self.num_slots + min(self._num_experts, self.num_rows)
            )
        ]
        self.row_expert = list(first) + [-1] * (self.num_rows - len(first))
        self.row_stamp = [0] * self.num_rows
        read_many([(self.reader, [(e, self._dests(r)) for r, e in enumerate(first)])])
        self.reader.drop_staging()  # sized for all R rows; later reads take at most K
        if not getattr(layer, next(iter(self.host))).is_cuda:
            return  # a layer on the CPU (unit tests only)
        # Host-planned moves copy through the same kernel as the captured ones.
        self.bind_device_gather(layer)
        self.device = self._gather_args[0].device
        if not device_planned:
            return
        model = current_model()
        if model.device_state is None:
            model.device_state = _DeviceState()
        self._device_state = model.device_state
        self.layer = layer
        k, e, r = self.num_slots, self._num_experts, self.num_rows
        self.state = self._device_state.register(self, self.device)
        (
            doorbell,
            self.d_expert_row,
            self.d_row_expert,
            self.d_row_stamp,
            self.d_clock,
        ) = self.state.split([3 * k, e, r, r, 1])
        self.doorbell = doorbell.view(3, k)
        self.rows_out = torch.full((k,), -1, dtype=torch.int32, device=self.device)
        self.write_maps()

    def _dests(self, row: int) -> Dict[str, int]:
        """Where host row ``row`` of each tensor is (made once per row: rows never move; not to
        be changed by the caller)."""
        cache = self.__dict__.get("_dests_cache")
        if cache is None or cache[0] is not self.host:
            cache = self._dests_cache = (self.host, {})
        dests = cache[1].get(row)
        if dests is None:
            dests = cache[1][row] = {
                name: rows[row].data_ptr()
                for name, rows in self.host.items()
                if rows.numel()
            }
        return dests

    # -- host-planned moves -------------------------------------------------------------

    def read_maps(self, row) -> None:
        """Take the row maps from a host copy (tensor or numpy) of the layer's device ``state``
        row."""
        k, e, r = self.num_slots, self._num_experts, self.num_rows
        maps = row[3 * k + e :].tolist()
        self.row_expert, self.row_stamp, self.clock = (
            maps[:r],
            maps[r : 2 * r],
            maps[-1],
        )

    def maps(self) -> List[int]:
        """The row maps as the device ``state`` row holds them after the doorbell: expert_row
        ``[E]``, row_expert and row_stamp ``[R]``, clock."""
        expert_row = [-1] * self._num_experts
        for row, expert in enumerate(self.row_expert):
            if expert >= 0:
                expert_row[expert] = row
        return expert_row + self.row_expert + self.row_stamp + [self.clock]

    def write_maps(self) -> None:
        """Put the row maps into the layer's device ``state`` row, in one copy that does not
        wait for the GPU."""
        host = torch.tensor(self.maps(), dtype=torch.int32).pin_memory()
        self.state[3 * self.num_slots :].copy_(host, non_blocking=True)

    @contextlib.contextmanager
    def host_planned(self):
        # Steps decided on the GPU move the device maps: one read before the waves, one write
        # after.
        if self.state is not None:
            self.read_maps(self.state.cpu())
        try:
            yield
        finally:
            if self.state is not None:
                self.write_maps()

    def in_host_rows(self) -> AbstractSet[int]:
        return {e for e in self.row_expert if e >= 0}

    def plan(self, loads: Sequence[Tuple[int, int]]):
        """Plan moving each ``(expert, slot)``: return the (row, slot) copies and the disk reads
        ``(expert, {tensor: address})`` that must come first."""
        where = {e: r for r, e in enumerate(self.row_expert) if e >= 0}
        # A device-planned step publishes before it copies its host-row hits, so the rows those
        # copies read (stamped with the published clock) may still be in flight while the host
        # reads: read into them last. A captured step's misses never need them (hits + misses
        # <= K <= R); a host-planned step waits for the stream before its reads.
        pending = self.clock if self.rows_out is not None else None
        self.clock += 1
        copies, reads, used = [], [], set()
        disk = []
        for expert, slot in loads:
            row = where.get(expert)
            if row is None:
                disk.append((expert, slot))
                continue
            copies.append((row, slot))
            used.add(row)
            self.row_stamp[row] = self.clock
        if disk:
            free = self._free_rows(used, pending)
            # A step plans at most K experts, and a layer has at least K rows.
            assert len(free) >= len(disk), (len(free), len(disk))
            for (expert, slot), row in zip(disk, free):
                reads.append((expert, self._dests(row)))
                copies.append((row, slot))
                self.row_expert[row] = expert
                self.row_stamp[row] = self.clock
        return copies, reads

    def _free_rows(self, used: AbstractSet[int], pending: Optional[int]) -> List[int]:
        """The rows a read may take (not ``used``), best first: rows the step's copies may still
        read last (stamped ``pending``), then empty rows before full ones, least recently
        stamped first, ties by row: one stable sort of a numpy key."""
        stamps = np.array(self.row_stamp, dtype=np.int64)
        key = (np.array(self.row_expert, dtype=np.int64) >= 0).astype(np.int64) << 32
        key |= stamps + (1 << 31)  # int32 stamps, kept in order below bit 32
        if pending is not None:
            key |= (stamps == pending).astype(np.int64) << 33
        excluded = [r for r in used if 0 <= r < len(key)]
        key[excluded] = np.iinfo(np.int64).max
        return np.argsort(key, kind="stable")[: len(key) - len(excluded)].tolist()

    def apply(self, layer, copies: Sequence[Tuple[int, int]]) -> None:
        """Copy host rows into GPU slots, after the reads landed."""
        if not copies:
            return
        if self.device is None:  # a layer on the CPU (unit tests only)
            for name, host in self.host.items():
                gpu = getattr(layer, name).data
                for row, slot in copies:
                    gpu[slot].copy_(host[row])
            return
        from sglang.kernels.ops.moe.paged_experts import paged_experts_rows_to_slots

        n = len(copies)
        plan = [n, *(r for r, _ in copies), *(s for _, s in copies)]
        plan = torch.tensor(plan, dtype=torch.int32).pin_memory()
        plan = plan.to(self.device, non_blocking=True)
        paged_experts_rows_to_slots(
            *self._gather_args, plan[1 : 1 + n], plan[1 + n :], plan[:1]
        )

    def page_in(self, layer, loads: Sequence[Tuple[int, int]]) -> None:
        if not loads:
            return
        copies, reads = self.plan(loads)
        if reads and self.device is not None:
            # The GPU must be done with the rows the reads are about to rewrite.
            torch.cuda.current_stream().synchronize()
        read_many([(self.reader, reads)])
        self.apply(layer, copies)

    # -- device-planned moves (captured decode) -----------------------------------------

    def gather(self, src, dst, count) -> None:
        """Copy each planned expert ``src[i]`` (``i < count[0]``) from its host row into slot
        ``dst[i]`` (the slot's old expert is dropped); experts only on disk are flagged for the
        host. In a captured step the maps and the missing flags go first, then the layer's row
        is published (the host starts reading the disk), then the host-row hits are copied
        while it reads, and the step waits for its reads (``await_reads``). The host never
        reads into a row these copies read: ``plan`` skips the rows stamped with the published
        clock. Outside a capture the host pages them in after the step (``resolve_pending``)."""
        from sglang.kernels.ops.moe.paged_experts import (
            paged_experts_rows_to_slots,
            paged_experts_tiered_plan,
        )

        capturing = torch.cuda.is_current_stream_capturing()
        publish = self._device_state.publish(self) if capturing else None
        paged_experts_tiered_plan(
            src,
            dst,
            count,
            self.d_expert_row,
            self.d_row_expert,
            self.d_row_stamp,
            self.d_clock,
            self.doorbell,
            self.rows_out,
            *(publish or ()),
        )
        paged_experts_rows_to_slots(*self._gather_args, self.rows_out, dst, count)
        if capturing:
            self._device_state.await_reads(self)

    def resolve_pending(self) -> None:
        """Outside a graph replay: page in at once what the last ``gather`` left on disk."""
        self._device_state.resolve(only=self)


class _Mailbox:
    """Pinned host memory through which captured steps and the host talk, one row per layer:
    the row a step publishes once the layer has run and its flag; and, when steps wait for
    their disk reads, the host's reply (ticket, n, rows [K], slots [K], maps) and the step's
    ack of it."""

    def __init__(self, capacity: int, k: int, e: int, r: int):
        pinned = dict(dtype=torch.int32, pin_memory=True)
        self.rows = torch.zeros(capacity, 3 * k + e + 2 * r + 1, **pinned)
        self.flags = torch.zeros(capacity, **pinned)
        self.replies = torch.zeros(capacity, 3 + 2 * k + e + 2 * r, **pinned)
        self.acks = torch.zeros(capacity, **pinned)
        self._addresses = None
        # numpy views of the same memory: the resolve's poll reads them without torch calls
        self.flags_np = self.flags.numpy()
        self.rows_np = self.rows.numpy()
        self.replies_np = self.replies.numpy()

    def device_addresses(self, i: int) -> Tuple[int, int]:
        """Where the GPU reads row ``i``'s reply and writes its ack."""
        from sglang.kernels.ops.moe.paged_experts import (
            paged_experts_host_device_pointer,
        )

        if self._addresses is None:
            self._addresses = (
                paged_experts_host_device_pointer(self.replies),
                paged_experts_host_device_pointer(self.acks),
            )
        replies, acks = self._addresses
        return replies + i * self.replies.shape[1] * 4, acks + i * 4

    def publish_addresses(self, i: int) -> Tuple[int, int]:
        """Where the GPU writes row ``i`` and its flag (``_DeviceState.publish``)."""
        from sglang.kernels.ops.moe.paged_experts import (
            paged_experts_host_device_pointer,
        )

        if getattr(self, "_publish", None) is None:
            self._publish = (
                paged_experts_host_device_pointer(self.rows),
                paged_experts_host_device_pointer(self.flags),
            )
        rows, flags = self._publish
        return rows + i * self.rows.shape[1] * 4, flags + i * 4

    def reply(self, i: int, ticket: int, copies, maps: List[int]) -> None:
        """Answer row ``i``'s waiting step: the payload first, the ticket last."""
        k = (self.replies.shape[1] - len(maps) - 2) // 2
        row = self.replies_np[i]
        row[1] = len(copies)
        row[2 : 2 + len(copies)] = [r for r, _ in copies]
        row[2 + k : 2 + k + len(copies)] = [s for _, s in copies]
        row[2 + 2 * k :] = maps
        row[0] = ticket

    def clear(self) -> None:
        self.flags.zero_()
        self.replies[:, 0] = 0
        self.acks.zero_()


def _newly_landed(flags: np.ndarray, seen: np.ndarray) -> List[int]:
    """The layers whose row landed (nonzero flag) and were not seen yet, in layer order; marks
    them seen."""
    new = np.flatnonzero((flags[: len(seen)] != 0) & ~seen).tolist()
    seen[new] = True
    return new


class _DeviceState:
    """The device state of a model's tiered layers whose steps are decided on the GPU, one row
    per layer: what the last captured step left on disk (missing flags, src, dst; K
    each) and the layer's row maps (expert_row E, row_expert R, row_stamp R, clock). After a
    graph replay, one check finds every layer's misses, and one copy reads the flagged layers'
    misses and maps together. Layers of one shape share a buffer.

    On a GPU, a captured step also writes each layer's row to pinned host memory once the
    layer has planned, then a nonzero flag (``publish``, done by the plan kernel): the host
    plans that layer's disk reads while the step runs (``_resolve_streaming``), and the step
    waits at the layer (``await_reads``): the host reads the layer's missing experts, replies
    with where they are, and the step copies them in and runs on with the experts it routed
    to, so a step that misses is not replayed, and later layers route as the settled step
    would. A step that gives up waiting (``WAIT_TIMEOUT_MS``) runs on with the slots' old
    experts and is replayed once the host has paged them in."""

    def __init__(self):
        # (k, e, r) -> (device rows, stores, the host mailbox or None)
        self.groups: Dict[
            Tuple[int, int, int], Tuple[torch.Tensor, list, Optional[_Mailbox]]
        ] = {}
        # Whether the replay hook is registered (at the first tiered layer).
        self.can_miss = False
        # Whether captured steps publish their rows (``publish`` ran in a capture), and
        # whether they wait for their disk reads (``await_reads`` did).
        self.streaming = False
        self.timeouts = 0  # replayed steps: a disk read did not land in time
        self.failed: Optional[BaseException] = (
            None  # a read that failed: no step runs on
        )

    def register(self, store: TieredExpertStore, device) -> torch.Tensor:
        """The store's row (a layer binds once, at its final K)."""
        from sglang.srt.runtime_context import process_model_config

        k, e, r = store.num_slots, store.num_experts, store.num_rows
        if (k, e, r) not in self.groups:
            capacity = process_model_config().num_hidden_layers
            buffers = torch.zeros(
                capacity, 3 * k + e + 2 * r + 1, dtype=torch.int32, device=device
            )
            # Layers on the CPU (unit tests only) have no mailbox: they are resolved after a
            # step from a copy of their rows.
            mailbox = _Mailbox(capacity, k, e, r) if buffers.is_cuda else None
            self.groups[(k, e, r)] = (buffers, [], mailbox)
        buffers, stores, _ = self.groups[(k, e, r)]
        if len(stores) >= buffers.shape[0]:
            raise RuntimeError(
                "Paged experts: more MoE layers than the model config's num_hidden_layers"
            )
        stores.append(store)
        if not self.can_miss:
            from sglang.srt.model_executor.runner_backend_utils.replay_hooks import (
                register_replay_hook,
            )

            self.can_miss = True
            register_replay_hook(self.resolve)
        return buffers[len(stores) - 1]

    def publish(self, store: TieredExpertStore) -> Optional[Tuple[int, int, int]]:
        """In a captured step, for the layer's plan kernel (``paged_experts_tiered_plan``),
        which writes the layer's row to the host and then its clock (never 0 after a step) as
        the flag that the row has landed: the row's and the flag's UVA addresses and the row's
        length, or None when the layer does not publish (a layer on the CPU: unit tests only)."""
        _, stores, mailbox = self.groups[
            (store.num_slots, store.num_experts, store.num_rows)
        ]
        if mailbox is None:
            return None
        # The kernel copies the row from the doorbell's first entry.
        assert store.doorbell.data_ptr() == store.state.data_ptr()
        self.streaming = True
        row, flag = mailbox.publish_addresses(stores.index(store))
        return row, flag, store.state.numel()

    def await_reads(self, store: TieredExpertStore) -> None:
        """In a captured step, after ``publish``: wait for the host to read what the layer left
        on disk, then copy it into the slots."""
        from sglang.kernels.ops.moe.paged_experts import (
            paged_experts_await_rows,
            paged_experts_rows_to_slots,
        )

        k, e, r = store.num_slots, store.num_experts, store.num_rows
        _, stores, mailbox = self.groups[(k, e, r)]
        if store.await_plan is None:
            store.await_plan = torch.zeros(
                1 + 2 * k, dtype=torch.int32, device=store.state.device
            )
        reply, ack = mailbox.device_addresses(stores.index(store))
        plan = store.await_plan
        paged_experts_await_rows(
            store.doorbell,
            store.state[3 * k :],
            store.d_clock,
            reply,
            ack,
            plan,
            WAIT_TIMEOUT_MS * 1_000_000,
        )
        paged_experts_rows_to_slots(
            *store._gather_args, plan[1 : 1 + k], plan[1 + k :], plan[:1]
        )

    def resolve(self, only: Optional[TieredExpertStore] = None) -> bool:
        """Page in what the last step left on disk; whether there was anything (after a
        graph replay: whether to replay)."""
        if self.failed is not None:
            raise RuntimeError(
                "Paged experts: a disk read failed in an earlier step, so the tier's host "
                "rows may hold partial experts"
            ) from self.failed
        if only is None and self.streaming:
            return self._resolve_streaming()
        flagged = []  # (store, a host copy of its row)
        if only is not None:
            k = only.num_slots
            if bool(only.state[:k].any()):  # synchronizes with the step
                flagged.append((only, only.state.cpu()))
                only.state[:k].zero_()
        for (k, _, _), (buffers, stores, _) in [] if only else self.groups.items():
            used = buffers[: len(stores)]
            missing = used[:, :k].any(dim=1)
            if not bool(missing.any()):  # synchronizes with the replay
                continue
            layers = missing.nonzero().flatten().tolist()
            host = used[layers].cpu()
            flagged += [(stores[j], host[i]) for i, j in enumerate(layers)]
            used[:, :k].zero_()
        if not flagged:
            return False
        plans = [(store, *self._plan(store, row)) for store, row in flagged]
        read_many([(store.reader, reads) for store, _, reads in plans])
        for store, copies, _ in plans:
            store.apply(store.layer, copies)
            store.write_maps()
        return True

    def _timed_out(self) -> None:
        """Say, at 1, 2, 4, ... such steps, that decode steps waited out their disk reads."""
        self.timeouts += 1
        if self.timeouts & (self.timeouts - 1) == 0:
            logger.warning(
                "Paged experts: %d decode step(s) waited more than %d ms for their disk reads "
                "and were replayed (correct, but slower); raise "
                "--paged-experts-host-memory-gb so that fewer experts are read, or put "
                "SGLANG_CACHE_DIR on a faster NVMe drive",
                self.timeouts,
                WAIT_TIMEOUT_MS,
            )

    @staticmethod
    def _plan(store: TieredExpertStore, row):
        """Plan a layer's loads from a host copy of its device row (tensor or numpy)."""
        if isinstance(row, torch.Tensor):
            row = row.numpy()
        store.read_maps(row)
        k = store.num_slots
        doorbell = row[: 3 * k].reshape(3, k)
        mask = doorbell[0] != 0
        loads = zip(*(doorbell[i][mask].tolist() for i in (1, 2)))
        return store.plan(list(loads))

    def _resolve_streaming(self) -> bool:
        """While the replay runs: plan each layer's disk reads as soon as its row lands on the
        host, read them and answer the step waiting at the layer (``await_reads``). Whether to
        replay: a layer whose step did not take the answer (timed out) is copied in after the
        step instead, and the step replays."""
        _fine_timer_slack()
        done = torch.cuda.Event()
        done.record()
        answered = []
        flagged = False
        failed: Optional[BaseException] = None
        # Per group: its mailbox's flags and which layers were seen; only rows that landed and
        # were not seen yet are looked at, in layer order.
        watched = [
            (key, stores, mailbox, np.zeros(len(stores), dtype=bool))
            for key, (_, stores, mailbox) in self.groups.items()
            if mailbox is not None
        ]
        try:
            while True:
                finished = (
                    done.query()
                )  # rows that landed before the step ended are all seen
                for key, stores, mailbox, seen in watched:
                    landed = mailbox.flags_np
                    for i in _newly_landed(landed, seen):
                        store = stores[i]
                        ticket = int(landed[i])
                        row = mailbox.rows_np[i].copy()
                        if not row[: key[0]].any():
                            continue
                        if failed is None:
                            try:
                                copies, reads = self._plan(store, row)
                                read_many([(store.reader, reads)])
                            except BaseException as e:  # noqa: BLE001
                                failed = e
                        if failed is not None:
                            # After a failed read no layer waits out its timeout: each runs
                            # on at once with its slots and maps as they are, and the step
                            # fails below (its output is never used).
                            mailbox.reply(i, ticket, [], row[3 * key[0] :].tolist())
                            continue
                        mailbox.reply(i, ticket, copies, store.maps())
                        answered.append((store, copies, mailbox, i, ticket))
                if finished:
                    break
                time.sleep(50e-6)
            if failed is not None:
                self.failed = failed
                raise failed
            for store, copies, mailbox, i, ticket in answered:
                if (
                    int(mailbox.acks[i]) != ticket
                ):  # the step timed out before the answer: copy in after it, replay
                    flagged = True
                    store.apply(store.layer, copies)
                    store.write_maps()
            if flagged:
                self._timed_out()
        finally:
            # A failed read leaves the step running: it must not publish into a cleared
            # mailbox (it would wait, and the next step would see its flags).
            done.synchronize()
            for (k, _, _), (buffers, stores, mailbox) in self.groups.items():
                if mailbox is not None:
                    buffers[: len(stores), :k].zero_()
                    mailbox.clear()
        return flagged
