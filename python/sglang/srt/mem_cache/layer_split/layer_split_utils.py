"""Shared communication scheduling, page results and layer/rank geometry."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from concurrent.futures import Future
from dataclasses import astuple, dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Sequence, Tuple

import psutil
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.hicache_storage import PoolName

logger = logging.getLogger(__name__)
PREFETCH, BACKUP = 0, 1
STAGE_NAMES = ("prefetch", "backup")

LOGICAL_COMPONENT_POOLS = {
    "target": PoolName.KV,
    "indexer": PoolName.INDEXER,
}


def complete_page_mask(
    results: Dict[Any, Sequence[bool]], page_count: int
) -> List[bool]:
    """A staged page is complete only if every component's independent I/O succeeded."""
    masks = []
    for name in LOGICAL_COMPONENT_POOLS.values():
        mask = results.get(name)
        if mask is None:
            raise ValueError(
                f"batch I/O result is missing the {name.name} page mask; got {list(results)}"
            )
        masks.append(mask)
    return [
        all(i < len(mask) and bool(mask[i]) for mask in masks)
        for i in range(page_count)
    ]


def owned_layer_range(rank: int, shard_size: int, layer_count: int) -> Tuple[int, int]:
    """Use the same partition as the GPU pool; no independent geometry copy."""
    from sglang.srt.layers.cp.utils import get_layer_shard_range

    return get_layer_shard_range(rank, shard_size, layer_count)


def layer_split_blocks(
    world_size: int, shard_size: int, exchange_ranks: Sequence[int]
) -> Tuple[List[List[int]], List[int]]:
    """The world's layer-split blocks, and which one this rank belongs to.

    Deterministic across ranks: contiguous blocks of ``shard_size`` in rank order.
    Every rank derives the identical list, which is what lets each of them call
    ``new_group`` for *every* block in the same order -- a requirement, since
    ``new_group`` is a collective over the whole world and a rank that skips the
    blocks it is not a member of leaves the ranks that do call them deadlocked.

    Raises when the world is not partitioned the way the exchange assumes, rather
    than proceeding with per-destination split sizes that would not match
    ``shard_size``.

    Shared by both staging directions, so the two cannot disagree about the
    partition they create their groups against.
    """
    if world_size % shard_size != 0:
        raise ValueError(
            f"staging exchange needs the world ({world_size}) to divide into "
            f"layer-split groups of {shard_size}; only the GLM-5.2 "
            "DSA layer-split shape is supported"
        )
    blocks = [
        list(range(start, start + shard_size))
        for start in range(0, world_size, shard_size)
    ]
    mine = sorted(exchange_ranks)
    if mine not in blocks:
        raise ValueError(
            f"staging exchange_ranks {mine} is not a contiguous block of "
            f"{shard_size} in a world of {world_size}; only the "
            "GLM-5.2 DSA layer-split shape is supported"
        )
    return blocks, mine


def _request_shutdown():
    try:
        psutil.Process(os.getppid()).send_signal(signal.SIGQUIT)
    except Exception:
        logger.exception("Could not signal the parent process")


@dataclass
class WindowJob:
    """A ready window, identified by cross-rank plan/window content, not local op_id."""

    fingerprint: int
    values: Callable
    execute: Callable
    result: Future = field(default_factory=Future)


class SharedWindowPGPool:
    """One set of data PGs, serially scheduled across two independently-ready lanes.

    The control PG first MIN-reduces fixed-width readiness/identity probes. A
    deterministic alternating preference picks one lane ready on every rank.
    Only that lane then agrees its window result and submits data collectives.
    An idle lane never blocks the other. The dispatcher polls control every
    10 ms when no lane is globally ready; measure that cost before production use.
    """

    def __init__(self, config, exchange_ranks):
        self.config = config
        self.exchange_ranks = list(exchange_ranks)
        self.data_groups = []
        self.control_group = None
        self._jobs = [None, None]
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._abort = threading.Event()
        self._completion = Future()
        self._thread = None
        self.geometry = None

    def attach(self):
        if not dist.is_initialized():
            raise RuntimeError("Shared staging requires an initialized process group")
        world = dist.get_world_size()
        options = [None] * world
        dist.all_gather_object(options, (astuple(self.config), self.geometry))
        if any(option != options[0] for option in options):
            raise ValueError("Shared staging configuration differs across ranks")
        blocks, mine = layer_split_blocks(
            world, self.config.shard_size, self.exchange_ranks
        )
        for _ in range(self.config.exchange_group_count):
            for block in blocks:
                group = dist.new_group(
                    ranks=block,
                    backend="gloo",
                    timeout=timedelta(seconds=self.config.exchange_timeout_s),
                )
                if block == mine:
                    self.data_groups.append(group)
        for block in blocks:
            group = dist.new_group(
                ranks=block,
                backend="gloo",
                timeout=timedelta(seconds=self.config.window_agreement_timeout_s),
            )
            if block == mine:
                self.control_group = group
        self._thread = threading.Thread(
            target=self._dispatch, name="l3-shared-communication", daemon=True
        )
        self._thread.start()

    def _min(self, values):
        tensor = torch.tensor(values, dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.control_group)
        return tensor.tolist()

    def submit(self, lane, job):
        with self._lock:
            if self._completion.done():
                self._completion.result()
                raise RuntimeError("Shared communication thread has stopped")
            # One caller per direction, one submitted window at a time.
            self._jobs[lane] = job
        return job.result.result(timeout=self.config.window_agreement_timeout_s)

    def abort(self):
        self._abort.set()

    def _dispatch(self):
        preferred = PREFETCH
        try:
            while True:
                with self._lock:
                    jobs = tuple(self._jobs)
                probe = []
                for job in jobs:
                    fp = job.fingerprint if job is not None else 0
                    probe.extend((int(job is not None), fp, -fp))
                probe.extend(
                    (int(self._closing.is_set()), int(not self._abort.is_set()))
                )
                agreed = self._min(probe)
                if not agreed[7]:
                    raise RuntimeError(
                        "A shared staging participant requested shutdown"
                    )
                if agreed[6]:
                    break
                ready = [
                    lane for lane in (preferred, 1 - preferred) if agreed[3 * lane]
                ]
                if not ready:
                    time.sleep(0.01)
                    continue
                lane = ready[0]
                # BACKUP_ORDER_CONTRACT: this checks the upstream sequence;
                # it does not reorder different heads or recover missing ops.
                if agreed[3 * lane + 1] != -agreed[3 * lane + 2]:
                    raise RuntimeError("Shared staging ranks offered different windows")
                job = jobs[lane]
                window_result = self._min(job.values())
                output = job.execute(window_result)
                # Remove before waking the producer, which may enqueue its next window.
                with self._lock:
                    self._jobs[lane] = None
                job.result.set_result(output)
                preferred = 1 - lane
        except Exception as exc:
            logger.exception("Shared staging communication failed; requesting shutdown")
            with self._lock:
                self._completion.set_exception(exc)
                for job in self._jobs:
                    if job is not None and not job.result.done():
                        job.result.set_exception(exc)
            _request_shutdown()
        else:
            self._completion.set_result(None)

    def assert_idle(self):
        with self._lock:
            if any(job is not None for job in self._jobs):
                raise RuntimeError("Shared communication has queued or active windows")
            if self._completion.done():
                self._completion.result()

    def close(self):
        """Collective close after both local producer threads have returned."""
        self._closing.set()
        self._thread.join(self.config.window_agreement_timeout_s + 1)
        if self._thread.is_alive():
            raise TimeoutError("Shared communication thread did not stop")
        self._completion.result()
        for group in (self.control_group, *reversed(self.data_groups)):
            dist.destroy_process_group(group)
        self.control_group = None
        self.data_groups.clear()
