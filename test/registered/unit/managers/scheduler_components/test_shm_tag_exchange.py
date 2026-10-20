"""Unit tests for srt/managers/scheduler_components/shm_tag_exchange.py."""

import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils import stale_shm_cleanup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Arbitrary, with both 32-bit halves non-zero: the shm row carries the digest folded.
_DIGEST = (7 << 32) | 5


def _queue(size: int) -> Queue:
    queue = Queue()
    for _ in range(size):
        queue.put(None)
    return queue


def _hicache_tp_rank(rank: int, extra_pools: int) -> UnifiedRadixCache:
    # Count i is i on one rank and i + 1 on the other, alternating, so the minimum is i.
    counts = [i + (rank if i % 2 else 1 - rank) for i in range(6 + extra_pools)]
    queues = [_queue(count) for count in counts[2:]]
    ready = SimpleNamespace(finish_event=SimpleNamespace(query=lambda: True))
    cache = object.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_storage=True, write_back_duplicate_reclaim_digest=_DIGEST
    )
    cache.pp_rank, cache.pp_size, cache.pp_group = 0, 1, None
    cache.host_memory_mode = "cache"
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.tp_group, cache.tp_world_size = torch.distributed.group.WORLD, 2
    cache.cache_controller = SimpleNamespace(
        ack_write_queue=[ready] * counts[0],
        ack_load_queue=[ready] * counts[1],
        prefetch_hit_queue=queues[0],
        ack_prefetch_queue=queues[1],
        ack_backup_queue=queues[2],
        host_mem_release_queue=queues[3],
        extra_host_mem_release_queues={
            f"extra{i}": queue for i, queue in enumerate(queues[4:])
        },
    )
    return cache


def _sync_hicache_ready_counts(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=2
    )
    try:
        with (
            patch.dict(os.environ, {"SGLANG_Q35MF_EDGE_SHM_SYNC": "1"}),
            # macOS caps POSIX shared-memory names at 30 bytes;
            # elsewhere keep the prefix the CI stale-shm sweep reclaims.
            (
                patch.object(stale_shm_cleanup, "_SGL_SHM_PREFIX", "s")
                if sys.platform == "darwin"
                else nullcontext()
            ),
        ):
            # A row holds 6 ready counts, one more per extra host pool, and the folded
            # digest's 2 words: 8, 9 and 10 values here. The shm table holds 8.
            for extra_pools in (0, 1, 2):
                cache = _hicache_tp_rank(rank, extra_pools)
                want = list(range(6 + extra_pools))
                names = tuple(cache.cache_controller.extra_host_mem_release_queues)
                got = cache._sync_hicache_ready_counts()
                assert got == (want[0], want[1], tuple(want[2:]), names), (
                    extra_pools,
                    got,
                )
            exchange = cache._shm_count_exchange()
            assert exchange.seq == 1, f"{exchange.seq} rows went through shm, not 1"
            exchange.close()
    finally:
        torch.distributed.destroy_process_group()


class TestHiCacheReadyCountsThroughShm(CustomTestCase):
    def test_rows_wider_than_the_table_keep_the_gloo_reduce(self):
        """With HiCache storage on, each extra host pool adds a ready count. A row
        wider than the shm table must still reduce to the minimum over ranks of each
        count, not fail the exchange or overwrite the next rank's row."""
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _sync_hicache_ready_counts,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
