"""Unit tests for decode HiCache TreeCore interactions."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_storage_probe_and_prefetch_use_node_handles(self):
        ongoing_prefetch = {}

        def register_prefetch(req_id, *_args, **_kwargs):
            ongoing_prefetch[req_id] = object()

        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch=ongoing_prefetch,
            has_ongoing_prefetch=ongoing_prefetch.__contains__,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="h2"),
            get_prefix_hash_values=Mock(return_value=["h0", "h1"]),
            query_storage_hit_length=Mock(return_value=2),
            prefetch_from_storage=Mock(side_effect=register_prefetch),
        )
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=True),
            tree_cache=tree_cache,
        )
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            extra_key="model",
            cache_salt="tenant-a",
        )
        result = SimpleNamespace(
            device_indices=torch.tensor([10, 11]),
            host_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        prefix_match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
            harness, req, result
        )

        self.assertEqual(prefix_match.l3_storage_hit_length, 2)
        tree_cache.query_storage_hit_length.assert_called_once_with(
            22,
            [4, 5, 6, 7],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertTrue(prefix_match.prefetch_registered)
        tree_cache.prefetch_from_storage.assert_called_once_with(
            req.cache_request_handle,
            22,
            [4, 5],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

    def test_stale_prefetch_anchor_degrades_to_l2(self):
        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch={},
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            extra_key=None,
            cache_salt=None,
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertFalse(prefix_match.prefetch_registered)
        tree_cache.get_prefix_hash_values.assert_not_called()
        tree_cache.prefetch_from_storage.assert_not_called()

    def test_rank_divergent_load_events_issue_no_collective(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                _run_local_restore_rank,
                args=(str(Path(tmp) / "init"),),
                nprocs=2,
                join=True,
            )


def _run_local_restore_rank(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        event = SimpleNamespace(query=lambda: rank == 0)
        cache = object.__new__(UnifiedRadixCache)
        cache.__dict__.update(
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(
                    events=[SimpleNamespace(finish_event=event)] * 3,
                    producer_index=0,
                    num_counters=3,
                ),
                ack_load_queue=[SimpleNamespace(finish_event=event)] * 3,
            ),
            tree_core=SimpleNamespace(write_back_duplicate_reclaim_digest=0),
            pp_rank=0,
            pp_size=1,
            attn_cp_group=None,
            attn_tp_group=torch.distributed.group.WORLD,
        )
        decode_req = SimpleNamespace(
            hicache_restore_status=HiCacheRestoreResult.PENDING,
            prefix_match=SimpleNamespace(needs_local_restore=True),
            hicache_restored_node=7,
            hicache_load_consumer_index=0,
        )
        queue = SimpleNamespace(
            tree_cache=cache,
            _try_hicache_queue_load_back=Mock(return_value=False),
        )
        pending = SimpleNamespace(**vars(decode_req))
        pending.hicache_restored_node = None

        DecodeHiCacheTransferMixin._process_hicache_local_restores(
            queue, [decode_req, pending]
        )

        queue._try_hicache_queue_load_back.assert_not_called()
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
