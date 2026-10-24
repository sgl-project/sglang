import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.unified_cache.linker_attachment import CacheLinkerAttachment
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _attachment():
    kvcache = SimpleNamespace(register_layer_transfer_counter=Mock())
    cache = SimpleNamespace(close_cache_linker=Mock(), _all_reduce_attn_groups=Mock())
    attachment = CacheLinkerAttachment(
        cache,
        server_args=None,
        params=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: kvcache)
        ),
        register_layer_transfer_counter=Mock(),
    )
    attachment.storage_backend = "mori"
    return attachment


def _detach_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        for failure in (None, "close", "kv_counter", "worker_counter", "detached_peer"):
            attachment = _attachment()
            attachment._cache._all_reduce_attn_groups = dist.all_reduce
            if failure == "detached_peer" and rank == 0:
                attachment.storage_backend = None
            if rank == 1 and failure is not None:
                operation = {
                    "close": attachment._cache.close_cache_linker,
                    "detached_peer": attachment._cache.close_cache_linker,
                    "kv_counter": attachment._kvcache.register_layer_transfer_counter,
                    "worker_counter": attachment._register_layer_transfer_counter,
                }[failure]
                operation.side_effect = RuntimeError("injected detach failure")
            try:
                attachment.detach()
            except RuntimeError as error:
                assert failure is not None
                assert "scheduler must stop" in str(error), str(error)
                assert (error.__cause__ is not None) == (rank == 1)
            else:
                assert failure is None, "A rank resumed after a peer failed detach"
                assert attachment.storage_backend is None
            # Both ranks reached the same decision; no rank returned early.
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestLinkerDetachConsensus(unittest.TestCase):
    def test_detach_failure_reaches_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "store").as_uri()
            mp.spawn(_detach_worker, args=(rendezvous,), nprocs=2, join=True)

    def test_teardown_does_not_wait_for_peers(self):
        attachment = _attachment()
        attachment.detach(synchronize=False)
        attachment._cache.close_cache_linker.assert_called_once()
        attachment._cache._all_reduce_attn_groups.assert_not_called()
        self.assertIsNone(attachment.storage_backend)

    def test_teardown_failure_does_not_wait_for_peers(self):
        attachment = _attachment()
        attachment._cache.close_cache_linker.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            attachment.detach(synchronize=False)
        attachment._cache._all_reduce_attn_groups.assert_not_called()


if __name__ == "__main__":
    unittest.main()
