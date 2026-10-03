"""Two real CPU/Gloo ranks exercise asymmetric backup pressure."""

import tempfile
import unittest
from datetime import timedelta

import test_mamba_write_through_pressure as pressure
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def worker(rank, init_file, missing_ack):
    fixture = pressure.TestMambaWriteThroughPressure()
    cache, pool, nodes = fixture.make_cache(active=1 if rank == 0 else 2)
    cc = fixture.submit_backup(cache, nodes)
    if missing_ack and rank == 0:
        cc.ack_write_queue.clear()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        cache.tp_world_size = 2
        cache.tp_group = dist.group.WORLD
        cache.attn_tp_group = cache.attn_cp_group = None
        waited = cache._wait_for_mamba_write_through(1)
        assert waited == (not missing_ack)
        assert len(cache.ongoing_write_through) == (30 if missing_ack else 29)
        if waited:
            assert cache.mamba_evictable_size() == 1
    finally:
        dist.destroy_process_group()


class TestMambaWriteThroughConsensus(unittest.TestCase):
    def test_rank_with_free_slot_participates_in_peer_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(f"{directory}/gloo", False), nprocs=2)

    def test_missing_ack_on_one_rank_stops_both_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(f"{directory}/gloo", True), nprocs=2)


if __name__ == "__main__":
    unittest.main()
