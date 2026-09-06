"""A preallocation timeout must enter rank consensus before queue mutation."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver
from sglang.srt.disaggregation.utils import poll_and_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _check_two_rank_timeout(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        status = [KVPoll.WaitingForInput]
        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = SimpleNamespace(
            waiting_timeout=5,
            check_status=lambda room: status[0],
            record_failure=lambda room, message: None,
            update_status=lambda room, value: status.__setitem__(0, value),
        )
        receiver.bootstrap_room = 1
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.abort_notified = True
        receiver._connection_pool_entries = {}
        with patch("sglang.srt.disaggregation.common.conn.time.time") as clock:
            clock.return_value = 10.0
            assert receiver.poll() == KVPoll.WaitingForInput
            # Only rank zero reaches the deadline. Neither rank has allocated
            # or published KV destinations; both must observe the failed poll.
            clock.return_value = 15.0 if rank == 0 else 14.0
            polls = poll_and_all_reduce([receiver], dist.group.WORLD)
            assert polls == [KVPoll.Failed], (rank, polls)
            assert not receiver.started_transfer
    finally:
        dist.destroy_process_group()


class TestPreallocationTimeoutConsensus(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "requires Gloo"
    )
    def test_one_rank_timeout_fails_both_ranks_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "gloo-rendezvous")
            mp.spawn(_check_two_rank_timeout, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
