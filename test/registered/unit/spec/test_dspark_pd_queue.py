"""All PP/TP participants agree on P/D readiness before mutating queues."""

import multiprocessing as mp
import tempfile
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.speculative.dspark_components.dspark_pd_queue import (
    DSparkPDQueueCoordinator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


class World:
    def all_gather_object(self, value):
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, value)
        return records


def exercise(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/init",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        sync = DSparkPDQueueCoordinator(
            World(), SimpleNamespace(disaggregation_transfer_backend="mooncake")
        )
        request = SimpleNamespace(
            rid="ready",
            bootstrap_room=71,
            bootstrap_host="127.0.0.1",
            finished_reason=None,
        )
        status = KVPoll.Success
        receiver = SimpleNamespace(poll=lambda: status)
        entry = SimpleNamespace(
            req=request, kv_receiver=receiver, metadata_buffer_index=0
        )
        metadata = SimpleNamespace(
            bootstrap_room=torch.tensor([[0 if rank == 2 else 71]])
        )
        assert sync.poll(
            "transfer", [entry], is_send=False, metadata_buffers=metadata
        ) == [KVPoll.Transferring]
        metadata.bootstrap_room[0, 0] = 71
        assert sync.poll(
            "transfer", [entry], is_send=False, metadata_buffers=metadata
        ) == [KVPoll.Success]
        metadata.bootstrap_room[0, 0] = 99 if rank == 1 else 71
        assert sync.poll(
            "corrupt", [entry], is_send=False, metadata_buffers=metadata
        ) == [KVPoll.Failed]
        metadata.bootstrap_room[0, 0] = 71
        status = KVPoll.Failed if rank == 3 else KVPoll.Transferring
        assert sync.poll("failure", [entry], is_send=False) == [KVPoll.Failed]
        assert sync.poll("pending_failure", [entry], is_send=False, terminal=True) == [
            KVPoll.Transferring
        ]
        status = KVPoll.Failed if rank == 3 else KVPoll.Success
        assert sync.poll("finished_failure", [entry], is_send=False, terminal=True) == [
            KVPoll.Failed
        ]
        status = KVPoll.WaitingForInput
        request.finished_reason = FINISH_ABORT("cancelled") if rank == 0 else None
        assert sync.poll("abort", [entry], is_send=False) == [KVPoll.Failed]
        status = KVPoll.Transferring
        request.finished_reason = FINISH_ABORT("cancelled")
        assert sync.poll("abort_inflight", [entry], is_send=False, terminal=True) == [
            KVPoll.Transferring
        ]
        status = KVPoll.Success
        assert sync.poll("abort_finished", [entry], is_send=False, terminal=True) == [
            KVPoll.Failed
        ]
        status = KVPoll.WaitingForInput
        request.finished_reason = None
        request.disagg_kv_sender = receiver
        assert sync.poll("bootstrap", [request], is_send=True) == [
            KVPoll.WaitingForInput
        ]
        assert sync.minimum("capacity", (8 - rank, 1000 - 10 * rank, rank - 2)) == (
            5,
            970,
            -2,
        )
        assert sync.poll("empty", [], is_send=False) == []

        def must_fail(action, message):
            try:
                action()
            except RuntimeError as error:
                assert message in str(error), error
            else:
                raise AssertionError("accepted divergent P/D queue state")

        request.rid = "other" if rank == 1 else "ready"
        must_fail(
            lambda: sync.poll("identity", [entry], is_send=False), "request order"
        )
        request.rid = "ready"
        must_fail(
            lambda: sync.poll(
                "empty_mismatch", [] if rank == 2 else [entry], is_send=False
            ),
            "request order",
        )
        must_fail(lambda: sync.agree("state", rank), "queue state")
        must_fail(
            lambda: sync.minimum("shape", (1, 2) if rank == 3 else (1,)),
            "capacity shape",
        )
        must_fail(
            lambda: sync.poll(
                "wrong_phase" if rank == 0 else "phase", [], is_send=False
            ),
            "phase failed",
        )

        def broken():
            raise RuntimeError("backend poll failed")

        if rank == 2:
            receiver.poll = broken
        must_fail(
            lambda: sync.poll("exception", [entry], is_send=False),
            "backend poll failed",
        )
        receiver.poll = lambda: KVPoll.Success
        assert sync.poll("healthy", [entry], is_send=False) == [KVPoll.Success]
    finally:
        dist.destroy_process_group()


class TestDSparkPDQueues(CustomTestCase):
    def test_bootstrap_respects_the_smallest_metadata_capacity(self):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.pp_size = 2
        queue.queue = [
            SimpleNamespace(rid=f"req-{i}", metadata_buffer_index=-1, time_stats=Mock())
            for i in range(3)
        ]
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 8
        )
        queue.scheduler = SimpleNamespace(
            dspark_pd_queue_coordinator=SimpleNamespace(
                poll=lambda *args, **kwargs: [KVPoll.WaitingForInput] * 3,
                minimum=lambda *args: (1,),
            )
        )
        queue.finalize_bootstrap = Mock(return_value=True)
        with patch(
            "sglang.srt.disaggregation.prefill.should_force_retry", return_value=False
        ):
            ready = queue.pop_bootstrapped()
        self.assertEqual([req.rid for req in ready], ["req-0"])
        self.assertEqual([req.rid for req in queue.queue], ["req-1", "req-2"])
        queue.finalize_bootstrap.assert_called_once_with(ready[0])

    def test_four_rank_readiness_capacity_and_failures(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=exercise, args=(rank, root)) for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 180
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
