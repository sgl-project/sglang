"""Real two-process Gloo agreement; no CUDA device or EP bindings required."""

import multiprocessing
import sys
import time
from datetime import timedelta

import pytest
import torch.distributed as dist

from sglang.srt.layers.moe.token_dispatcher.nccl_ep_admission import (
    NcclEpGraphAdmission,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _rank(rank, rendezvous, queue):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        admission = NcclEpGraphAdmission(dist.group.WORLD)
        # Eligibility includes local bucket capacity and mode. The third
        # scenario represents an active FULL rank beside an idle NULL rank.
        cases = [
            ((True, 0, 0), (True, 0, 0)),
            ((False, 0, 0), (True, 0, 0)),
            ((True, 2, 2), (True, 0, 0)),
            ((True, 2, 2), (True, 0, 2)),
            ((True, 0, 2), (True, 0, 2)),
            ((True, 0, 0), (False, 0, 0)),
            ((True, 0, 0), (True, 0, 0)),
        ]
        results = []
        for pair in cases:
            eligible, required, captured = pair[rank]
            decision = admission.decide(
                eligible=eligible, required_mode=required, captured_mode=captured
            )
            results.append(
                (decision.can_run, decision.capture_hidden_mode, decision.recapture)
            )
        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


def test_all_ranks_agree_on_eager_replay_and_recapture(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    rendezvous = "file://" + str(tmp_path / "gloo")
    processes = [
        context.Process(target=_rank, args=(rank, rendezvous, queue))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(
            not process.is_alive() for process in processes
        ), "Gloo agreement hung"
        assert [process.exitcode for process in processes] == [0, 0]
        results = dict(queue.get(timeout=2) for _ in processes)
        expected = [
            (True, 0, False),
            (False, 0, False),
            (True, 2, True),
            (True, 2, False),
            (True, 0, True),
            (False, 0, False),
            (True, 0, False),
        ]
        assert results == {0: expected, 1: expected}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        queue.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
