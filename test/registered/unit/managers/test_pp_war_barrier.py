"""The write-after-read barrier under unified memory on every device, and once per
launched micro-batch in the PP loop, before an aborted chunk's row is reused."""

import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.srt.arg_groups.kv_cache_hook import handle_page_major_kv_layout
from sglang.srt.environ import envs
from sglang.srt.managers import utils as manager_utils
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.utils import GenerationBatchResult, is_war_barrier_enabled
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

# Ids of the aborted request (A) and of the request that reuses its row (B).
OLD_OWNER, NEW_OWNER = 1, 2


class TestUnifiedMemoryHasTheBarrier(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))
        enter_scope(self, envs.SGLANG_ENABLE_WAR_BARRIER.override(False))
        # Off CUDA (ROCm, XPU, MUSA, ...).
        enter_scope(self, mock.patch.object(manager_utils, "is_cuda", lambda: False))

    def test_off_cuda_the_barrier_follows_unified_memory(self):
        for unified in (True, False):
            with (
                self.subTest(unified=unified),
                get_memory().override(enable_unified_memory=unified),
            ):
                self.assertEqual(is_war_barrier_enabled(), unified)

    def test_unified_memory_refused_without_device_streams(self):
        server_args = ServerArgs(model_path="dummy")
        for name, value in {"enable_unified_memory": True, "device": "mps"}.items():
            msgspec.Struct.__setattr__(server_args, name, value)
        with self.assertRaisesRegex(ValueError, "not supported on MPS"):
            handle_page_major_kv_layout(server_args)


class _StopLoop(Exception):
    pass


class _PPStage:
    """A last PP stage running the real `event_loop_pp` with stand-ins: chunk 1 of A
    takes row R, then A is aborted and B takes R at once; `log` keeps host order."""

    def __init__(
        self,
        *,
        schedule_stream,
        forward_stream,
        launch_forward,
        write_row,
        reads_end_early,
        log=None,
    ):
        self.log = [] if log is None else log
        # Whether the forward publishes the end of its shared reads, or the
        # barrier falls back to waiting for the whole forward.
        self.reads_end_early = reads_end_early
        self._write_row = write_row
        self._launch_forward = launch_forward
        self._plan = iter(["A, chunk 1", "B"])

        s = Scheduler.__new__(Scheduler)
        s._war_barrier_enabled = True
        s.token_to_kv_pool_allocator = None  # not unified memory
        s.schedule_stream = schedule_stream
        s.forward_stream = forward_stream
        s.model_worker = SimpleNamespace(
            last_shared_read_runner=SimpleNamespace(shared_read_done_event=None)
        )
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)
        s.pause_state = PauseState()
        s._sched_idled = False
        s.increment_scheduler_iterations = mock.Mock()
        s.invariant_checker = mock.Mock()
        s.on_idle = mock.Mock()
        s._pp_recv_proxy_tensors = mock.Mock(return_value=None)
        s._pp_commit_comm_work = mock.Mock()
        s._pp_commit_proxy_send_work = mock.Mock()
        s._pp_ingest_requests = mock.Mock(side_effect=[[], [], _StopLoop()])
        s.get_next_batch_to_run = self._schedule
        s._pp_launch_batch = self._launch
        s._pp_commit_send_output_work_and_preprocess_output_tensors = self._take_output
        s._pp_process_batch_result = self._process
        self.scheduler = s
        self.done = {}

    def run(self):
        # Communication is mocked: no dedicated communication stream.
        with (
            get_parallel().override(pp_size=2, pp_async_batch_depth=0),
            envs.SGLANG_PP_COMM_OVERLAP.override(False),
        ):
            try:
                Scheduler.event_loop_pp(self.scheduler)
            except _StopLoop:
                return
        raise AssertionError("the loop must run until intake stops it")

    def _schedule(self, running_batch, last_batch):
        batch = next(self._plan, None)
        if batch == "A, chunk 1":
            self.log.append(("schedule", "write", OLD_OWNER))
            self._write_row(OLD_OWNER)
        elif batch == "B":
            # The abort frees A's row; B is admitted into it at once.
            self.log.append(("schedule", "free", OLD_OWNER))
            self.log.append(("schedule", "write", NEW_OWNER))
            self._write_row(NEW_OWNER)
        return SimpleNamespace(running_batch=running_batch, batch_to_run=batch)

    def _launch(self, mb_id, batch, *args):
        self.log.append(("forward", "launch", batch))
        read_done, self.done[batch] = self._launch_forward(batch)
        runner = self.scheduler.model_worker.last_shared_read_runner
        runner.shared_read_done_event = read_done if self.reads_end_early else None
        return SimpleNamespace(), None

    def _take_output(self, next_first_rank_mb_id, next_mb_id):
        batch = self.scheduler.mbs[next_mb_id]
        if batch is None:
            return None, None, None
        return None, GenerationBatchResult(), self.done[batch]

    def _process(self, batch, result):
        self.log.append(("host", "process", batch))


class _LoggingStream:
    def __init__(self, name, log):
        self.name = name
        self._log = log

    def wait_event(self, event):
        self._log.append((self.name, "wait", event))

    def wait_stream(self, stream):
        self._log.append((self.name, "wait", stream.name))


class TestPPLoopBarrierPlacement(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler", pp_size=2))
        enter_scope(self, envs.SGLANG_FORCE_COARSE_WAR_BARRIER.override(False))

    def _run(self, reads_end_early):
        log = []
        stage = _PPStage(
            schedule_stream=_LoggingStream("schedule", log),
            forward_stream=_LoggingStream("forward", log),
            launch_forward=lambda batch: (
                f"reads of {batch} done",
                SimpleNamespace(synchronize=lambda: None),
            ),
            write_row=lambda owner: None,
            reads_end_early=reads_end_early,
            log=log,
        )
        stage.run()
        return log

    def test_row_reuse_waits_for_the_running_chunk(self):
        for reads_end_early, waited in (
            (True, "reads of A, chunk 1 done"),
            (False, "forward"),
        ):
            with self.subTest(reads_end_early=reads_end_early):
                log = self._run(reads_end_early)
                barrier = log.index(("schedule", "wait", waited))
                self.assertLess(log.index(("forward", "launch", "A, chunk 1")), barrier)
                self.assertLess(barrier, log.index(("schedule", "free", OLD_OWNER)))
                self.assertLess(barrier, log.index(("schedule", "write", NEW_OWNER)))

    def test_barrier_follows_result_processing(self):
        """The output of the micro-batch being processed is not held behind the
        forward just launched: its barrier comes after the processing."""
        log = self._run(reads_end_early=False)
        waits = [i for i, entry in enumerate(log) if entry[:2] == ("schedule", "wait")]
        # One barrier per launched micro-batch.
        self.assertEqual(len(waits), 2)
        launch_b = log.index(("forward", "launch", "B"))
        process_a = log.index(("host", "process", "A, chunk 1"))
        self.assertLess(launch_b, process_a)
        self.assertLess(process_a, waits[1])


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA streams")
class TestPPRowReuseOnDevice(CustomTestCase):
    """With real streams: the chunk's forward reads row R late; the scheduler
    hands R to the new owner meanwhile. Its read must still see the old owner."""

    SLEEP_CYCLES = 200_000_000

    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler", pp_size=2))
        enter_scope(self, envs.SGLANG_FORCE_COARSE_WAR_BARRIER.override(False))

    def _seen_by_running_chunk(self, *, barrier, reads_end_early):
        schedule_stream = torch.cuda.Stream()
        forward_stream = torch.cuda.Stream()
        row = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
        schedule_stream.wait_stream(torch.cuda.current_stream())
        seen = {}

        def launch_forward(batch):
            with torch.cuda.stream(forward_stream):
                forward_stream.wait_stream(schedule_stream)
                torch.cuda._sleep(self.SLEEP_CYCLES)
                seen[batch] = row.clone()
                read_done = torch.cuda.Event()
                read_done.record()
                # The rest of the forward, after its shared reads.
                torch.cuda._sleep(self.SLEEP_CYCLES)
                done = torch.cuda.Event()
                done.record()
            return read_done, done

        stage = _PPStage(
            schedule_stream=schedule_stream,
            forward_stream=forward_stream,
            launch_forward=launch_forward,
            write_row=row.fill_,
            reads_end_early=reads_end_early,
        )
        stage.scheduler._war_barrier_enabled = barrier
        with torch.cuda.stream(schedule_stream):
            stage.run()
        torch.cuda.synchronize()
        return set(seen["A, chunk 1"].unique().tolist())

    def test_running_chunk_reads_the_old_owner(self):
        for reads_end_early in (True, False):
            with self.subTest(reads_end_early=reads_end_early):
                self.assertEqual(
                    self._seen_by_running_chunk(
                        barrier=True, reads_end_early=reads_end_early
                    ),
                    {OLD_OWNER},
                )

    def test_without_the_barrier_the_new_owner_overtakes_the_read(self):
        """The stand-in forward does race: without the barrier the reuse write
        lands before the running chunk reads the row."""
        self.assertIn(
            NEW_OWNER,
            self._seen_by_running_chunk(barrier=False, reads_end_early=True),
        )


if __name__ == "__main__":
    unittest.main()
