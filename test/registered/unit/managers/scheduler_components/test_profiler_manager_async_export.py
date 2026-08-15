"""The scheduler thread must not write profiler traces itself.

Exporting a chrome trace costs seconds per profiled second, so doing it inline in
_stop_profile() stalls the whole rank: requests in flight wait for it and health
checks in the window fail. These tests pin the trace write to the flush thread,
and pin what has to stay on the scheduler thread with it -- the barrier, so all
ranks keep issuing group collectives in the same order.
"""

import gzip
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import wait
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.managers.scheduler_components import profiler_manager
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeTorchProfiler:
    """Stands in for torch.profiler.profile, with an export we control."""

    def __init__(self, index=0):
        self.index = index
        self.stopped = False
        self.exported = []
        self.raise_on_export = None
        self.entered_export = threading.Event()
        self.release = threading.Event()
        self.release.set()  # a test that wants a slow export clears this

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        self.entered_export.set()
        self.release.wait(timeout=60)
        if self.raise_on_export is not None:
            raise self.raise_on_export
        self.exported.append(path)
        with open(path, "w") as f:
            json.dump({"traceEvents": [], "sglang_test_profiler": self.index}, f)


class FakeProfilerFactory:
    def __init__(self):
        self.instances = []

    def __call__(self, **kwargs):
        self.instances.append(FakeTorchProfiler(len(self.instances)))
        return self.instances[-1]


class FakePS:
    tp_rank = dp_rank = pp_rank = moe_ep_rank = 0
    dp_size = pp_size = moe_ep_size = 1
    gpu_id = 0


class TestProfilerManagerAsyncExport(CustomTestCase):
    def setUp(self):
        self.factory = FakeProfilerFactory()
        self.managers = []

        profile_patcher = patch("torch.profiler.profile", self.factory)
        profile_patcher.start()
        self.addCleanup(profile_patcher.stop)

        barrier_patcher = patch("torch.distributed.barrier")
        self.barrier = barrier_patcher.start()
        self.addCleanup(barrier_patcher.stop)

        # Never leave a blocked export behind: the flush thread is not a daemon,
        # so one would keep the test process alive after a failed assertion.
        self.addCleanup(self._release_everything)

    def _release_everything(self):
        for profiler in self.factory.instances:
            profiler.release.set()
        for mgr in self.managers:
            if mgr._flush_executor is not None:
                mgr._flush_executor.shutdown(wait=True)

    def _make_manager(self, output_dir, **init_kwargs):
        responses = []
        mgr = SchedulerProfilerManager(
            ps=FakePS(),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
            send_response=lambda *, output, recv_req: responses.append(
                (output, recv_req)
            ),
        )
        self.managers.append(mgr)
        init = dict(
            output_dir=output_dir,
            start_step=None,
            num_steps=None,
            activities=["CPU"],
            with_stack=False,
            record_shapes=False,
            profile_by_stage=False,
            profile_id="test-id",
        )
        init.update(init_kwargs)
        self.assertTrue(mgr._init_profile(**init).success)
        return mgr, responses

    def _assert_trace_written(self, directory, name):
        """The trace on disk reads back as gzip, and no intermediate file is left."""
        path = os.path.join(directory, name)
        with gzip.open(path, "rt") as f:
            json.load(f)
        self.assertFalse(os.path.exists(path[: -len(".gz")]))

    def _published_by(self, directory, name):
        """The profiler whose export is the trace now under this name."""
        with gzip.open(os.path.join(directory, name), "rt") as f:
            return json.load(f)["sglang_test_profiler"]

    def _gzip_stored_name(self, path):
        """The name stored in the gzip header, which readers offer as a filename."""
        with open(path, "rb") as f:
            header = f.read(10)
            self.assertEqual(header[:2], b"\x1f\x8b")
            self.assertTrue(header[3] & 0x08, "the archive stores no name")
            name = b""
            while (char := f.read(1)) not in (b"\x00", b""):
                name += char
        return name.decode()

    def _drain(self, mgr):
        """Let every in-flight write finish, then run one scheduler-loop poll."""
        wait([pending.export for pending in mgr._pending_flushes], timeout=60)
        mgr.check_pending_flush()

    def test_stop_profile_returns_before_the_trace_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            profiler = self.factory.instances[-1]
            profiler.release.clear()

            recv_req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            started = time.perf_counter()
            output = mgr._stop_profile(reply_to=recv_req)
            elapsed = time.perf_counter() - started

            self.assertTrue(profiler.stopped)
            self.assertTrue(profiler.entered_export.wait(timeout=10))
            self.assertLess(elapsed, 5.0)  # i.e. it did not wait for the export
            self.assertFalse(mgr.profile_in_progress)
            # The barrier stays on the scheduler thread, where every rank reaches
            # it at the same point in the group's sequence of collectives.
            self.assertEqual(self.barrier.call_count, 1)

            # Nobody is answered until the trace is actually on disk.
            self.assertIsNone(output)
            mgr.check_pending_flush()
            self.assertEqual(responses, [])

            profiler.release.set()
            self._drain(mgr)
            self.assertEqual(len(responses), 1)
            output, answered_req = responses[0]
            self.assertTrue(output.success, output.message)
            self.assertIs(answered_req, recv_req)
            self.assertEqual(len(mgr._pending_flushes), 0)
            self.assertEqual(len(profiler.exported), 1)
            self._assert_trace_written(tmp, "test-id-TP-0.trace.json.gz")

    def test_detailed_annotations_are_off_before_the_deferred_return(self):
        """The step-span flag is process-wide, so the deferred reply cannot own it.

        `_stop_profile` now answers later than it used to. If the reset rode along
        with that reply, every forward after a profile would keep emitting the
        detailed spans until some later profile turned them off.
        """
        from sglang.srt.model_executor.step_span_utils import (
            detailed_annotations_enabled,
            set_detailed_annotations_enabled,
        )

        self.addCleanup(set_detailed_annotations_enabled, False)
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = self._make_manager(tmp, detailed_annotations=True)
            self.assertTrue(mgr._start_profile().success)
            self.assertTrue(detailed_annotations_enabled())
            profiler = self.factory.instances[-1]
            profiler.release.clear()

            recv_req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            self.assertIsNone(mgr._stop_profile(reply_to=recv_req))
            self.assertFalse(detailed_annotations_enabled())

            profiler.release.set()
            self._drain(mgr)

    def test_stop_without_a_request_answers_nobody(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            self.assertIsNone(mgr._stop_profile())
            self._drain(mgr)
            self.assertEqual(responses, [])

    def test_profiling_restarts_while_the_previous_trace_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            first = self.factory.instances[-1]
            first.release.clear()
            self.assertIsNone(mgr._stop_profile())

            self.assertTrue(
                mgr._init_profile(
                    output_dir=tmp,
                    start_step=None,
                    num_steps=None,
                    activities=["CPU"],
                    with_stack=False,
                    record_shapes=False,
                    profile_by_stage=False,
                    profile_id="second-id",
                ).success
            )
            self.assertTrue(mgr._start_profile().success)
            second = self.factory.instances[-1]
            self.assertIsNot(second, first)
            self.assertTrue(mgr.profile_in_progress)
            self.assertIsNone(mgr._stop_profile())

            # One writer, so the second trace waits its turn behind the first.
            self.assertEqual(len(mgr._pending_flushes), 2)
            self.assertFalse(second.entered_export.is_set())
            first.release.set()
            self._drain(mgr)
            self.assertEqual(len(mgr._pending_flushes), 0)
            self.assertEqual(len(first.exported), 1)
            self._assert_trace_written(tmp, "test-id-TP-0.trace.json.gz")
            self._assert_trace_written(tmp, "second-id-TP-0.trace.json.gz")

    def test_by_stage_flush_at_the_decode_boundary_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp, profile_by_stage=True, num_steps=4)
            mgr._profile_batch_predicate(
                SimpleNamespace(forward_mode=ForwardMode.EXTEND)
            )
            prefill = self.factory.instances[-1]
            prefill.release.clear()

            # The prefill trace is flushed here, between two batches, which is
            # where a synchronous export freezes a live serving loop.
            started = time.perf_counter()
            mgr._profile_batch_predicate(
                SimpleNamespace(forward_mode=ForwardMode.DECODE)
            )
            elapsed = time.perf_counter() - started

            self.assertTrue(prefill.stopped)
            self.assertTrue(prefill.entered_export.wait(timeout=10))
            self.assertLess(elapsed, 5.0)
            self.assertEqual(len(mgr._pending_flushes), 1)
            # The decode stage is already recording while that write goes on.
            self.assertTrue(mgr.profile_in_progress)
            self.assertIsNot(self.factory.instances[-1], prefill)

            prefill.release.set()
            self._drain(mgr)
            self.assertEqual(responses, [])
            self._assert_trace_written(tmp, "test-id-TP-0-EXTEND.trace.json.gz")

    def test_export_failure_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            self.factory.instances[-1].raise_on_export = RuntimeError("no space left")

            recv_req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            self.assertIsNone(mgr._stop_profile(reply_to=recv_req))
            self._drain(mgr)

            self.assertEqual(len(responses), 1)
            output, answered_req = responses[0]
            self.assertFalse(output.success)
            self.assertIn("no space left", output.message)
            self.assertIs(answered_req, recv_req)

    def test_merge_profiles_keeps_the_flush_synchronous(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp, merge_profiles=True)
            self.assertTrue(mgr._start_profile().success)
            profiler = self.factory.instances[-1]

            recv_req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            output = mgr._stop_profile(reply_to=recv_req)

            # The merge reads every rank's trace, so this path answers inline and
            # only after its own export is done.
            self.assertIsNotNone(output)
            self.assertTrue(output.success, output.message)
            self.assertEqual(len(profiler.exported), 1)
            self._assert_trace_written(tmp, "test-id-TP-0.trace.json.gz")
            self.assertEqual(len(mgr._pending_flushes), 0)
            self.assertEqual(responses, [])

    def test_shutdown_answers_a_stop_whose_trace_is_still_being_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            recv_req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            self.assertIsNone(mgr._stop_profile(reply_to=recv_req))

            # The scheduler loop breaks on shutdown without polling again, so the
            # teardown path has to deliver this one itself.
            mgr.drain_pending_flushes()

            self.assertEqual(len(responses), 1)
            output, answered_req = responses[0]
            self.assertTrue(output.success, output.message)
            self.assertIs(answered_req, recv_req)
            self.assertEqual(len(mgr._pending_flushes), 0)
            self._assert_trace_written(tmp, "test-id-TP-0.trace.json.gz")

    def test_a_stop_after_shutdown_still_writes_its_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            self.assertIsNone(mgr._stop_profile())  # starts the flush thread
            mgr.drain_pending_flushes()  # and takes it away again

            self.assertTrue(
                mgr._init_profile(
                    output_dir=tmp,
                    start_step=None,
                    num_steps=None,
                    activities=["CPU"],
                    with_stack=False,
                    record_shapes=False,
                    profile_by_stage=False,
                    profile_id="second-id",
                ).success
            )
            self.assertTrue(mgr._start_profile().success)

            # Nothing left to drain a deferred flush, so this answers inline as
            # every stop used to.
            output = mgr._stop_profile(
                reply_to=ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            )
            self.assertIsNotNone(output)
            self.assertTrue(output.success, output.message)
            self.assertEqual(responses, [])
            self.assertEqual(len(mgr._pending_flushes), 0)
            self._assert_trace_written(tmp, "test-id-TP-0.trace.json.gz")
            self._assert_trace_written(tmp, "second-id-TP-0.trace.json.gz")

    def test_an_uncompressed_trace_in_the_output_dir_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Somebody's earlier uncompressed export of this same profile: the
            # compression step must not write through it, let alone delete it.
            bystander = os.path.join(tmp, "test-id-TP-0.trace.json")
            with open(bystander, "w") as f:
                f.write('{"traceEvents": ["keep me"]}')

            mgr, _ = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            self.assertIsNone(mgr._stop_profile())
            self._drain(mgr)

            with open(bystander) as f:
                self.assertEqual(f.read(), '{"traceEvents": ["keep me"]}')

    def test_an_export_that_fails_before_writing_leaves_nothing_behind(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as td:
            with patch.object(tempfile, "tempdir", td):
                mgr, responses = self._make_manager(tmp)
                self.assertTrue(mgr._start_profile().success)
                self.factory.instances[-1].raise_on_export = RuntimeError(
                    "no space left"
                )
                self.assertIsNone(mgr._stop_profile())
                self._drain(mgr)

            # No trace at the path the profile advertises, and the json that was
            # staged for it is gone as well. A failure part-way through the copy is
            # not covered, and needs no cover: it leaves a partial archive, which is
            # what torch leaves too, gzipping straight to the destination.
            self.assertEqual(os.listdir(tmp), [])
            self.assertEqual(os.listdir(td), [])

    def test_the_archive_names_the_trace_and_not_the_staged_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            self.assertIsNone(mgr._stop_profile())
            self._drain(mgr)

            name = "test-id-TP-0.trace.json.gz"
            path = os.path.join(tmp, name)
            # The archive names the trace, not the temporary file that carried it.
            self.assertEqual(self._gzip_stored_name(path), name[: -len(".gz")])

    def test_on_mps_the_profiler_is_handed_the_advertised_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test-id-TP-0.trace.json.gz")
            profiler = FakeTorchProfiler()
            # A Metal trace is saved alongside this one under a name built from
            # the path the profiler is given, so a temporary path renames it too.
            with patch.object(profiler_manager, "_is_mps", True):
                profiler_manager._export_trace(profiler, path)

            self.assertEqual(profiler.exported, [path])

    def test_a_long_profile_id_still_writes_its_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A profile id can bring the name of the trace to within a few bytes of
            # the limit a filesystem puts on one -- and the limit counts bytes, not
            # characters -- so nothing may derive a longer name from it.
            profile_id = "😀" * 56
            mgr, _ = self._make_manager(tmp, profile_id=profile_id)
            self.assertTrue(mgr._start_profile().success)
            self.assertIsNone(mgr._stop_profile())
            self._drain(mgr)

            self._assert_trace_written(tmp, f"{profile_id}-TP-0.trace.json.gz")

    def test_the_uncompressed_trace_stays_out_of_the_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as td:
            # The json is an order of magnitude larger than the archive, so it
            # goes where torch puts it and not onto the volume the traces are
            # collected on.
            with patch.object(tempfile, "tempdir", td):
                mgr, _ = self._make_manager(tmp)
                self.assertTrue(mgr._start_profile().success)
                profiler = self.factory.instances[-1]
                self.assertIsNone(mgr._stop_profile())
                self._drain(mgr)

            self.assertEqual(os.listdir(tmp), ["test-id-TP-0.trace.json.gz"])
            self.assertTrue(profiler.exported[0].startswith(td), profiler.exported[0])
            self.assertEqual(os.listdir(td), [])

    def test_a_merge_waits_for_a_trace_that_is_still_being_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = self._make_manager(tmp)
            self.assertTrue(mgr._start_profile().success)
            recorded = self.factory.instances[-1]
            recorded.release.clear()
            self.assertIsNone(mgr._stop_profile())

            # A merge reads every trace whose name starts with the profile id, so
            # one this rank recorded earlier under that id has to be written
            # first: both publish to the same name, and the merge would otherwise
            # read whichever landed last.
            self.assertTrue(
                mgr._init_profile(
                    output_dir=tmp,
                    start_step=None,
                    num_steps=None,
                    activities=["CPU"],
                    with_stack=False,
                    record_shapes=False,
                    profile_by_stage=False,
                    profile_id="test-id",
                    merge_profiles=True,
                ).success
            )
            self.assertTrue(mgr._start_profile().success)
            merged = self.factory.instances[-1]
            threading.Timer(0.2, recorded.release.set).start()
            output = mgr._stop_profile()

            self.assertIsNotNone(output)
            self.assertEqual(len(mgr._pending_flushes), 0)
            self._drain(mgr)
            name = "test-id-TP-0.trace.json.gz"
            self._assert_trace_written(tmp, name)
            self.assertEqual(self._published_by(tmp, name), merged.index)

    def test_a_stop_request_is_answered_once_its_trace_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, responses = self._make_manager(tmp)
            start = ProfileReq(
                req_type=ProfileReqType.START_PROFILE,
                output_dir=tmp,
                activities=["CPU"],
                profile_id="test-id",
            )
            self.assertTrue(mgr._profile(start).success)

            # The request path, which is what a /stop_profile actually reaches.
            stop = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            self.assertIsNone(mgr._profile(stop))
            self.assertEqual(responses, [])
            self._drain(mgr)

            self.assertEqual(len(responses), 1)
            output, answered_req = responses[0]
            self.assertTrue(output.success, output.message)
            self.assertIs(answered_req, stop)

    def test_recorded_traces_do_not_pile_up_while_the_writer_is_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, _ = self._make_manager(tmp)
            blocked = []
            for i in range(3):
                if i:
                    self.assertTrue(
                        mgr._init_profile(
                            output_dir=tmp,
                            start_step=None,
                            num_steps=None,
                            activities=["CPU"],
                            with_stack=False,
                            record_shapes=False,
                            profile_by_stage=False,
                            profile_id=f"id-{i}",
                        ).success
                    )
                self.assertTrue(mgr._start_profile().success)
                profiler = self.factory.instances[-1]
                if i == 0:
                    # Hold up the writer, then let it go while the third stop is
                    # waiting for it, which is what bounds the queue.
                    profiler.release.clear()
                    threading.Timer(0.2, profiler.release.set).start()
                blocked.append(profiler)
                mgr._stop_profile()
                self.assertLessEqual(len(mgr._pending_flushes), 2)

            self._drain(mgr)
            self.assertTrue(blocked[0].release.is_set())
            for i in range(3):
                name = "test-id" if i == 0 else f"id-{i}"
                self._assert_trace_written(tmp, f"{name}-TP-0.trace.json.gz")


if __name__ == "__main__":
    unittest.main()
