"""Unit tests for ``DecodeCudaGraphRunner`` capture-phase profiling — CPU-only.

These cover the changes from the "cuda graph profile traces" PR that make the
graph-capture phase export one torch profiler trace per captured batch size
when ``--enable-profile-cuda-graph`` is set:

  * ``_init_profile_context_and_memory_record`` creates the
    ``<SGLANG_TORCH_PROFILER_DIR>/capture_traces`` directory, primes the
    per-bs bookkeeping (``_profile_bs_list`` reversed to match capture order,
    ``_profile_bs_idx`` reset to 0), and builds the profiler with the
    ``wait=2, warmup=0, active=1, repeat=0`` schedule and the trace-export
    knobs (record_shapes / with_stack / with_flops / profile_memory).
  * The ``on_trace_ready`` callback names each trace
    ``bs_{bs}_rank{rank}.json.gz`` against the reversed capture-bs list and
    advances the index on every flush, so successive batch sizes land in
    distinct files.

The profiler / CUDA-memory APIs are mocked; the directory + naming + schedule
logic is pure-Python and runs on CPU. The method is invoked unbound against a
lightweight stand-in so no model or server is constructed.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestInitProfileContext(CustomTestCase):
    def _invoke(self, *, capture_bs, rank=0, profiler_dir=None):
        """Call the unbound method against a stand-in self and return
        (fake_self, mock_profile, mock_schedule, mock_record_history)."""
        fake_self = SimpleNamespace(capture_bs=list(capture_bs))

        env = {}
        if profiler_dir is not None:
            env["SGLANG_TORCH_PROFILER_DIR"] = profiler_dir

        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            mod, "get_tensor_model_parallel_rank", return_value=rank
        ), mock.patch.object(mod, "profile") as mock_profile, mock.patch(
            "torch.profiler.schedule"
        ) as mock_schedule, mock.patch(
            "torch.cuda.memory._record_memory_history"
        ) as mock_record_history:
            if profiler_dir is None:
                os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            ctx = DecodeCudaGraphRunner._init_profile_context_and_memory_record(
                fake_self
            )

        self.assertIs(ctx, mock_profile.return_value)
        return fake_self, mock_profile, mock_schedule, mock_record_history

    def test_creates_capture_traces_dir_under_profiler_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._invoke(capture_bs=[1, 2, 4], profiler_dir=tmp)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "capture_traces")))

    def test_primes_reversed_bs_list_and_zero_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, *_ = self._invoke(capture_bs=[1, 2, 4, 8], profiler_dir=tmp)
            # Capture iterates large -> small, so the bs list is reversed.
            self.assertEqual(fake_self._profile_bs_list, [8, 4, 2, 1])
            self.assertEqual(fake_self._profile_bs_idx, 0)

    def test_profiler_built_with_trace_export_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, mock_profile, mock_schedule, mock_record_history = self._invoke(
                capture_bs=[1, 2], profiler_dir=tmp
            )
            self.assertEqual(mock_profile.call_count, 1)
            kwargs = mock_profile.call_args.kwargs
            self.assertTrue(kwargs["record_shapes"])
            self.assertTrue(kwargs["with_stack"])
            self.assertTrue(kwargs["with_flops"])
            self.assertTrue(kwargs["profile_memory"])
            self.assertTrue(callable(kwargs["on_trace_ready"]))
            # Schedule skips the two dummy/warmup runs and records the capture.
            mock_schedule.assert_called_once_with(wait=2, warmup=0, active=1, repeat=0)
            self.assertIs(kwargs["schedule"], mock_schedule.return_value)
            # Memory history recording is armed alongside the profiler.
            mock_record_history.assert_called_once()

    def test_default_dir_used_when_env_unset(self):
        # No SGLANG_TORCH_PROFILER_DIR -> falls back to the "traces" base dir.
        # Patch makedirs so the test never writes to the cwd.
        fake_self = SimpleNamespace(capture_bs=[1])
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            mod, "get_tensor_model_parallel_rank", return_value=0
        ), mock.patch.object(mod, "profile") as mock_profile, mock.patch(
            "torch.profiler.schedule"
        ), mock.patch(
            "torch.cuda.memory._record_memory_history"
        ), mock.patch.object(
            mod.os, "makedirs"
        ) as mock_makedirs:
            # patch.dict restores the original environ on exit.
            os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)

        mock_makedirs.assert_called_once()
        created_dir = mock_makedirs.call_args.args[0]
        self.assertEqual(created_dir, os.path.join("traces", "capture_traces"))


class TestOnTraceReadyNaming(CustomTestCase):
    def _build_on_trace_ready(self, *, capture_bs, rank, tmp):
        fake_self = SimpleNamespace(capture_bs=list(capture_bs))
        with mock.patch.dict(
            os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmp}, clear=False
        ), mock.patch.object(
            mod, "get_tensor_model_parallel_rank", return_value=rank
        ), mock.patch.object(
            mod, "profile"
        ) as mock_profile, mock.patch(
            "torch.profiler.schedule"
        ), mock.patch(
            "torch.cuda.memory._record_memory_history"
        ):
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
        on_trace_ready = mock_profile.call_args.kwargs["on_trace_ready"]
        return fake_self, on_trace_ready

    def test_exports_one_named_trace_per_bs_and_advances_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_bs = [1, 2, 4]  # reversed -> [4, 2, 1]
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=capture_bs, rank=0, tmp=tmp
            )
            trace_dir = os.path.join(tmp, "capture_traces")

            exported = []
            for expected_bs in [4, 2, 1]:
                prof = mock.Mock()
                prof.export_chrome_trace.side_effect = lambda p: exported.append(p)
                on_trace_ready(prof)
                prof.export_chrome_trace.assert_called_once_with(
                    os.path.join(trace_dir, f"bs_{expected_bs}_rank0.json.gz")
                )

            self.assertEqual(
                exported,
                [
                    os.path.join(trace_dir, "bs_4_rank0.json.gz"),
                    os.path.join(trace_dir, "bs_2_rank0.json.gz"),
                    os.path.join(trace_dir, "bs_1_rank0.json.gz"),
                ],
            )
            # Index advanced once per flush.
            self.assertEqual(fake_self._profile_bs_idx, 3)

    def test_rank_in_trace_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[8], rank=3, tmp=tmp
            )
            prof = mock.Mock()
            on_trace_ready(prof)
            prof.export_chrome_trace.assert_called_once_with(
                os.path.join(tmp, "capture_traces", "bs_8_rank3.json.gz")
            )


if __name__ == "__main__":
    unittest.main()
