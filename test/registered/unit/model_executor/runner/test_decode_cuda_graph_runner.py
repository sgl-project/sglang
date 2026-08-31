"""Unit tests for ``DecodeCudaGraphRunner`` capture-phase profiling — CPU-only.

Two capture-trace modes plus their precedence:

  * **Original single-trace** (``SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE``):
    ``_init_profile_context_and_memory_record`` builds an *unscheduled* profiler
    (``record_shapes`` only, no schedule / no ``on_trace_ready``); the combined
    trace is exported in ``_post_process_after_profile`` via
    ``export_cuda_graph_capture_trace``.
  * **Per-batch-size traces** (``SGLANG_GRAPH_BATCH_CAPTURE``): a *scheduled*
    profiler (``wait=2, warmup=0, active=1, repeat=0``) with the trace-export
    knobs (record_shapes / with_stack / with_flops / profile_memory) and an
    ``on_trace_ready`` hook that writes one trace per batch size to
    ``<SGLANG_TORCH_PROFILER_DIR>/graph_capture_profile/`` named
    ``{runner_name}_bs_{bs}_rank{rank}.json.gz``.
  * **Precedence**: when both env vars are set, the original single-trace path
    wins (no per-bs schedule / dir / bookkeeping).

The profiler / CUDA-memory APIs are mocked; the directory + naming + schedule
logic is pure-Python and runs on CPU. The method is invoked unbound against a
lightweight stand-in (with the real precedence helper bound) so no model or
server is constructed.
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
from sglang.srt.utils import profile_utils as putils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_CAPTURE_TRACE = "SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE"
_BATCH_CAPTURE = "SGLANG_GRAPH_BATCH_CAPTURE"


def _make_fake_self(capture_bs):
    """Stand-in ``self`` with the real precedence helper bound so the env-var
    gating in ``_init_profile_context_and_memory_record`` applies."""
    fake_self = SimpleNamespace(capture_bs=list(capture_bs))
    fake_self._graph_batch_capture_active = (
        DecodeCudaGraphRunner._graph_batch_capture_active.__get__(fake_self)
    )
    return fake_self


class TestInitProfileBatchMode(CustomTestCase):
    """SGLANG_GRAPH_BATCH_CAPTURE -> scheduled per-bs profiler."""

    def _invoke(self, *, capture_bs, rank=0, profiler_dir=None):
        fake_self = _make_fake_self(capture_bs)
        env = {_BATCH_CAPTURE: "1"}
        if profiler_dir is not None:
            env["SGLANG_TORCH_PROFILER_DIR"] = profiler_dir
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(tp_rank=rank)
        ), mock.patch.object(mod, "profile") as mock_profile, mock.patch(
            "torch.profiler.schedule"
        ) as mock_schedule, mock.patch(
            "torch.cuda.memory._record_memory_history"
        ) as mock_record_history:
            os.environ.pop(_CAPTURE_TRACE, None)  # original flag off
            if profiler_dir is None:
                os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            ctx = DecodeCudaGraphRunner._init_profile_context_and_memory_record(
                fake_self
            )
        self.assertIs(ctx, mock_profile.return_value)
        return fake_self, mock_profile, mock_schedule, mock_record_history

    def test_creates_graph_capture_profile_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._invoke(capture_bs=[1, 2, 4], profiler_dir=tmp)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "graph_capture_profile")))

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

    def test_default_dir_used_when_profiler_dir_env_unset(self):
        # No SGLANG_TORCH_PROFILER_DIR -> falls back to the envs default base dir.
        # Patch makedirs so the test never writes to the cwd.
        fake_self = _make_fake_self([1])
        with mock.patch.dict(
            os.environ, {_BATCH_CAPTURE: "1"}, clear=False
        ), mock.patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ), mock.patch.object(
            mod, "profile"
        ), mock.patch(
            "torch.profiler.schedule"
        ), mock.patch(
            "torch.cuda.memory._record_memory_history"
        ), mock.patch.object(
            mod.os, "makedirs"
        ) as mock_makedirs:
            os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            os.environ.pop(_CAPTURE_TRACE, None)
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)

        mock_makedirs.assert_called_once()
        self.assertEqual(
            mock_makedirs.call_args.args[0],
            os.path.join("/tmp", "graph_capture_profile"),
        )


class TestInitProfileOriginalMode(CustomTestCase):
    """No flag, original flag only, or both (precedence) -> unscheduled pass with
    no per-bs schedule / directory / bookkeeping."""

    def _invoke_original(self, *, env):
        fake_self = _make_fake_self([1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            environ = dict(env)
            environ["SGLANG_TORCH_PROFILER_DIR"] = tmp
            with mock.patch.dict(os.environ, environ, clear=False), mock.patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
            ), mock.patch.object(mod, "profile") as mock_profile, mock.patch(
                "torch.profiler.schedule"
            ) as mock_schedule, mock.patch(
                "torch.cuda.memory._record_memory_history"
            ):
                for k in (_CAPTURE_TRACE, _BATCH_CAPTURE):
                    if k not in environ:
                        os.environ.pop(k, None)
                DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
            kwargs = mock_profile.call_args.kwargs
            # Unscheduled pass: record_shapes only, no schedule / on_trace_ready.
            self.assertTrue(kwargs["record_shapes"])
            self.assertIsNone(kwargs.get("schedule"))
            self.assertIsNone(kwargs.get("on_trace_ready"))
            mock_schedule.assert_not_called()
            self.assertFalse(os.path.isdir(os.path.join(tmp, "graph_capture_profile")))
            self.assertFalse(hasattr(fake_self, "_profile_bs_list"))

    def test_no_flags(self):
        self._invoke_original(env={})

    def test_original_flag_only(self):
        self._invoke_original(env={_CAPTURE_TRACE: "1"})

    def test_both_flags_original_takes_precedence(self):
        self._invoke_original(env={_CAPTURE_TRACE: "1", _BATCH_CAPTURE: "1"})


class TestOnTraceReadyNaming(CustomTestCase):
    def _build_on_trace_ready(self, *, capture_bs, rank, tmp):
        fake_self = _make_fake_self(capture_bs)
        with mock.patch.dict(
            os.environ,
            {"SGLANG_TORCH_PROFILER_DIR": tmp, _BATCH_CAPTURE: "1"},
            clear=False,
        ), mock.patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(tp_rank=rank)
        ), mock.patch.object(
            mod, "profile"
        ) as mock_profile, mock.patch(
            "torch.profiler.schedule"
        ), mock.patch(
            "torch.cuda.memory._record_memory_history"
        ):
            os.environ.pop(_CAPTURE_TRACE, None)
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
        on_trace_ready = mock_profile.call_args.kwargs["on_trace_ready"]
        return fake_self, on_trace_ready

    def test_exports_one_named_trace_per_bs_and_advances_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_bs = [1, 2, 4]  # reversed -> [4, 2, 1]
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=capture_bs, rank=0, tmp=tmp
            )
            trace_dir = os.path.join(tmp, "graph_capture_profile")
            runner = type(fake_self).__name__

            exported = []
            for expected_bs in [4, 2, 1]:
                prof = mock.Mock()
                prof.export_chrome_trace.side_effect = lambda p: exported.append(p)
                on_trace_ready(prof)
                prof.export_chrome_trace.assert_called_once_with(
                    os.path.join(trace_dir, f"{runner}_bs_{expected_bs}_rank0.json.gz")
                )

            self.assertEqual(
                exported,
                [
                    os.path.join(trace_dir, f"{runner}_bs_4_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_2_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_1_rank0.json.gz"),
                ],
            )
            # Index advanced once per flush.
            self.assertEqual(fake_self._profile_bs_idx, 3)

    def test_rank_in_trace_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[8], rank=3, tmp=tmp
            )
            runner = type(fake_self).__name__
            prof = mock.Mock()
            on_trace_ready(prof)
            prof.export_chrome_trace.assert_called_once_with(
                os.path.join(
                    tmp, "graph_capture_profile", f"{runner}_bs_8_rank3.json.gz"
                )
            )


class TestOriginalTraceExport(CustomTestCase):
    """export_cuda_graph_capture_trace (original single combined trace per rank),
    gated by SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE, and the shared dir helper.
    Both trace modes land under graph_capture_profile/."""

    def test_writes_named_trace_when_flag_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"SGLANG_TORCH_PROFILER_DIR": tmp, _CAPTURE_TRACE: "1"},
                clear=False,
            ):
                prof = mock.Mock()
                putils.export_cuda_graph_capture_trace(
                    prof, runner_name="DecodeCudaGraphRunner", tp_rank=2
                )
                expected = os.path.join(
                    tmp,
                    "graph_capture_profile",
                    "cuda_graph_capture-DecodeCudaGraphRunner-TP-2.json.gz",
                )
                prof.export_chrome_trace.assert_called_once_with(expected)
                self.assertTrue(
                    os.path.isdir(os.path.join(tmp, "graph_capture_profile"))
                )

    def test_noop_when_flag_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmp}, clear=False
            ):
                os.environ.pop(_CAPTURE_TRACE, None)
                prof = mock.Mock()
                putils.export_cuda_graph_capture_trace(
                    prof, runner_name="DecodeCudaGraphRunner", tp_rank=0
                )
                prof.export_chrome_trace.assert_not_called()
                self.assertFalse(
                    os.path.isdir(os.path.join(tmp, "graph_capture_profile"))
                )

    def test_dir_helper_uses_profiler_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmp}, clear=False
            ):
                self.assertEqual(
                    putils.graph_capture_profile_dir(),
                    os.path.join(tmp, "graph_capture_profile"),
                )


if __name__ == "__main__":
    unittest.main()
