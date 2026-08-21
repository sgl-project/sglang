"""Warmup execution on entrypoints that host no HTTP startup request.

`server` mode must still warm up offline, and a latency may be reported as
warmup-excluded only when some executor actually ran.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.server_warmup import (
    run_sync_client_warmup,
    should_run_local_synthetic_warmup,
    warmup_runs_before_first_request,
)


def _make_server_args(
    *,
    warmup_mode: str,
    warmup_resolutions: list[str] | None,
    task_type: ModelTaskType = ModelTaskType.T2I,
) -> MagicMock:
    server_args = MagicMock()
    server_args.warmup_mode = warmup_mode
    server_args.warmup_resolutions = warmup_resolutions
    server_args.warmup_steps = 1
    server_args.enable_cfg_parallel = False
    server_args.enable_torch_compile = False
    server_args.enable_breakable_cuda_graph = False
    server_args.backend = "native"
    server_args.pipeline_class_name = None
    server_args.is_arg_explicitly_set.return_value = False
    server_args.pipeline_config = SimpleNamespace(
        task_type=task_type,
        vae_stride=None,
        vae_scale_factor=None,
        vae_config=None,
    )
    return server_args


def _not_realtime():
    return patch(
        "sglang.multimodal_gen.runtime.server_warmup.is_realtime_serving",
        return_value=False,
    )


class TestLocalSyntheticWarmup(unittest.TestCase):
    """Which offline configurations run the synthetic warmup themselves."""

    def test_server_mode_without_resolutions_warms_up_locally(self):
        # Regression guard: this is the configuration `_adjust_warmup` picks
        # for --enable-torch-compile, and nothing used to execute it offline.
        server_args = _make_server_args(warmup_mode="server", warmup_resolutions=None)
        with _not_realtime():
            self.assertTrue(should_run_local_synthetic_warmup(server_args))

    def test_explicit_resolutions_warm_up_locally(self):
        server_args = _make_server_args(
            warmup_mode="request", warmup_resolutions=["512x512"]
        )
        with _not_realtime():
            self.assertTrue(should_run_local_synthetic_warmup(server_args))

    def test_request_mode_defers_to_the_scheduler(self):
        # The scheduler clones the first real request instead, so the client
        # must not also inject a synthetic warmup.
        server_args = _make_server_args(warmup_mode="request", warmup_resolutions=None)
        with _not_realtime():
            self.assertFalse(should_run_local_synthetic_warmup(server_args))

    def test_warmup_off_runs_nothing(self):
        server_args = _make_server_args(warmup_mode="off", warmup_resolutions=None)
        with _not_realtime():
            self.assertFalse(should_run_local_synthetic_warmup(server_args))

    def test_action_pipeline_has_no_synthetic_warmup(self):
        server_args = _make_server_args(
            warmup_mode="server",
            warmup_resolutions=None,
            task_type=ModelTaskType.VLA_ACTION,
        )
        with _not_realtime():
            self.assertFalse(should_run_local_synthetic_warmup(server_args))


class TestWarmupRunsBeforeFirstRequest(unittest.TestCase):
    """The predicate guarding the `(with warmup excluded)` claim."""

    def test_true_for_scheduler_request_based_warmup(self):
        server_args = _make_server_args(warmup_mode="request", warmup_resolutions=None)
        with _not_realtime():
            self.assertTrue(warmup_runs_before_first_request(server_args))

    def test_true_for_local_synthetic_warmup(self):
        server_args = _make_server_args(warmup_mode="server", warmup_resolutions=None)
        with _not_realtime():
            self.assertTrue(warmup_runs_before_first_request(server_args))

    def test_false_when_warmup_is_off(self):
        server_args = _make_server_args(warmup_mode="off", warmup_resolutions=None)
        with _not_realtime():
            self.assertFalse(warmup_runs_before_first_request(server_args))

    def test_false_when_no_executor_can_run(self):
        # A non-visual task supports no synthetic warmup, and `server` mode
        # never reaches the scheduler's request-based clone. Reporting the
        # first request as warmup-excluded here would be a lie.
        server_args = _make_server_args(
            warmup_mode="server",
            warmup_resolutions=None,
            task_type=ModelTaskType.VLA_ACTION,
        )
        with _not_realtime():
            self.assertFalse(warmup_runs_before_first_request(server_args))


class TestSyncClientWarmupFailurePolicy(unittest.TestCase):
    """Auto-selected warmup fails open; an explicitly requested one does not."""

    def _run(self, *, fail_open: bool):
        server_args = _make_server_args(warmup_mode="server", warmup_resolutions=None)
        forward = MagicMock(side_effect=RuntimeError("warmup forward exploded"))
        with patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=SamplingParams(width=512, height=512),
        ):
            run_sync_client_warmup(server_args, forward, fail_open=fail_open)

    def test_fail_open_swallows_warmup_failure(self):
        self._run(fail_open=True)

    def test_fail_closed_propagates_warmup_failure(self):
        with self.assertRaises(RuntimeError):
            self._run(fail_open=False)


if __name__ == "__main__":
    unittest.main()
