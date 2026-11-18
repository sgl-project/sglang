"""The per-forward join tick: which pending states it acts on and which it leaves to
other paths."""

import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

SCALE_TIMEOUT_S = 1


def _tick(*, effective, pending, backend="mooncake", active=None, overdue=False):
    """One maybe_join_ep_ranks call; returns (fail_scale mock, recover mock)."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(elastic_ep_scale_timeout=SCALE_TIMEOUT_S)
    runner._fail_scale = mock.Mock()
    runner._finalize_recovered_ep_ranks = mock.Mock()
    active = [1] * effective if active is None else active
    state = SimpleNamespace(
        has_scaled=False,
        active_ranks_cpu=torch.tensor(active, dtype=torch.int32),
        pending_since=time.monotonic() - 10 * SCALE_TIMEOUT_S if overdue else None,
        pending_recover_ranks=[],
        # A phase the tick has no branch for, so the test stops at what it checks.
        scale_phase="joining",
    )
    manager = mock.Mock()
    manager.is_scaling.return_value = True
    manager.instance.return_value = state
    manager.get_effective_ep_size.return_value = effective
    manager.get_pending_ep_size.return_value = pending
    manager.begin_scale.return_value = False
    exec_cfg = SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=backend))
    with ExitStack() as stack:
        for name, value in {
            "ElasticEPStateManager": manager,
            "get_exec": lambda: exec_cfg,
            "current_platform": mock.Mock(),
        }.items():
            stack.enter_context(mock.patch.object(model_runner_module, name, value))
        recover = stack.enter_context(
            mock.patch.object(
                model_runner_module, "try_recover_faulted_ranks", return_value=False
            )
        )
        runner.maybe_join_ep_ranks()
    return runner._fail_scale, recover


class TestFaultRecoveryBackend(unittest.TestCase):
    def test_nixl_only_reports_the_fault(self):
        """Recovery waits on Mooncake peer state, which nixl never reports."""
        _, recover = _tick(
            effective=4, pending=None, backend="nixl", active=[1, 0, 1, 1]
        )
        recover.assert_not_called()

    def test_mooncake_recovers_the_faulted_rank(self):
        _, recover = _tick(effective=4, pending=None, active=[1, 0, 1, 1])
        recover.assert_called_once_with([1])


class TestShrinkTimeout(unittest.TestCase):
    def test_an_overdue_shrink_is_left_to_its_own_path(self):
        """Failing it here would skip the abandon that releases peers' barriers."""
        fail_scale, _ = _tick(effective=4, pending=2, overdue=True)
        fail_scale.assert_not_called()

    def test_an_overdue_grow_still_fails(self):
        fail_scale, _ = _tick(effective=2, pending=4, overdue=True)
        fail_scale.assert_called_once()
        self.assertIn("target EP size 4", fail_scale.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
