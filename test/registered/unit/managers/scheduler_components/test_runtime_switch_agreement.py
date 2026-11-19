"""Unit tests for srt/managers/scheduler_components/runtime_switch_agreement.py."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace

from sglang.srt.environ import envs, nv_mega_runtime_switches
from sglang.srt.managers.scheduler_components import runtime_switch_agreement
from sglang.srt.managers.scheduler_components.runtime_switch_agreement import (
    agree_runtime_switches,
    resolved_runtime_switches,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MASTER = envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION
_CHILD = envs.SGLANG_OPT_SCHEDULER_SHM_SYNC


def _rank_env(values: dict) -> ExitStack:
    """Every switch unset except this rank's explicit values."""
    stack = ExitStack()
    for field in nv_mega_runtime_switches():
        stack.enter_context(field.override(False))
        field.clear()
    for field, value in values.items():
        stack.enter_context(field.override(value))
    return stack


class TestRuntimeSwitchAgreement(CustomTestCase):
    def _agree(self, *, rank: int, ranks: list) -> None:
        """Run the check as `rank`; ranks[r] is the environment of rank r."""
        gathered = []
        for values in ranks:
            with _rank_env(values):
                gathered.append(resolved_runtime_switches())
        group = SimpleNamespace(
            world_size=len(ranks),
            rank_in_group=rank,
            all_gather_object=lambda obj: (
                gathered[:rank] + [obj] + gathered[rank + 1 :]
            ),
        )
        with _rank_env(ranks[rank]):
            agree_runtime_switches(group=group)

    def _failure_lines(self, *, rank: int, ranks: list) -> list:
        with self.assertRaises(RuntimeError) as ctx:
            self._agree(rank=rank, ranks=ranks)
        return str(ctx.exception).splitlines()[1:]

    def test_child_off_on_one_rank_fails_every_rank_naming_only_that_child(self):
        """A rank that skipped the startup error would wait in a collective forever."""
        ranks = [{_MASTER: True}, {_MASTER: True, _CHILD: "0"}]
        for rank in range(len(ranks)):
            with self.subTest(rank=rank):
                lines = self._failure_lines(rank=rank, ranks=ranks)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].strip().startswith(_CHILD.name + ":"))

    def test_master_missing_on_one_rank_is_named_first(self):
        # Unset children follow the master, so each of them differs too.
        lines = self._failure_lines(rank=1, ranks=[{_MASTER: True}, {}])
        self.assertTrue(lines[0].strip().startswith(_MASTER.name + ":"))
        self.assertEqual(len(lines), len(nv_mega_runtime_switches()))

    def test_agreed_set_is_logged_once_and_only_when_a_switch_is_on(self):
        ranks = [{_MASTER: True, _CHILD: "0"}] * 2
        with self.assertLogs(runtime_switch_agreement.logger, level="INFO") as logs:
            self._agree(rank=0, ranks=ranks)
        (line,) = logs.output
        names = set(line.rsplit(": ", 1)[1].split(", "))
        self.assertIn(_MASTER.name, names)
        self.assertNotIn(_CHILD.name, names)
        with self.assertNoLogs(runtime_switch_agreement.logger, level="INFO"):
            self._agree(rank=1, ranks=ranks)
            self._agree(rank=0, ranks=[{}, {}])


if __name__ == "__main__":
    unittest.main()
