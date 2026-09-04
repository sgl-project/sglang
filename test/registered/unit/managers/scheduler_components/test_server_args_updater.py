"""Unit tests for srt/managers/scheduler_components/server_args_updater.

Critical-path bookkeeping of runtime server-args updates: the allowlist gate
(a non-allowlisted field must be rejected), per-field validation boundaries,
DSpark worker-command routing staying out of the config bags, and the
override landing in the bags with provenance.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler_components.server_args_updater import (
    HOT_UPDATABLE_SERVER_ARGS,
    apply_server_args_update,
    validate_server_args_update,
)
from sglang.srt.runtime_context import (
    get_context,
    get_schedule,
    get_serving,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _no_spec_algorithm():
    return SimpleNamespace(is_dspark=lambda: False, is_none=lambda: True)


class TestValidateServerArgsUpdate(CustomTestCase):
    def _validate(self, server_args_dict, **overrides):
        kwargs = dict(
            max_running_requests=64,
            pp_size=2,
            spec_algorithm=_no_spec_algorithm(),
            draft_worker=None,
        )
        kwargs.update(overrides)
        return validate_server_args_update(server_args_dict, **kwargs)

    def test_non_allowlisted_field_is_rejected(self):
        self.assertIn("not supported", self._validate({"tp_size": 4}))
        self.assertNotIn("tp_size", HOT_UPDATABLE_SERVER_ARGS)

    def test_pp_max_micro_batch_size_range(self):
        self.assertIsNone(self._validate({"pp_max_micro_batch_size": 32}))
        self.assertIn("valid range", self._validate({"pp_max_micro_batch_size": 33}))
        self.assertIn("valid range", self._validate({"pp_max_micro_batch_size": 0}))

    def test_schedule_conservativeness_must_be_positive(self):
        self.assertIsNone(self._validate({"schedule_conservativeness": 0.5}))
        self.assertIn("positive", self._validate({"schedule_conservativeness": 0}))

    def test_stream_interval_must_be_positive_int(self):
        self.assertIsNone(self._validate({"stream_interval": 4}))
        self.assertIn(">= 1", self._validate({"stream_interval": 0}))

    def test_dspark_commands_require_dspark_worker(self):
        self.assertIn("DSpark", self._validate({"dspark_force_budget_frac": 0.5}))
        self.assertIn("DSpark", self._validate({"dspark_clear_info_records": 1}))

    def test_dspark_budget_frac_range_with_dspark_worker(self):
        dspark = SimpleNamespace(is_dspark=lambda: True, is_none=lambda: False)
        worker = MagicMock()
        self.assertIsNone(
            self._validate(
                {"dspark_force_budget_frac": 0.5},
                spec_algorithm=dspark,
                draft_worker=worker,
            )
        )
        self.assertIn(
            "(0, 1]",
            self._validate(
                {"dspark_force_budget_frac": 1.5},
                spec_algorithm=dspark,
                draft_worker=worker,
            ),
        )


class TestApplyServerArgsUpdate(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)

    def test_override_lands_in_bags_with_provenance(self):
        new_value = get_schedule().schedule_conservativeness * 2

        applied = apply_server_args_update(
            {"schedule_conservativeness": new_value, "stream_interval": 7},
            draft_worker=None,
        )

        self.assertEqual(
            applied,
            {"schedule_conservativeness": new_value, "stream_interval": 7},
        )
        self.assertEqual(get_schedule().schedule_conservativeness, new_value)
        self.assertEqual(get_serving().stream_interval, 7)
        self.assertIn("update_server_args", str(get_context().overrides_log()))

    def test_dspark_commands_route_to_worker_not_bags(self):
        worker = MagicMock()

        applied = apply_server_args_update(
            {"dspark_force_budget_frac": 0.5, "dspark_clear_info_records": 1},
            draft_worker=worker,
        )

        self.assertEqual(applied, {})
        worker.set_dspark_forced_budget_frac.assert_called_once_with(0.5)
        worker.clear_info_records.assert_called_once()
        self.assertNotIn("dspark_force_budget_frac", str(get_context().overrides_log()))


if __name__ == "__main__":
    unittest.main()
