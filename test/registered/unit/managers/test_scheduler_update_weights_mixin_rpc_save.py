"""Unit tests for scheduler_update_weights_mixin RPC save helpers.

Regression target: Scheduler.handle_rpc_request dispatches RPC parameters as
``func(**recv_req.parameters)``. The mixin save helpers therefore must accept
keyword arguments directly, not only a single ``params`` dict.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerUpdateWeightsMixinRpcSave(CustomTestCase):
    def setUp(self):
        self.scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=MagicMock()),
            draft_worker=None,
        )

    def test_save_sharded_model_accepts_rpc_kwargs(self):
        SchedulerUpdateWeightsMixin.save_sharded_model(
            self.scheduler,
            path="/tmp/out",
            pattern="model-rank{rank}.safetensors",
            max_size=1024,
        )

        self.scheduler.tp_worker.model_runner.save_sharded_model.assert_called_once_with(
            path="/tmp/out",
            pattern="model-rank{rank}.safetensors",
            max_size=1024,
        )

    def test_save_sharded_model_accepts_legacy_params_dict(self):
        SchedulerUpdateWeightsMixin.save_sharded_model(
            self.scheduler,
            {"path": "/tmp/out", "pattern": None, "max_size": 2048},
        )

        self.scheduler.tp_worker.model_runner.save_sharded_model.assert_called_once_with(
            path="/tmp/out",
            pattern=None,
            max_size=2048,
        )

    def test_save_remote_model_accepts_rpc_kwargs_for_tp_and_draft(self):
        self.scheduler.draft_worker = SimpleNamespace(model_runner=MagicMock())

        SchedulerUpdateWeightsMixin.save_remote_model(
            self.scheduler,
            url="s3://bucket/tp",
            draft_url="s3://bucket/draft",
        )

        self.scheduler.tp_worker.model_runner.save_remote_model.assert_called_once_with(
            "s3://bucket/tp"
        )
        self.scheduler.draft_worker.model_runner.save_remote_model.assert_called_once_with(
            "s3://bucket/draft"
        )

    def test_save_remote_model_accepts_legacy_params_dict(self):
        SchedulerUpdateWeightsMixin.save_remote_model(
            self.scheduler,
            {"url": "s3://bucket/tp"},
        )

        self.scheduler.tp_worker.model_runner.save_remote_model.assert_called_once_with(
            "s3://bucket/tp"
        )

    def test_save_sharded_model_kwargs_take_priority_over_params_dict(self):
        SchedulerUpdateWeightsMixin.save_sharded_model(
            self.scheduler,
            {"path": "/from-dict", "pattern": "from-dict", "max_size": 1},
            path="/from-kwargs",
            pattern="from-kwargs",
            max_size=2,
        )

        self.scheduler.tp_worker.model_runner.save_sharded_model.assert_called_once_with(
            path="/from-kwargs",
            pattern="from-kwargs",
            max_size=2,
        )

    def test_save_sharded_model_missing_path_raises_assertion_not_keyerror(self):
        with self.assertRaises(AssertionError) as ctx:
            SchedulerUpdateWeightsMixin.save_sharded_model(
                self.scheduler,
                {"pattern": "only-pattern"},
            )

        self.assertIn("path must be provided", str(ctx.exception))

    def test_save_remote_model_missing_url_raises_assertion_not_keyerror(self):
        with self.assertRaises(AssertionError) as ctx:
            SchedulerUpdateWeightsMixin.save_remote_model(
                self.scheduler,
                {"draft_url": "s3://bucket/draft"},
            )

        self.assertIn("url must be provided", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
