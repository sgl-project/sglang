"""Unit tests for who publishes weights in remote-instance (R-Fork) transfer.

Regression guard: a seed launched with
--remote-instance-weight-loader-start-seed-via-transfer-engine plus
speculative decoding used to have its draft ModelRunner publish to the
engine-info bootstrap server. The bootstrap server keys entries by tp_rank
alone, and the draft runner initializes after the target, so the draft's
session id and weight layout overwrote the target's -- the client then pulled
the draft model's memory regions for its target model.
"""

import unittest
from unittest import mock

from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRemoteInstanceWeightTransporterPublishing(CustomTestCase):
    def setUp(self):
        # The seed short-circuits this predicate to True off
        # --remote-instance-weight-loader-start-seed-via-transfer-engine,
        # independently of load_format -- so overriding the draft's
        # load_format (--speculative-draft-load-format) does not keep the
        # draft runner off the publish side. Only is_draft_worker does.
        patcher = mock.patch(
            "sglang.srt.model_executor.model_runner_components."
            "remote_instance_weight_transporter."
            "remote_instance_transfer_engine_enabled",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make_transporter(self, *, is_draft_worker: bool):
        return RemoteInstanceWeightTransporter(
            get_model=lambda: self.fail(
                "a non-publishing runner must not touch its model"
            ),
            tp_rank=0,
            gpu_id=0,
            is_draft_worker=is_draft_worker,
        )

    def test_target_runner_publishes(self):
        # Pins the positive branch: without this the gate could degrade to
        # always-False and silently disable R-Fork altogether.
        self.assertTrue(self._make_transporter(is_draft_worker=False).publishes_weights)

    def test_draft_runner_never_publishes(self):
        # Reached with an engine in hand, which is the configuration that
        # actually happens: init_engine is gated separately (a draft loading
        # over remote_instance still needs one to receive weights), so the
        # publish side has to refuse on its own.
        transporter = self._make_transporter(is_draft_worker=True)
        transporter.engine = object()

        self.assertFalse(transporter.publishes_weights)
        transporter.maybe_register_and_publish_weight_info()

        self.assertIsNone(transporter.weight_info)


if __name__ == "__main__":
    unittest.main()
