"""Unit tests for remote-instance (R-Fork) weight transfer participation.

Regression guard: a seed launched with
--remote-instance-weight-loader-start-seed-via-transfer-engine plus
speculative decoding used to have its draft ModelRunner open its own
TransferEngine and publish to the engine-info bootstrap server. The bootstrap
server keys entries by tp_rank alone, and the draft runner initializes after
the target, so the draft's session id and weight layout overwrote the target's
-- the client then pulled the draft model's memory regions for its target model.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRemoteInstanceWeightTransporterParticipation(CustomTestCase):
    def _make_transporter(self, *, is_draft_worker: bool):
        # The seed short-circuits this predicate to True off
        # --remote-instance-weight-loader-start-seed-via-transfer-engine,
        # independently of load_format -- so overriding the draft's
        # load_format (--speculative-draft-load-format) does not keep the
        # draft runner out. Only is_draft_worker does.
        server_args = SimpleNamespace(
            remote_instance_weight_loader_use_transfer_engine=lambda: True,
            remote_instance_weight_loader_backend="nccl",
        )
        return RemoteInstanceWeightTransporter(
            server_args=server_args,
            get_model=lambda: self.fail(
                "a non-participating runner must not touch its model"
            ),
            tp_rank=0,
            gpu_id=0,
            is_draft_worker=is_draft_worker,
        )

    def test_target_runner_participates(self):
        # Pins the positive branch: without this the gate could degrade to
        # always-False and silently disable R-Fork altogether.
        self.assertTrue(self._make_transporter(is_draft_worker=False).transfers_weights)

    def test_draft_runner_opens_no_engine(self):
        transporter = self._make_transporter(is_draft_worker=True)

        self.assertFalse(transporter.transfers_weights)
        transporter.maybe_init_engine()
        self.assertIsNone(transporter.engine)

    def test_draft_runner_never_publishes(self):
        # Second gate, checked independently of the first: even handed an
        # engine, the draft must not register memory or publish under the
        # target's tp_rank key.
        transporter = self._make_transporter(is_draft_worker=True)
        transporter.engine = object()

        transporter.maybe_register_and_publish_weight_info()

        self.assertIsNone(transporter.weight_info)


if __name__ == "__main__":
    unittest.main()
