"""Every weight mutation must invalidate capture before touching parameters."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCaptureWeightUpdates(CustomTestCase):
    def requests(self):
        return (
            ("disk", UpdateWeightFromDiskReqInput(model_path="new-weights")),
            (
                "distributed",
                UpdateWeightsFromDistributedReqInput(names=[], dtypes=[], shapes=[]),
            ),
            ("tensor", UpdateWeightsFromTensorReqInput(serialized_named_tensors=[])),
            ("ipc", UpdateWeightsFromIPCReqInput(zmq_handles={})),
        )

    def manager(self):
        worker = SimpleNamespace(training_capture=Mock())
        return SchedulerWeightUpdaterManager(
            tp_worker=worker,
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=Mock(return_value=True),
            is_fully_idle=lambda: True,
        )

    def test_disable_precedes_success_failure_and_partial_load_exception(self):
        for source, request in self.requests():
            for outcome in (True, False, RuntimeError("partial weight mutation")):
                with self.subTest(source=source, outcome=outcome):
                    manager = self.manager()
                    capture = manager.tp_worker.training_capture

                    def load(_, capture=capture, outcome=outcome):
                        capture.disable.assert_called_once_with("target_weights_update")
                        if isinstance(outcome, Exception):
                            raise outcome
                        return outcome, "test loader result"

                    setattr(manager.tp_worker, "update_weights_from_" + source, load)
                    update = getattr(manager, "update_weights_from_" + source)
                    with patch("torch.distributed.barrier"):
                        if isinstance(outcome, Exception):
                            with self.assertRaisesRegex(RuntimeError, "partial weight"):
                                update(request)
                        else:
                            self.assertEqual(update(request).success, outcome)
                    capture.disable.assert_called_once_with("target_weights_update")
                    self.assertEqual(
                        manager.flush_cache.call_count, int(outcome is True)
                    )

    def test_target_kv_rejection_preserves_capture_and_weights(self):
        for source, request in self.requests():
            with self.subTest(source=source):
                manager = self.manager()
                manager.draft_worker = SimpleNamespace(
                    draft_model_runner=SimpleNamespace(
                        model_config=SimpleNamespace(
                            hf_config=SimpleNamespace(
                                architectures=["DSparkTargetKVDraftModel"]
                            )
                        )
                    )
                )
                load = Mock(side_effect=AssertionError("weights must not be touched"))
                setattr(manager.tp_worker, "update_weights_from_" + source, load)
                response = getattr(manager, "update_weights_from_" + source)(request)
                self.assertFalse(response.success)
                self.assertIn("deploy a new service instance", response.message)
                load.assert_not_called()
                manager.tp_worker.training_capture.disable.assert_not_called()
                manager.flush_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
