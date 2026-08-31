import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _ServingStub:
    def __init__(self, weight_version: str):
        self.weight_version = weight_version


class _ContextStub:
    def __init__(self, serving: _ServingStub):
        self.serving = serving

    def override(self, source, **fields):
        self.serving.weight_version = fields["weight_version"]


class TestSchedulerRecordWeightVersionChange(CustomTestCase):
    def _serving(self, version: str) -> _ServingStub:
        serving = _ServingStub(version)
        for name, value in (
            ("get_serving", serving),
            ("get_context", _ContextStub(serving)),
        ):
            patcher = patch(f"sglang.srt.managers.scheduler.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return serving

    def _scheduler(
        self, *, inflight=(), waiting=(), chunked=None, staging=()
    ) -> SimpleNamespace:
        return SimpleNamespace(
            collect_inflight_reqs=lambda: set(inflight),
            waiting_queue=list(waiting),
            chunked_req=chunked,
            hisparse_coordinator=(
                SimpleNamespace(
                    ack_staging_queue=[SimpleNamespace(req=req) for req in staging]
                )
                if staging
                else None
            ),
        )

    def test_a_new_version_is_adopted(self):
        """The scheduler has to end up on the version it was told about, or nothing downstream can read it."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version="v2")

        self.assertEqual(serving.weight_version, "v2")

    def test_same_version_is_a_noop(self):
        """Re-announcing the current version must not be treated as a change."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version="v1")

        self.assertEqual(serving.weight_version, "v1")

    def test_none_version_is_a_noop(self):
        """An update that carries no version must leave the recorded one alone."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version=None)

        self.assertEqual(serving.weight_version, "v1")

    def test_every_source_of_live_requests_is_stamped(self):
        """A request missed here keeps attributing its next tokens to the superseded version."""
        self._serving("v1")
        inflight, queued, chunked, staged = (object() for _ in range(4))
        scheduler = self._scheduler(
            inflight=[inflight], waiting=[queued], chunked=chunked, staging=[staged]
        )

        with patch(
            "sglang.srt.managers.scheduler.record_weight_version_events",
            return_value=0,
        ) as recorder:
            Scheduler.record_weight_version_change(scheduler, new_version="v2")

        self.assertEqual(
            set(recorder.call_args.args[0]), {inflight, queued, chunked, staged}
        )


class TestRecordWeightVersionAfterUpdate(CustomTestCase):
    def _updater(
        self, target_result, draft_result=None, method="update_weights_from_disk"
    ):
        self.recorded = []
        return SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(**{method: lambda recv_req: target_result}),
            draft_worker=(
                None
                if draft_result is None
                else SimpleNamespace(**{method: lambda recv_req: draft_result})
            ),
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                )
            ),
        )

    def _request(self, **fields):
        return SimpleNamespace(
            weight_version="v2",
            flush_cache=True,
            torch_empty_cache=False,
            **fields,
        )

    def test_successful_update_records_the_version(self):
        """A refit that reports success advances the scheduler-side version."""
        updater = self._updater(target_result=(True, "ok"))

        output = updater.update_weights_from_disk(self._request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_update_does_not_record_the_version(self):
        """A refit that fails must leave the version alone, or later tokens are mislabelled."""
        updater = self._updater(target_result=(False, "boom"))

        output = updater.update_weights_from_disk(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_draft_failure_does_not_record_the_version(self):
        """The target succeeding is not enough: a failed draft refit leaves the engine mixed."""
        updater = self._updater(
            target_result=(True, "ok"), draft_result=(False, "draft boom")
        )

        output = updater.update_weights_from_disk(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_distributed_update_records_the_version(self):
        """The distributed refit is the path an RL trainer actually drives, so it must record too."""
        updater = self._updater(
            target_result=(True, "ok"), method="update_weights_from_distributed"
        )

        output = updater.update_weights_from_distributed(self._request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_distributed_update_does_not_record_the_version(self):
        """A failed distributed refit leaves the version alone, exactly like the disk path."""
        updater = self._updater(
            target_result=(False, "boom"), method="update_weights_from_distributed"
        )

        output = updater.update_weights_from_distributed(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_tensor_update_records_the_version(self):
        """The tensor refit records the version once the load reports success."""
        updater = self._updater(
            target_result=(True, "ok"), method="update_weights_from_tensor"
        )

        with patch("torch.distributed.barrier"):
            output = updater.update_weights_from_tensor(
                self._request(disable_draft_model=True)
            )

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_successful_ipc_update_records_the_version(self):
        """The checkpoint-engine IPC refit records the version like every other path."""
        updater = self._updater(
            target_result=(True, "ok"), method="update_weights_from_ipc"
        )

        with patch("torch.distributed.barrier"):
            output = updater.update_weights_from_ipc(self._request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_ipc_update_does_not_record_the_version(self):
        """The IPC path branches on success separately from the cache flush, so failure must record nothing."""
        updater = self._updater(
            target_result=(False, "boom"), method="update_weights_from_ipc"
        )

        with patch("torch.distributed.barrier"):
            output = updater.update_weights_from_ipc(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])


if __name__ == "__main__":
    unittest.main()
