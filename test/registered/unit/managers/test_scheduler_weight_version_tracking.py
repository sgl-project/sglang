import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


def _runner(result=(True, "ok")):
    runner = Mock()
    for method in (
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
        "load_weights_from_distributed",
    ):
        getattr(runner.weight_updater, method).return_value = result
    return runner


def _request(**fields):
    return SimpleNamespace(
        **{
            "weight_version": "v2",
            "flush_cache": True,
            "torch_empty_cache": False,
            "model_path": "m",
            "recapture_cuda_graph": False,
            "load_format": None,
            "selector": "all",
            "names": ["model.layers.0.weight"],
            "dtypes": ["float32"],
            "shapes": [[1]],
            "group_name": "g",
            "serialized_named_tensors": [b""],
            **fields,
        }
    )


class _WeightUpdaterManagerTestBase(CustomTestCase):
    def setUp(self):
        patcher = patch("torch.distributed.barrier")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recorded = []

    def _build_manager(self, target, draft):
        return SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=target,
                weight_update_runners=lambda: [("target", target)],
                deserialize_own_rank=lambda payloads: [],
            ),
            draft_worker=(
                None
                if draft is None
                else SimpleNamespace(weight_update_runners=lambda: [("draft", draft)])
            ),
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                )
            ),
        )

    def _manager(self, target, draft=None, *, session=True):
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = self._build_manager(target, draft)
        if session:
            # update_weights_from_* require an open session
            manager._session = _WeightUpdateSession(selector="all")
        return manager


class TestRecordWeightVersionAfterUpdate(_WeightUpdaterManagerTestBase):
    def test_successful_update_records_the_version(self):
        """A refit that reports success advances the scheduler-side version."""
        output = self._manager(_runner()).update_weights_from_disk(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_update_does_not_record_the_version(self):
        """A refit that fails must leave the version alone, or later tokens are mislabelled."""
        output = self._manager(_runner((False, "boom"))).update_weights_from_disk(
            _request()
        )

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_draft_failure_does_not_record_the_version(self):
        """The target succeeding is not enough: a failed draft refit leaves the engine mixed."""
        manager = self._manager(_runner(), draft=_runner((False, "draft boom")))

        output = manager.update_weights_from_disk(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_distributed_update_records_the_version(self):
        """The distributed refit is the path an RL trainer actually drives, so it must record too."""
        output = self._manager(_runner()).update_weights_from_distributed(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_distributed_update_does_not_record_the_version(self):
        """A failed distributed refit leaves the version alone, exactly like the disk path."""
        output = self._manager(
            _runner((False, "boom"))
        ).update_weights_from_distributed(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_tensor_update_records_the_version(self):
        """The tensor refit records the version once the load reports success."""
        output = self._manager(_runner()).update_weights_from_tensor(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_successful_ipc_update_records_the_version(self):
        """The checkpoint-engine IPC refit records the version like every other path."""
        output = self._manager(_runner()).update_weights_from_ipc(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_ipc_update_does_not_record_the_version(self):
        """The IPC path branches on success separately from the cache flush, so failure must record nothing."""
        output = self._manager(_runner((False, "boom"))).update_weights_from_ipc(
            _request()
        )

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])


class TestWeightUpdateSession(_WeightUpdaterManagerTestBase):
    def test_distributed_update_receives_once_on_target_loads_into_each(self):
        """A draft runner that received its own broadcast would deadlock the update group."""
        target, draft = _runner(), _runner()
        weights = target.weight_updater.receive_weights_from_distributed.return_value
        req = _request()

        output = self._manager(target, draft).update_weights_from_distributed(req)

        self.assertTrue(output.success)
        target.weight_updater.receive_weights_from_distributed.assert_called_once_with(
            names=req.names,
            dtypes=req.dtypes,
            shapes=req.shapes,
            group_name=req.group_name,
            load_format=req.load_format,
        )
        target.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.receive_weights_from_distributed.assert_not_called()

    def test_distributed_update_target_only_selector_skips_draft(self):
        """selector="target" must not load into the draft."""
        target, draft = _runner(), _runner()

        output = self._manager(target, draft).update_weights_from_distributed(
            _request(selector="target")
        )

        self.assertTrue(output.success)
        target.weight_updater.load_weights_from_distributed.assert_called_once()
        draft.weight_updater.load_weights_from_distributed.assert_not_called()

    def test_end_runs_post_load_on_both_when_load_was_bypassed(self):
        """A P2P/RDMA session never calls load_weights, so end must run post_load_weights."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)

        output = manager.end_weight_update(EndWeightUpdateReqInput())

        self.assertTrue(output.success)
        for runner in (target, draft):
            runner.weight_updater.end_weight_update.assert_called_once_with(
                run_post_load=True
            )
        self.assertIsNone(manager._session)

    def test_end_skips_post_load_on_both_when_weights_loaded(self):
        """load_weights already ran post_load_weights; running it twice would double-apply."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)
        manager._session = _WeightUpdateSession(
            selector="all", loaded_roles=frozenset({"target", "draft"})
        )

        manager.end_weight_update(EndWeightUpdateReqInput())

        for runner in (target, draft):
            runner.weight_updater.end_weight_update.assert_called_once_with(
                run_post_load=False
            )

    def test_session_selector_confines_begin_and_end_to_selected_runners(self):
        """end finalizing a runner begin never restored would repack unrestored weights."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft, session=False)

        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
        manager.end_weight_update(EndWeightUpdateReqInput())

        target.weight_updater.begin_weight_update.assert_not_called()
        target.weight_updater.end_weight_update.assert_not_called()
        draft.weight_updater.begin_weight_update.assert_called_once_with()
        draft.weight_updater.end_weight_update.assert_called_once()

    def test_begin_rejects_reentry(self):
        """A second begin would leave the first session's runners unfinalized."""
        target = _runner()

        output = self._manager(target).begin_weight_update(BeginWeightUpdateReqInput())

        self.assertFalse(output.success)
        self.assertIn("already open", output.message)
        target.weight_updater.begin_weight_update.assert_not_called()

    def test_end_without_session_is_rejected(self):
        """Finalizing runners begin never restored would repack weights twice."""
        target = _runner()

        output = self._manager(target, session=False).end_weight_update(
            EndWeightUpdateReqInput()
        )

        self.assertFalse(output.success)
        self.assertIn("begin_weight_update", output.message)
        target.weight_updater.end_weight_update.assert_not_called()

    def test_update_without_session_is_rejected_without_loading(self):
        """A caller that skips begin gets an error back instead of crashing the scheduler."""
        target = _runner()

        output = self._manager(target, session=False).update_weights_from_distributed(
            _request()
        )

        self.assertFalse(output.success)
        self.assertIn("begin_weight_update", output.message)
        target.weight_updater.receive_weights_from_distributed.assert_not_called()


class _PostLoadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor(1.0))

    def load_weights(self, named_tensors):
        for _, tensor in named_tensors:
            self.weight.copy_(tensor)
        self.post_load_weights()

    def post_load_weights(self):
        self.weight.mul_(2)


def _stateful_runner(*, fail=False):
    model = _PostLoadModel()
    updater = SimpleNamespace(device="cpu", get_model=lambda: model)

    def load(named_tensors, **kwargs):
        if fail:
            return False, "draft load failed"
        model.load_weights(named_tensors)
        return True, "Success"

    updater.begin_weight_update = lambda: None
    updater.receive_weights_from_distributed = lambda **kwargs: [
        ("weight", torch.tensor(3.0))
    ]
    updater.load_weights_from_distributed = load
    updater.update_weights_from_tensor = load
    updater.end_weight_update = lambda **kwargs: WeightUpdater.end_weight_update(
        updater, **kwargs
    )
    return SimpleNamespace(model=model, weight_updater=updater)


class TestWeightUpdatePartialLoads(_WeightUpdaterManagerTestBase):
    def _stateful_manager(self, *, draft_fails=False, draft_steps=1):
        target = _stateful_runner()
        drafts = [
            _stateful_runner(fail=draft_fails and step == draft_steps - 1)
            for step in range(draft_steps)
        ]
        draft = drafts[-1]
        manager = self._manager(target, draft, session=False)
        manager.draft_worker.weight_update_runners = lambda: [
            ("draft" if draft_steps == 1 else f"draft_step_{step}", runner)
            for step, runner in enumerate(drafts)
        ]
        manager.tp_worker.deserialize_own_rank = lambda payloads: [
            ("weight", torch.tensor(3.0))
        ]
        cache = ["tokens computed with old weights"]
        manager.flush_cache = lambda **kwargs: cache.clear() or True
        manager.begin_weight_update(BeginWeightUpdateReqInput())
        return manager, target.model, draft.model, cache

    def test_partial_failure_invalidates_target_cache(self):
        """A failed draft refit must not leave cached tokens from the old target."""
        for source in ("distributed", "tensor"):
            with self.subTest(source=source):
                manager, target, _, cache = self._stateful_manager(draft_fails=True)
                output = getattr(manager, f"update_weights_from_{source}")(_request())

                self.assertFalse(output.success)
                self.assertEqual(target.weight.item(), 6.0)
                self.assertEqual(cache, [])
                self.assertEqual(self.recorded, [])

    def test_partial_failure_finalizes_each_runner_once(self):
        """Closing a failed fan-out must not apply the target's model fixup twice."""
        for source in ("distributed", "tensor"):
            with self.subTest(source=source):
                manager, target, draft, _ = self._stateful_manager(draft_fails=True)
                output = getattr(manager, f"update_weights_from_{source}")(_request())
                self.assertFalse(output.success)

                manager.end_weight_update(EndWeightUpdateReqInput())

                self.assertEqual(target.weight.item(), 6.0)
                self.assertEqual(draft.weight.item(), 2.0)
                self.assertIsNone(manager._session)

    def test_subset_load_still_finalizes_unloaded_runner(self):
        """Loading one role must not suppress the other role's end-of-session fixup."""
        for source in ("distributed", "tensor"):
            for selector in ("target", "draft"):
                with self.subTest(source=source, selector=selector):
                    manager, target, draft, _ = self._stateful_manager()
                    output = getattr(manager, f"update_weights_from_{source}")(
                        _request(selector=selector)
                    )
                    self.assertTrue(output.success)

                    manager.end_weight_update(EndWeightUpdateReqInput())

                    self.assertEqual(
                        (target.weight.item(), draft.weight.item()),
                        (6.0, 2.0) if selector == "target" else (2.0, 6.0),
                    )

    def test_separate_updates_accumulate_loaded_roles(self):
        """A later role update must not forget the earlier role's completed fixup."""
        manager, target, draft, _ = self._stateful_manager()
        manager.update_weights_from_tensor(_request(selector="target"))
        manager.update_weights_from_distributed(_request(selector="draft"))

        manager.end_weight_update(EndWeightUpdateReqInput())

        self.assertEqual((target.weight.item(), draft.weight.item()), (6.0, 6.0))

    def test_later_draft_failure_preserves_earlier_draft_fixup(self):
        """A failed later draft step must not reapply a completed earlier step's fixup."""
        for source in ("distributed", "tensor"):
            with self.subTest(source=source):
                manager, target, _, _ = self._stateful_manager(
                    draft_fails=True, draft_steps=2
                )
                draft_models = [
                    runner.model
                    for _, runner in manager.draft_worker.weight_update_runners()
                ]
                output = getattr(manager, f"update_weights_from_{source}")(_request())
                self.assertFalse(output.success)

                manager.end_weight_update(EndWeightUpdateReqInput())

                self.assertEqual(
                    [model.weight.item() for model in [target, *draft_models]],
                    [6.0, 6.0, 2.0],
                )
                self.assertEqual(self.recorded, [])

    def test_draft_exception_still_invalidates_target_cache(self):
        """A loader exception must not strand old cache entries after a target refit."""
        for source in ("distributed", "tensor"):
            with self.subTest(source=source):
                manager, target, draft, cache = self._stateful_manager()

                def fail(*args, **kwargs):
                    raise RuntimeError("draft loader raised")

                draft_runner = manager.draft_worker.weight_update_runners()[0][1]
                setattr(
                    draft_runner.weight_updater,
                    "load_weights_from_distributed"
                    if source == "distributed"
                    else "update_weights_from_tensor",
                    fail,
                )
                with self.assertRaisesRegex(RuntimeError, "draft loader raised"):
                    getattr(manager, f"update_weights_from_{source}")(_request())

                self.assertEqual(cache, [])
                self.assertEqual(self.recorded, [])
                manager.end_weight_update(EndWeightUpdateReqInput())
                self.assertEqual(
                    (target.weight.item(), draft.weight.item()), (6.0, 2.0)
                )


if __name__ == "__main__":
    unittest.main()
