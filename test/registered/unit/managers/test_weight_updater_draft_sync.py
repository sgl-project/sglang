"""Distributed (NCCL-broadcast) weight sync also refreshes the speculative draft.

Regression coverage for #27718. The distributed path previously updated only
the target model, leaving the speculative draft (e.g. MTP/NEXTN layers) stale.
The fix loads the *same* broadcasted tensors into the draft model(s) via
`extra_models` — a single broadcast feeds every model, so there is no second
NCCL collective on the co-located draft (which would deadlock, since the sender
broadcasts each tensor exactly once).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager(tp_worker, draft_worker):
    flush_calls = []

    def flush_cache(**kwargs):
        flush_calls.append(kwargs)
        return True

    mgr = SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=None,
        memory_saver_adapter=None,
        flush_cache=flush_cache,
        is_fully_idle=lambda: True,
    )
    return mgr, flush_calls


def _recv_req():
    # Only the fields the scheduler-side path touches.
    return SimpleNamespace(flush_cache=True, torch_empty_cache=False)


class _RecordingModel:
    """Minimal stand-in for a model: records the names passed to load_weights."""

    def __init__(self):
        self.load_calls = []

    def load_weights(self, weights):
        self.load_calls.append([name for name, _ in weights])


class TestDistributedWeightUpdateDraftSync(unittest.TestCase):
    def test_manager_forwards_draft_models_as_extra(self):
        # Core regression for #27718: the draft model(s) must be refreshed too,
        # by piggy-backing on the target's single broadcast.
        tp_worker = MagicMock()
        tp_worker.update_weights_from_distributed.return_value = (True, "")
        draft_worker = MagicMock()
        draft_models = [object(), object()]
        draft_worker.draft_models = draft_models
        mgr, flush_calls = _make_manager(tp_worker, draft_worker)

        recv_req = _recv_req()
        out = mgr.update_weights_from_distributed(recv_req)

        self.assertTrue(out.success)
        tp_worker.update_weights_from_distributed.assert_called_once_with(
            recv_req, extra_models=draft_models
        )
        # The draft must NOT be refreshed via a second (deadlock-prone)
        # collective on the co-located draft worker.
        draft_worker.update_weights_from_distributed.assert_not_called()
        self.assertEqual(len(flush_calls), 1)

    def test_no_draft_worker_passes_empty_extra(self):
        tp_worker = MagicMock()
        tp_worker.update_weights_from_distributed.return_value = (True, "")
        mgr, flush_calls = _make_manager(tp_worker, draft_worker=None)

        recv_req = _recv_req()
        mgr.update_weights_from_distributed(recv_req)

        tp_worker.update_weights_from_distributed.assert_called_once_with(
            recv_req, extra_models=[]
        )
        self.assertEqual(len(flush_calls), 1)

    def test_target_failure_skips_flush(self):
        tp_worker = MagicMock()
        tp_worker.update_weights_from_distributed.return_value = (False, "boom")
        draft_worker = MagicMock()
        draft_worker.draft_models = []
        mgr, flush_calls = _make_manager(tp_worker, draft_worker)

        out = mgr.update_weights_from_distributed(_recv_req())

        self.assertFalse(out.success)
        self.assertEqual(out.message, "boom")
        self.assertEqual(len(flush_calls), 0)

    def test_weight_updater_single_broadcast_dual_load(self):
        # The crux: exactly one broadcast per weight, loaded into the target
        # AND every extra model. Proves there is no second broadcast for the
        # co-located draft (the deadlock the naive fan-out would cause).
        try:
            from sglang.srt.model_executor.model_runner_components.weight_updater import (
                WeightUpdater,
            )
        except Exception as e:  # pragma: no cover - import-env dependent
            self.skipTest(f"WeightUpdater not importable in this env: {e}")

        target = _RecordingModel()
        draft = _RecordingModel()
        mr = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: target,
            update_model_fields=lambda **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        mr._model_update_group["g"] = object()

        broadcasts = []

        class _Handle:
            def wait(self):
                pass

        def fake_broadcast(tensor, src, group, async_op=False):
            broadcasts.append((src, tuple(tensor.shape)))
            return _Handle()

        original = torch.distributed.broadcast
        torch.distributed.broadcast = fake_broadcast
        try:
            ok, msg = mr.update_weights_from_distributed(
                names=["w0", "w1"],
                dtypes=[torch.float32, torch.float32],
                shapes=[(2,), (3,)],
                group_name="g",
                extra_models=[draft],
            )
        finally:
            torch.distributed.broadcast = original

        self.assertTrue(ok, msg)
        # One broadcast per weight — NOT doubled for the draft.
        self.assertEqual(len(broadcasts), 2)
        # The same materialized weights are loaded into both models.
        self.assertEqual(target.load_calls, [["w0", "w1"]])
        self.assertEqual(draft.load_calls, [["w0", "w1"]])


if __name__ == "__main__":
    unittest.main()
