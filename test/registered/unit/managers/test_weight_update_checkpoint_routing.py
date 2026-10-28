"""A from-disk weight update must reach the model that owns the checkpoint.

A speculative deployment serves two models and a from-disk update names one
checkpoint. Handing that path to every runner loads the target's weights into
the draft, which fits and therefore passes unnoticed, and speculation breaks.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TARGET_PATH = "/models/target"
DRAFT_PATH = "/models/draft"


def _runner(model_path):
    """A model runner that serves the checkpoint at ``model_path``."""
    runner = Mock()
    runner.model_config = SimpleNamespace(model_path=model_path)
    runner.weight_updater.update_weights_from_disk.return_value = (True, "ok")
    return runner


def _request(model_path, selector=None):
    """A from-disk update request.

    ``selector`` is not a field of ``UpdateWeightFromDiskReqInput`` yet (it
    arrives with the upstream selector PR), so it is added here when a test
    asks for it.
    """
    fields = dict(
        model_path=model_path,
        load_format="auto",
        recapture_cuda_graph=False,
        flush_cache=True,
        torch_empty_cache=False,
        weight_version="v2",
    )
    if selector is not None:
        fields["selector"] = selector
    return SimpleNamespace(**fields)


class TestDiskUpdateCheckpointRouting(CustomTestCase):
    def _manager(self, target, draft=None):
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            return SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(
                    model_runner=target,
                    weight_update_runners=lambda: [("target", target)],
                    deserialize_own_rank=lambda payloads: [],
                ),
                draft_worker=(
                    None
                    if draft is None
                    else SimpleNamespace(
                        weight_update_runners=lambda: [("draft", draft)]
                    )
                ),
                memory_saver_adapter=None,
                flush_cache=lambda **kwargs: True,
                is_fully_idle=lambda **kwargs: True,
                scheduler=SimpleNamespace(
                    record_weight_version_change=lambda new_version: None
                ),
            )

    def _updated(self, runner):
        return runner.weight_updater.update_weights_from_disk.call_count

    def test_target_checkpoint_skips_the_draft(self):
        """The target's checkpoint must not be loaded into the draft."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        output = self._manager(target, draft).update_weights_from_disk(
            _request(TARGET_PATH)
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 1)
        self.assertEqual(self._updated(draft), 0)

    def test_draft_checkpoint_skips_the_target(self):
        """The draft's checkpoint must not be loaded into the target."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        output = self._manager(target, draft).update_weights_from_disk(
            _request(DRAFT_PATH)
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 0)
        self.assertEqual(self._updated(draft), 1)

    def test_the_skipped_runner_keeps_its_own_checkpoint(self):
        """Skipping a runner must not move it onto the other model's path."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        self._manager(target, draft).update_weights_from_disk(_request(TARGET_PATH))

        self.assertEqual(draft.model_config.model_path, DRAFT_PATH)

    def test_the_selected_runner_receives_the_requested_path(self):
        """The runner that owns the checkpoint is handed that checkpoint."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        self._manager(target, draft).update_weights_from_disk(_request(DRAFT_PATH))

        draft.weight_updater.update_weights_from_disk.assert_called_once_with(
            DRAFT_PATH, "auto", recapture_cuda_graph=False
        )

    def test_explicit_selector_overrides_ownership(self):
        """The caller states which model it means, so the routing steps aside."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        output = self._manager(target, draft).update_weights_from_disk(
            _request(TARGET_PATH, selector="draft")
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 0)
        self.assertEqual(self._updated(draft), 1)

    def test_explicit_target_selector_overrides_ownership(self):
        """The same holds for the target."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        output = self._manager(target, draft).update_weights_from_disk(
            _request(DRAFT_PATH, selector="target")
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 1)
        self.assertEqual(self._updated(draft), 0)

    def test_an_unknown_path_updates_the_target_only(self):
        """A checkpoint neither runner serves is a new target checkpoint."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)

        output = self._manager(target, draft).update_weights_from_disk(
            _request("/models/brand-new")
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 1)
        self.assertEqual(self._updated(draft), 0)

    def test_a_shared_path_updates_both(self):
        """One checkpoint serving both models stays a fan-out."""
        target, draft = _runner(TARGET_PATH), _runner(TARGET_PATH)

        output = self._manager(target, draft).update_weights_from_disk(
            _request(TARGET_PATH)
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 1)
        self.assertEqual(self._updated(draft), 1)

    def test_unknown_runner_paths_keep_updating_every_runner(self):
        """Without a known checkpoint path the update cannot be attributed."""
        target, draft = _runner(None), _runner(None)

        output = self._manager(target, draft).update_weights_from_disk(
            _request("/models/target")
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 1)
        self.assertEqual(self._updated(draft), 1)

    def test_without_a_draft_worker_only_the_target_updates(self):
        """A plain deployment keeps its single-model behaviour."""
        target = _runner(TARGET_PATH)

        output = self._manager(target).update_weights_from_disk(
            _request(TARGET_PATH)
        )

        self.assertTrue(output.success)
        self.assertEqual(self._updated(target), 1)

    def test_a_failing_runner_stops_the_fan_out(self):
        """A failed update must not be reported as a success."""
        target, draft = _runner(TARGET_PATH), _runner(DRAFT_PATH)
        draft.weight_updater.update_weights_from_disk.return_value = (False, "boom")

        output = self._manager(target, draft).update_weights_from_disk(
            _request(DRAFT_PATH)
        )

        self.assertFalse(output.success)
        self.assertEqual(self._updated(target), 0)


if __name__ == "__main__":
    unittest.main()
