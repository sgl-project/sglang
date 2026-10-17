"""A runner's refusal must precede mutations in every selected runner."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    CheckWeightsReqInput,
    EndWeightUpdateReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import get_parallel


def _runner(reason=None):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    model[0].weight_update_unsupported_reason = reason
    updater = Mock()
    for method in (
        "update_weights_from_disk",
        "load_weights_from_distributed",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
    ):
        getattr(updater, method).return_value = (True, "Success")
    return SimpleNamespace(
        model=model, weight_updater=updater, check_weights=Mock(return_value=None)
    )


def _request(selector="all"):
    return SimpleNamespace(
        model_path="unused",
        load_format=None,
        recapture_cuda_graph=False,
        flush_cache=True,
        torch_empty_cache=False,
        weight_version="v2",
        selector=selector,
        serialized_named_tensors=[],
        names=[],
        dtypes=[],
        shapes=[],
        group_name="test",
    )


class TestSchedulerWeightUpdateRefusal(CustomTestCase):
    def setUp(self):
        super().setUp()
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.barrier = patches.enter_context(patch("torch.distributed.barrier"))
        patches.enter_context(patch("torch.distributed.get_world_size", return_value=1))
        patches.enter_context(
            patch(
                "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
                return_value=False,
            )
        )

    def _manager(self, target, draft):
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            return SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(
                    model_runner=target,
                    weight_update_runners=lambda: [("target", target)],
                    deserialize_own_rank=Mock(return_value=[]),
                ),
                draft_worker=SimpleNamespace(
                    weight_update_runners=lambda: [("draft", draft)]
                ),
                memory_saver_adapter=None,
                flush_cache=Mock(return_value=True),
                is_fully_idle=lambda **kwargs: True,
                scheduler=Mock(),
            )

    def test_begin_refuses_before_target_restore_or_barrier(self):
        target, draft = _runner(), _runner("draft weights cannot be updated")
        manager = self._manager(target, draft)

        output = manager.begin_weight_update(BeginWeightUpdateReqInput())

        self.assertFalse(output.success)
        self.assertEqual(output.message, "draft weights cannot be updated")
        self.assertIsNone(manager._session)
        target.weight_updater.begin_weight_update.assert_not_called()
        draft.weight_updater.begin_weight_update.assert_not_called()
        self.barrier.assert_not_called()
        self.assertFalse(manager.end_weight_update(EndWeightUpdateReqInput()).success)
        self.assertFalse(manager.update_weights_from_tensor(_request()).success)
        self.assertFalse(manager.update_weights_from_distributed(_request()).success)
        target.weight_updater.receive_weights_from_distributed.assert_not_called()
        target.weight_updater.end_weight_update.assert_not_called()

    def test_unselected_refusal_does_not_prevent_session(self):
        target, draft = _runner(), _runner("draft only")
        manager = self._manager(target, draft)

        output = manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector="target")
        )

        self.assertTrue(output.success)
        target.weight_updater.begin_weight_update.assert_called_once_with()
        draft.weight_updater.begin_weight_update.assert_not_called()
        self.assertTrue(manager.end_weight_update(EndWeightUpdateReqInput()).success)

    def test_disk_and_ipc_refuse_before_target_load(self):
        for method in ("update_weights_from_disk", "update_weights_from_ipc"):
            with self.subTest(method=method):
                target, draft = _runner(), _runner("draft cannot update")
                manager = self._manager(target, draft)
                output = getattr(manager, method)(_request())
                self.assertFalse(output.success)
                self.assertEqual(output.message, "draft cannot update")
                getattr(target.weight_updater, method).assert_not_called()
                getattr(draft.weight_updater, method).assert_not_called()
                manager.flush_cache.assert_not_called()
                manager.scheduler.record_weight_version_change.assert_not_called()
        self.barrier.assert_not_called()

    def test_session_load_checks_all_runners_before_writing(self):
        # An update can select more runners than begin did. A rejected draft
        # must not leave the target with partially updated weights.
        for method, load in (
            ("update_weights_from_tensor", "update_weights_from_tensor"),
            ("update_weights_from_distributed", "load_weights_from_distributed"),
        ):
            with self.subTest(method=method):
                target, draft = _runner(), _runner("draft cannot update")
                manager = self._manager(target, draft)
                self.assertTrue(
                    manager.begin_weight_update(
                        BeginWeightUpdateReqInput(selector="target")
                    ).success
                )
                output = getattr(manager, method)(_request())
                self.assertFalse(output.success)
                self.assertEqual(output.message, "draft cannot update")
                getattr(target.weight_updater, load).assert_not_called()
                getattr(draft.weight_updater, load).assert_not_called()
                manager.flush_cache.assert_not_called()
                self.assertFalse(manager._session.loaded_weights)
                if method == "update_weights_from_distributed":
                    # Do not strand the sender after it has entered a broadcast.
                    receive = target.weight_updater.receive_weights_from_distributed
                    receive.assert_called_once()

    def test_reset_checks_all_runners_but_read_only_checks_remain_available(self):
        target, draft = _runner(), _runner("draft cannot reset")
        manager = self._manager(target, draft)

        output = manager.check_weights(CheckWeightsReqInput(action="reset_tensors"))

        self.assertFalse(output.success)
        self.assertEqual(output.message, "draft cannot reset")
        target.check_weights.assert_not_called()
        draft.check_weights.assert_not_called()
        self.assertTrue(manager.check_weights(CheckWeightsReqInput()).success)
        target.check_weights.assert_called_once()
        draft.check_weights.assert_called_once()

    def test_no_declaration_preserves_load_and_reset_paths(self):
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)
        self.assertTrue(
            manager.begin_weight_update(BeginWeightUpdateReqInput()).success
        )
        for method in (
            "update_weights_from_disk",
            "update_weights_from_tensor",
            "update_weights_from_distributed",
            "update_weights_from_ipc",
        ):
            with self.subTest(method=method):
                self.assertTrue(getattr(manager, method)(_request()).success)
        self.assertTrue(manager.end_weight_update(EndWeightUpdateReqInput()).success)
        self.assertTrue(
            manager.check_weights(CheckWeightsReqInput(action="reset_tensors")).success
        )
        for runner in (target, draft):
            runner.check_weights.assert_called_once()


if __name__ == "__main__":
    unittest.main()
