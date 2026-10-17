"""Weight-send group guards, without a server or distributed communication."""

import unittest
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_executor.model_runner_components.weight_exporter as exporter_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestWeightExporter(CustomTestCase):
    def setUp(self):
        patcher = patch.object(exporter_mod, "current_platform", Mock())
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(torch.distributed, "is_initialized", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_exporter(self, rank=0, size=1):
        return exporter_mod.WeightExporter(
            tp_rank=rank,
            tp_size=size,
            gpu_id=rank,
            get_model_path=Mock(),
            get_model=Mock(),
        )

    def test_missing_or_none_group_returns_error_without_touching_model(self):
        for size in (1, 2):
            ports = ",".join(str(29500 + rank) for rank in range(size))
            for rank in range(size):
                key = f"weight_send_group_{29500 + rank}_{rank}"
                for present in (False, True):
                    with self.subTest(size=size, rank=rank, none_entry=present):
                        exporter = self.make_exporter(rank, size)
                        other_group = object()
                        exporter._weights_send_group["unrelated"] = other_group
                        if present:
                            exporter._weights_send_group[key] = None
                        before = exporter._weights_send_group.copy()
                        success, message = exporter.send_weights_to_remote_instance(
                            "127.0.0.1", ports, "weight_send_group"
                        )
                        self.assertFalse(success)
                        self.assertEqual(
                            message,
                            f"Group {key} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first.",
                        )
                        exporter.get_model.assert_not_called()
                        exporter_mod.current_platform.empty_cache.assert_not_called()
                        self.assertEqual(exporter._weights_send_group, before)

    def test_existing_group_sends_weights_and_removes_only_that_group(self):
        exporter = self.make_exporter()
        model = torch.nn.Linear(2, 1)
        exporter.get_model.return_value = model
        key = "weight_send_group_29500_0"
        group, other = object(), object()
        exporter._weights_send_group.update({key: group, "unrelated": other})
        with (
            patch.object(torch.distributed, "broadcast") as broadcast,
            patch.object(
                torch.distributed.distributed_c10d, "destroy_process_group"
            ) as destroy,
        ):
            success, message = exporter.send_weights_to_remote_instance(
                "127.0.0.1", "29500", "weight_send_group"
            )
        self.assertTrue(success)
        self.assertIn("Succeeded to send weights", message)
        self.assertEqual(broadcast.call_count, len(list(model.parameters())))
        for recorded, parameter in zip(broadcast.call_args_list, model.parameters()):
            self.assertIs(recorded.args[0], parameter)
            self.assertEqual(recorded.kwargs, {"src": 0, "group": group})
        destroy.assert_called_once_with(group)
        self.assertEqual(exporter._weights_send_group, {"unrelated": other})

    def test_broadcast_failure_still_removes_group_and_returns_error(self):
        exporter = self.make_exporter()
        exporter.get_model.return_value = torch.nn.Linear(2, 1)
        key = "weight_send_group_29500_0"
        group = object()
        exporter._weights_send_group[key] = group
        with (
            patch.object(
                torch.distributed, "broadcast", side_effect=RuntimeError("send failed")
            ),
            patch.object(
                torch.distributed.distributed_c10d, "destroy_process_group"
            ) as destroy,
        ):
            success, message = exporter.send_weights_to_remote_instance(
                "127.0.0.1", "29500", "weight_send_group"
            )
        self.assertFalse(success)
        self.assertEqual(message, "Failed to send weights: send failed.")
        self.assertNotIn(key, exporter._weights_send_group)
        destroy.assert_called_once_with(group)


if __name__ == "__main__":
    unittest.main()
