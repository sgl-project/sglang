# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime import launch_server as launcher


class TestMonolithicGpuIds(unittest.TestCase):
    def _launch(self, gpu_ids, *, num_gpus=2, nnodes=1, node_rank=0):
        args = SimpleNamespace(
            enable_metrics=False,
            num_gpus=num_gpus,
            nnodes=nnodes,
            node_rank=node_rank,
            gpu_ids=gpu_ids,
        )
        workers = [Mock() for _ in range(num_gpus // nnodes)]
        with (
            patch.object(launcher, "apply_plugin_hooks"),
            patch.object(launcher, "configure_logger"),
            patch.object(
                launcher, "_start_scheduler_processes", return_value=workers
            ) as start,
            patch.object(launcher, "shutdown_scheduler_processes"),
        ):
            result = launcher.launch_server(args, launch_http_server=False)
        self.assertIs(result, workers)
        return start.call_args.args[1]

    def test_explicit_device_order_does_not_change_distributed_ranks(self):
        for gpu_ids in ([6], [6, 2], [7, 1, 5, 3]):
            with self.subTest(gpu_ids=gpu_ids):
                self.assertEqual(
                    self._launch(gpu_ids, num_gpus=len(gpu_ids)),
                    list(zip(gpu_ids, range(len(gpu_ids)))),
                )

    def test_device_ids_are_local_while_global_ranks_include_node_offset(self):
        for gpu_ids, expected_devices in ((None, [0, 1]), ([6, 2], [6, 2])):
            with self.subTest(gpu_ids=gpu_ids):
                self.assertEqual(
                    self._launch(gpu_ids, num_gpus=4, nnodes=2, node_rank=1),
                    list(zip(expected_devices, [2, 3])),
                )

    def test_invalid_device_count_fails_before_payload_or_workers_are_created(self):
        for gpu_ids in ([], [6], [6, 2, 3]):
            args = SimpleNamespace(
                enable_metrics=False, num_gpus=2, nnodes=1, node_rank=0, gpu_ids=gpu_ids
            )
            with (
                self.subTest(gpu_ids=gpu_ids),
                patch.object(launcher, "apply_plugin_hooks"),
                patch.object(launcher, "configure_logger"),
                patch.object(launcher.ServerArgsPayload, "capture") as capture,
                patch.object(launcher, "_start_scheduler_processes") as start,
            ):
                with self.assertRaisesRegex(ValueError, "--gpu-ids"):
                    launcher.launch_server(args, launch_http_server=False)
                capture.assert_not_called()
                start.assert_not_called()
