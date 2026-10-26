"""Skip globally unchanged NPU layers without dropping remote senders."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_location_updater as updater
from sglang.srt.hardware_backend.npu import eplb as npu_eplb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuUnchangedMigration(unittest.TestCase):
    def run_update(self, old, new, *, elastic=None, device="npu"):
        metadata = lambda mapping: SimpleNamespace(
            num_local_physical_experts=4,
            physical_to_logical_map_cpu=torch.tensor(mapping),
        )
        weights = {layer: [torch.full((4, 2), layer)] for layer in range(len(old))}
        with (
            patch.object(
                npu_eplb, "get_device", return_value=SimpleNamespace(device=device)
            ),
            patch.object(
                updater.ElasticEPStateManager, "instance", return_value=elastic
            ),
            patch.object(updater.torch.distributed, "get_world_size", return_value=2),
            patch.object(
                updater, "create_temp_buffers", wraps=updater.create_temp_buffers
            ) as buffers,
            patch.object(updater, "update_expert_weights_single_layer") as migrate,
        ):
            result = updater._update_expert_weights_raw(
                weights, metadata(old), metadata(new), list(weights), 1, 0
            )
        self.assertEqual(result, {})
        return buffers, migrate

    def test_globally_unchanged_layers_allocate_and_migrate_nothing(self):
        mapping = [[0, 0, 1, 1, 2, 2, 3, 3]] * 2
        buffers, migrate = self.run_update(mapping, mapping)
        buffers.assert_not_called()
        migrate.assert_not_called()

    def test_unchanged_local_rank_still_participates_as_sender(self):
        old = [[0, 0, 1, 1, 2, 2, 3, 3]] * 2
        new = [old[0], [0, 0, 1, 1, 0, 2, 3, 3]]
        _, migrate = self.run_update(old, new)
        migrate.assert_called_once()
        args = migrate.call_args.kwargs
        self.assertEqual(
            args["old_physical_to_logical_map"][:4],
            args["new_physical_to_logical_map"][:4],
        )
        self.assertNotEqual(
            args["old_physical_to_logical_map"], args["new_physical_to_logical_map"]
        )
        self.assertEqual(args["routed_experts_weights"][0][0, 0].item(), 1)

    def test_elastic_recovery_and_other_devices_keep_existing_path(self):
        mapping = [[0, 0, 1, 1, 2, 2, 3, 3]]
        for options in ({"elastic": object()}, {"device": "cuda"}, {"device": "cpu"}):
            with self.subTest(options=options):
                _, migrate = self.run_update(mapping, mapping, **options)
                migrate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
