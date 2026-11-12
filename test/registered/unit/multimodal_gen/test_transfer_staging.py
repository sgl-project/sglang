# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for aligned diffusion transfer staging."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTransferStaging(CustomTestCase):
    def setUp(self):
        super().setUp()
        empty = torch.empty

        def unpinned_empty(*args, **kwargs):
            # CPU-only PyTorch has no pinned allocator. Keep the real pool
            # storage, buddy allocator, tensor copies, and manifest IO.
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        for scope in (
            patch.object(torch, "empty", side_effect=unpinned_empty),
            patch.object(torch, "get_device_module", return_value=torch.cpu),
            patch.object(torch.cpu, "is_available", return_value=False),
        ):
            scope.start()
            self.addCleanup(scope.stop)

    def assert_staging_round_trip(
        self, fields, offsets, slot_size, min_block_size=1 << 20
    ):
        buffer = TransferTensorBuffer(pool_size=2 << 20, min_block_size=min_block_size)
        manager = DiffusionTransferManager(Mock(spec=BaseTransferEngine), buffer)
        staged, event = manager.stage_tensors_async("request", fields, {"step": 1})
        self.assertIsNotNone(staged)
        self.assertIsNone(event)
        self.assertEqual(staged.scalar_fields, {"step": 1})
        self.assertEqual(
            {
                name: [entry["offset"] for entry in entries]
                for name, entries in staged.manifest.items()
            },
            offsets,
        )
        if slot_size is None:
            self.assertIsNone(staged.slot)
        else:
            self.assertEqual(staged.slot.size, slot_size)
            restored = buffer.read_tensors_from_manifest(staged.slot, staged.manifest)
            for name, value in fields.items():
                if value is None or (isinstance(value, list) and not value):
                    self.assertNotIn(name, restored)
                else:
                    torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
            if "parts" in staged.manifest:
                self.assertEqual(
                    [entry["list_index"] for entry in staged.manifest["parts"]],
                    [i for i, value in enumerate(fields["parts"]) if value is not None],
                )
        manager.free_staged("request")
        # The full pool must be reusable after staging cleanup.
        whole_pool = buffer.allocate(buffer.pool_size, "next-request")
        self.assertIsNotNone(whole_pool)
        self.assertTrue(buffer.free(whole_pool))

    def test_alignment_crosses_default_block_boundary(self):
        """Raw bytes fit 1 MiB, but alignment pushes the second tensor past its slot."""
        self.assert_staging_round_trip(
            {
                "first": torch.full((1_048_065,), 17, dtype=torch.uint8),
                "second": torch.full((511,), 29, dtype=torch.uint8),
            },
            {"first": [0], "second": [1_048_576]},
            2 << 20,
        )

    def test_list_holes_preserve_indices_and_alignment(self):
        """Skipped entries must not break sizing or shift decoded list indices."""
        self.assert_staging_round_trip(
            {
                "absent": None,
                "empty": [],
                "parts": [None, torch.arange(3), None, torch.arange(2)],
                "tail": torch.tensor([7], dtype=torch.uint8),
            },
            {"parts": [0, 512], "tail": [1024]},
            2048,
            min_block_size=512,
        )

    def test_alignment_boundaries(self):
        """Exercise both sides of 512B, aligned tensors, and a zero-length tail."""
        for sizes, offsets, slot_size, minimum in (
            ((511, 512), [0, 512], 1024, 512),
            ((512, 512), [0, 512], 1024, 512),
            ((513, 1), [0, 1024], 2048, 512),
            ((3,), [0], 1 << 20, 1 << 20),
            ((1, 0), [0, 512], 512, 1),
        ):
            with self.subTest(sizes=sizes):
                self.assert_staging_round_trip(
                    {"parts": [torch.ones(size, dtype=torch.uint8) for size in sizes]},
                    {"parts": offsets},
                    slot_size,
                    min_block_size=minimum,
                )

    def test_absent_fields_need_no_slot(self):
        """Absent tensors and all-None lists stage only scalar metadata."""
        for fields in ({"absent": None, "empty": []}, {"parts": [None, None]}):
            with self.subTest(fields=fields):
                self.assert_staging_round_trip(fields, {}, None)


if __name__ == "__main__":
    unittest.main()
