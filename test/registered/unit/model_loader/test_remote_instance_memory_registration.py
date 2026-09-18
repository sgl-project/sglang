"""Check remote weight registration without requiring GPU hardware."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    register_memory_region,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRemoteInstanceMemoryRegistration(unittest.TestCase):
    def setUp(self):
        # Model a caching allocator segment with non-weight data at its base,
        # separated weights, and two adjacent weight blocks.
        self.pool = torch.arange(64, dtype=torch.float32)
        self.model = torch.nn.Module()
        for name, start, end in (("a", 8, 16), ("b", 24, 32), ("c", 32, 40)):
            self.model.register_parameter(
                name, torch.nn.Parameter(self.pool[start:end])
            )
        self.base = self.pool.data_ptr()
        self.itemsize = self.pool.element_size()
        self.snapshot = [
            {
                "address": self.base,
                "total_size": self.pool.numel() * self.itemsize,
                "blocks": [
                    {
                        "address": self.base + start * self.itemsize,
                        "size": (end - start) * self.itemsize,
                        "state": state,
                    }
                    for start, end, state in (
                        (0, 8, "active_allocated"),
                        (8, 16, "active_allocated"),
                        (16, 24, "inactive"),
                        (24, 32, "active_allocated"),
                        (32, 40, "active_allocated"),
                        (40, 64, "inactive"),
                    )
                ],
            },
            {
                "address": 100,
                "total_size": 64,
                "blocks": [{"address": 100, "size": 64, "state": "active_allocated"}],
            },
        ]

    def _register(self, hip, engine):
        with (
            patch.object(torch.version, "hip", hip),
            patch.object(
                torch.cuda.memory, "memory_snapshot", return_value=self.snapshot
            ),
        ):
            return register_memory_region(self.model, engine)

    def test_hip_registration_preserves_suballocation_offsets(self):
        engine = Mock()
        engine.register_memory.return_value = 0
        metadata = self._register("7.2", engine)
        regions = [call.args for call in engine.register_memory.call_args_list]

        for name, weight in self.model.named_parameters():
            with self.subTest(name=name):
                ptr, numel, itemsize = metadata[name]
                registered_base = next(
                    start
                    for start, length in regions
                    if start <= ptr and ptr + numel * itemsize <= start + length
                )
                # HIP IPC exports the entire allocation. Mooncake relocates a
                # remote pointer as ptr - registered_base + imported_base.
                # Registering a suballocation loses its offset and reads the
                # allocation's first bytes instead of the requested weight.
                offset = (ptr - registered_base) // self.itemsize
                received = self.pool[offset : offset + numel]
                torch.testing.assert_close(received, weight)

        engine.register_memory.assert_called_once_with(
            self.base, self.pool.numel() * self.itemsize
        )

    def test_cuda_registers_only_weight_blocks(self):
        engine = Mock()
        engine.register_memory.return_value = 0
        self._register(None, engine)
        self.assertEqual(
            [call.args for call in engine.register_memory.call_args_list],
            [
                (self.base + 8 * self.itemsize, 8 * self.itemsize),
                (self.base + 24 * self.itemsize, 16 * self.itemsize),
            ],
        )

    def test_registration_failure_is_reported(self):
        engine = Mock()
        engine.register_memory.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "register memory failed"):
            self._register("7.2", engine)


if __name__ == "__main__":
    unittest.main()
