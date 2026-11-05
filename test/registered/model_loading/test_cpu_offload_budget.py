"""CPU offload must fit whole parameter storages within the shared byte budget."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.utils.offloader import OffloaderV1
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestCpuOffloadBudget(CustomTestCase):
    def test_skip_oversized_parameter_and_use_remaining_budget(self):
        for pin_memory in (False, True):
            with (
                self.subTest(pin_memory=pin_memory),
                patch(
                    "sglang.srt.utils.offloader.is_pin_memory_available",
                    return_value=pin_memory,
                ),
            ):
                # The 64-byte weight does not fit, but the 16-byte bias does.
                layer = nn.Linear(4, 4, device="cuda")
                inputs = torch.randn(2, 4, device="cuda")
                expected = layer(inputs)
                offloader = OffloaderV1(cpu_offload_max_bytes=16)

                offloader.maybe_offload_to_cpu(layer)

                self.assertEqual(layer.weight.device.type, "cuda")
                self.assertEqual(layer.bias.device.type, "cpu")
                self.assertEqual(layer.bias.is_pinned(), pin_memory)
                self.assertEqual(offloader._cpu_offload_bytes, 16)
                for _ in range(2):
                    torch.testing.assert_close(layer(inputs), expected)

    def test_budget_is_shared_across_modules(self):
        layers = [nn.Linear(2, 2, device="cuda") for _ in range(3)]
        inputs = torch.randn(2, 2, device="cuda")
        expected = [layer(inputs) for layer in layers]
        # First layer: 16-byte weight + 8-byte bias. Only the second bias fits.
        offloader = OffloaderV1(cpu_offload_max_bytes=32)

        offloader.wrap_modules(iter(layers))

        self.assertEqual(offloader._cpu_offload_bytes, 32)
        self.assertEqual(
            [[p.device.type for p in layer.parameters()] for layer in layers],
            [["cpu", "cpu"], ["cuda", "cpu"], ["cuda", "cuda"]],
        )
        for layer, reference in zip(layers, expected):
            torch.testing.assert_close(layer(inputs), reference)

    def test_parameter_that_does_not_fit_keeps_original_forward(self):
        for budget in (0, 15):
            with self.subTest(budget=budget):
                layer = nn.Linear(2, 2, bias=False, device="cuda")
                original_forward = layer.forward
                offloader = OffloaderV1(cpu_offload_max_bytes=budget)

                offloader.maybe_offload_to_cpu(layer)

                self.assertEqual(layer.weight.device.type, "cuda")
                self.assertEqual(layer.forward, original_forward)
                self.assertEqual(offloader._cpu_offload_bytes, 0)

    def test_strided_storage_budget(self):
        for dtype in (torch.float16, torch.float32):
            for stride in ((2, 1), (1, 2), (4, 1)):
                # The last layout has holes: 4 elements need 6 elements of storage.
                for fits in (False, True):
                    with self.subTest(dtype=dtype, stride=stride, fits=fits):
                        layer = nn.Linear(2, 2, bias=False, device="cuda", dtype=dtype)
                        weight = torch.empty_strided(
                            (2, 2), stride, device="cuda", dtype=dtype
                        )
                        weight.fill_(0.5)
                        layer.weight = nn.Parameter(weight)
                        storage_bytes = weight.untyped_storage().nbytes()
                        budget = storage_bytes if fits else storage_bytes - 1
                        inputs = torch.ones(1, 2, device="cuda", dtype=dtype)
                        expected = layer(inputs)
                        offloader = OffloaderV1(cpu_offload_max_bytes=budget)

                        offloader.maybe_offload_to_cpu(layer)

                        self.assertEqual(
                            layer.weight.device.type, "cpu" if fits else "cuda"
                        )
                        self.assertEqual(layer.weight.stride(), stride)
                        self.assertEqual(
                            offloader._cpu_offload_bytes, storage_bytes if fits else 0
                        )
                        torch.testing.assert_close(layer(inputs), expected)

    def test_storage_offset_does_not_charge_unused_prefix(self):
        layer = nn.Linear(2, 2, bias=False, device="cuda")
        # The CPU allocation starts at offset zero, without the CUDA prefix.
        layer.weight = nn.Parameter(torch.ones(8, device="cuda")[4:].view(2, 2))
        offloader = OffloaderV1(cpu_offload_max_bytes=16)

        offloader.maybe_offload_to_cpu(layer)

        self.assertEqual(layer.weight.device.type, "cpu")
        self.assertEqual(layer.weight.storage_offset(), 0)
        self.assertEqual(layer.weight.untyped_storage().nbytes(), 16)
        self.assertEqual(offloader._cpu_offload_bytes, 16)
        torch.testing.assert_close(
            layer(torch.ones(1, 2, device="cuda")),
            torch.full((1, 2), 2.0, device="cuda"),
        )


if __name__ == "__main__":
    unittest.main()
