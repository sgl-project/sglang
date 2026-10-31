"""Exercise the real allocation policy with CPU-only recording mocks."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/utils/offloader.py"


def offload(device, platform_pin=False, budget=24):
    # Extract the real class so this policy check needs neither torch nor an NPU.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OffloaderV1"
    )
    cpu_data = Mock(numel=Mock(return_value=6), element_size=Mock(return_value=4))
    allocate = Mock(return_value=cpu_data)
    namespace = {
        "BaseOffloader": object,
        "is_pin_memory_available": lambda: platform_pin,
        # Isolate allocation policy from model-specific residency classification.
        "_get_resident_parameter_ids": lambda module: set(),
        "torch": SimpleNamespace(
            device=lambda name: SimpleNamespace(type=name), empty_strided=allocate
        ),
    }
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    data = Mock(
        size=Mock(return_value=(2, 3)),
        stride=Mock(return_value=(1, 2)),
        dtype="float32",
        layout="strided",
    )
    parameter = SimpleNamespace(device=SimpleNamespace(type=device), data=data)
    module = Mock(parameters=Mock(side_effect=lambda: iter([parameter])))
    namespace["OffloaderV1"](budget).maybe_offload_to_cpu(module)
    return allocate, parameter, data


class TestCpuOffloadPolicy(unittest.TestCase):
    def test_npu_requests_pinned_storage(self):
        for platform_pin in (False, True):
            with self.subTest(platform_pin=platform_pin):
                allocate, parameter, original = offload("npu", platform_pin)
                allocate.assert_called_once_with(
                    size=(2, 3),
                    stride=(1, 2),
                    dtype="float32",
                    layout="strided",
                    device="cpu",
                    pin_memory=True,
                )
                self.assertIs(parameter.data, allocate.return_value)
                parameter.data.copy_.assert_called_once_with(original)

    def test_non_npu_keeps_platform_policy(self):
        for device in ("cuda", "xpu"):
            for platform_pin in (False, True):
                with self.subTest(device=device, platform_pin=platform_pin):
                    allocate, _, _ = offload(device, platform_pin)
                    self.assertEqual(
                        allocate.call_args.kwargs["pin_memory"], platform_pin
                    )

    def test_no_offload_keeps_storage(self):
        for device, budget in (("npu", 0), ("cpu", 24)):
            with self.subTest(device=device, budget=budget):
                allocate, parameter, original = offload(device, budget=budget)
                allocate.assert_not_called()
                self.assertIs(parameter.data, original)


if __name__ == "__main__":
    unittest.main()
