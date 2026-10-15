"""CPU-only policy regression against the actual OffloaderV1 source AST.

Usage: python test_cpu_offload_policy.py --source /path/to/offloader.py
No torch/NPU imports. Tensor allocation is a recording boundary; the real
class controls selection, budget and parameter replacement. This does NOT
test real pinned allocation, copies, forward, or graph correctness.
"""

import argparse
import ast
import types
import unittest


class Device:
    def __init__(self, name):
        self.type = name

    def __eq__(self, other):
        return isinstance(other, Device) and self.type == other.type


class Tensor:
    dtype = "float32"
    layout = "strided"

    def __init__(self, device, size, stride, pinned=False):
        self.device = Device(device)
        self._size = size
        self._stride = stride
        self.pinned = pinned
        self.copied_from = None

    def size(self):
        return self._size

    def stride(self):
        return self._stride

    def numel(self):
        result = 1
        for n in self._size:
            result *= n
        return result

    def element_size(self):
        return 4

    def copy_(self, source):
        self.copied_from = source


class Parameter:
    def __init__(self, data):
        self.data = data

    @property
    def device(self):
        return self.data.device


class Module:
    def __init__(self, device):
        self.items = [
            Parameter(Tensor(device, (2, 3), (1, 2))),
            Parameter(Tensor(device, (2, 3), (3, 1))),
        ]
        self.forward = lambda *a, **k: None

    def parameters(self):
        return iter(self.items)


def load_class(source, platform_pin):
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OffloaderV1"
    )
    allocations = []

    def allocate(**kw):
        allocations.append(kw)
        return Tensor(kw["device"], kw["size"], kw["stride"], kw["pin_memory"])

    namespace = {
        "BaseOffloader": object,
        "torch": types.SimpleNamespace(device=Device, empty_strided=allocate),
        "is_pin_memory_available": lambda: platform_pin,
    }
    # Keep class/function code exactly from --source; postpone annotations only.
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(unit), "<actual OffloaderV1>", "exec"),
        namespace,
    )
    return namespace["OffloaderV1"], allocations


class PolicyTests(unittest.TestCase):
    source = ""

    def test_npu_pins_offloaded_storage_without_changing_layout(self):
        cls, allocations = load_class(self.source, False)
        module = Module("npu")
        original = module.items[0].data
        cls(24).maybe_offload_to_cpu(module)
        self.assertTrue(allocations[0]["pin_memory"])
        self.assertEqual(module.items[0].data.stride(), original.stride())
        self.assertIs(module.items[0].data.copied_from, original)

    def test_non_npu_preserves_platform_policy(self):
        for target in ("cuda", "xpu"):
            for available in (False, True):
                with self.subTest(target=target, available=available):
                    cls, allocations = load_class(self.source, available)
                    cls(24).maybe_offload_to_cpu(Module(target))
                    self.assertEqual(allocations[0]["pin_memory"], available)

    def test_budget_counts_existing_parameter_granularity_across_modules(self):
        cls, allocations = load_class(self.source, False)
        offloader = cls(25)  # Existing contract: whole parameter may cross cap.
        first, second = Module("npu"), Module("npu")
        offloader.maybe_offload_to_cpu(first)
        offloader.maybe_offload_to_cpu(second)
        self.assertEqual(len(allocations), 2)
        self.assertEqual(offloader._cpu_offload_bytes, 48)
        self.assertEqual([p.device.type for p in second.items], ["npu", "npu"])

    def test_zero_budget_and_cpu_module_leave_storage_untouched(self):
        for target, budget in (("npu", 0), ("cpu", 100)):
            cls, allocations = load_class(self.source, False)
            module = Module(target)
            old = [p.data for p in module.items]
            offloader = cls(budget)
            offloader.maybe_offload_to_cpu(module)
            self.assertEqual(allocations, [])
            self.assertEqual(offloader._cpu_offload_bytes, 0)
            self.assertEqual([p.data for p in module.items], old)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    PolicyTests.source = args.source.read_text(encoding="utf-8")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
