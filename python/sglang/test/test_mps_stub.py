import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_mps_stub():
    module_name = "sglang._mps_stub_test"
    module_path = Path(__file__).resolve().parents[1] / "_mps_stub.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mps_stub = _load_mps_stub()


class TestMpsMemoryHelpers(unittest.TestCase):
    def test_get_available_metal_memory_subtracts_driver_usage(self):
        with (
            mock.patch("torch.mps.recommended_max_memory", return_value=100),
            mock.patch("torch.mps.driver_allocated_memory", return_value=30),
        ):
            self.assertEqual(mps_stub._get_available_metal_memory(), 70)

    def test_get_available_metal_memory_clamps_to_zero(self):
        with (
            mock.patch("torch.mps.recommended_max_memory", return_value=100),
            mock.patch("torch.mps.driver_allocated_memory", return_value=130),
        ):
            self.assertEqual(mps_stub._get_available_metal_memory(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
