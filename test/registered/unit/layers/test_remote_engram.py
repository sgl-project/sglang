import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch


def load_adapter():
    path = Path(__file__).parents[4] / "python/sglang/srt/layers/remote_engram.py"
    spec = importlib.util.spec_from_file_location("test_remote_engram_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoteEngramValidationTest(unittest.TestCase):
    def setUp(self):
        self.module = load_adapter()

    def test_checkpoint_shapes_and_dtypes(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        weight = torch.empty((0, 256), dtype=torch.float8_e4m3fn)
        scale = torch.empty((0, 8), dtype=torch.float8_e8m0fnu)
        table.check_weight(weight, torch.empty((4, 256), dtype=weight.dtype), weight)
        table.check_weight(scale, torch.empty((4, 8), dtype=scale.dtype), weight)
        self.assertEqual(table.loaded, {"weight", "scale"})
        with self.assertRaises(ValueError):
            table.check_weight(weight, torch.empty((3, 256), dtype=weight.dtype), weight)
        with self.assertRaises(ValueError):
            table.check_weight(scale, torch.empty((4, 8)), weight)

    def test_requires_both_checkpoint_components(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            table.finish_load(0, 4)

    def test_rejects_wrong_remote_table_size(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        table.loaded = {"weight", "scale"}
        self.module._runtime = mock.Mock(storage_layout="row264", metadata={"lengths": [1]})
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            table.finish_load(0, 4)

    def test_unsupported_layout_and_legacy_prefetch_rejected(self):
        with self.assertRaises(ValueError):
            self.module.RemoteEngramTable(4, 128, 1)
        with mock.patch.dict(os.environ, {"ENGRAM_SHARED_PREFETCH": "1"}):
            with self.assertRaisesRegex(ValueError, "prefetch"):
                self.module.RemoteEngramTable(4, 256, 1)


if __name__ == "__main__":
    unittest.main()
