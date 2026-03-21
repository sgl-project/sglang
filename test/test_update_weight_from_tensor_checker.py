import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, distribute_tensor

UTIL_PATH = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "runtime"
    / "utils"
    / "update_weight_from_tensor_checker.py"
)


def _install_layerwise_offload_stub():
    package_names = [
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.utils",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in package_names
    }
    for name in package_names:
        sys.modules.setdefault(name, types.ModuleType(name))

    layerwise_offload = types.ModuleType(
        "sglang.multimodal_gen.runtime.utils.layerwise_offload"
    )

    def iter_materialized_weights(module: nn.Module):
        yield from module.named_parameters()

    layerwise_offload.iter_materialized_weights = iter_materialized_weights
    original_modules[layerwise_offload.__name__] = sys.modules.get(
        layerwise_offload.__name__
    )
    sys.modules[layerwise_offload.__name__] = layerwise_offload
    return original_modules


def _load_checker_module():
    original_modules = _install_layerwise_offload_stub()
    try:
        spec = importlib.util.spec_from_file_location(
            "update_weight_from_tensor_checker_test_module",
            UTIL_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


_CHECKER_MODULE = _load_checker_module()
UpdateWeightFromTensorChecker = _CHECKER_MODULE.UpdateWeightFromTensorChecker
build_named_tensor_sha256 = _CHECKER_MODULE.build_named_tensor_sha256


class _ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_buffer("running_scale", torch.ones(4))


class _FakePipeline:
    def __init__(self, transformer: nn.Module):
        self._transformer = transformer

    def get_module(self, name: str):
        if name == "transformer":
            return self._transformer
        return None


def _iter_named_tensors(module: nn.Module):
    yield from module.named_parameters()
    yield from module.named_buffers()


class UpdateWeightFromTensorCheckerTest(unittest.TestCase):
    def test_matches_live_transformer(self):
        transformer = _ToyTransformer()
        checker = UpdateWeightFromTensorChecker(_FakePipeline(transformer))
        expected_transformer_sha256 = build_named_tensor_sha256(
            _iter_named_tensors(transformer)
        )

        success, message = checker.verify(expected_transformer_sha256)

        self.assertTrue(success)
        self.assertEqual(message, "Verified transformer update for 2 tensor(s).")

    def test_detects_modified_tensor(self):
        transformer = _ToyTransformer()
        checker = UpdateWeightFromTensorChecker(_FakePipeline(transformer))
        expected_transformer_sha256 = build_named_tensor_sha256(
            _iter_named_tensors(transformer)
        )

        with torch.no_grad():
            transformer.linear.weight.add_(1)

        success, message = checker.verify(expected_transformer_sha256)

        self.assertFalse(success)
        self.assertIn("checksum mismatch for 1 tensor(s): linear.weight", message)

    def test_detects_missing_tensor(self):
        transformer = _ToyTransformer()
        checker = UpdateWeightFromTensorChecker(_FakePipeline(transformer))

        success, message = checker.verify({"missing.weight": "deadbeef"})

        self.assertFalse(success)
        self.assertEqual(
            message,
            "Transformer update weight check failed: "
            "missing 1 tensor(s): missing.weight",
        )

    def test_supports_dtensor(self):
        if dist.is_initialized():
            self.skipTest("process group already initialized")

        with tempfile.NamedTemporaryFile() as rendezvous_file:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{rendezvous_file.name}",
                rank=0,
                world_size=1,
            )
            try:
                local_tensor = torch.arange(4, dtype=torch.float32)
                device_mesh = init_device_mesh("cpu", (1,))
                distributed_tensor = distribute_tensor(
                    local_tensor, device_mesh, [Replicate()]
                )

                expected_sha256 = build_named_tensor_sha256([("weight", local_tensor)])
                actual_sha256 = build_named_tensor_sha256(
                    [("weight", distributed_tensor)]
                )

                self.assertEqual(actual_sha256, expected_sha256)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
