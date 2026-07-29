import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_precision_module():
    package_names = (
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.utils",
    )
    stub_names = (
        *package_names,
        "sglang.multimodal_gen.runtime.platforms",
        "sglang.multimodal_gen.utils",
    )
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in stub_names}

    try:
        utils_module = types.ModuleType("sglang.multimodal_gen.utils")
        utils_module.PRECISION_TO_TYPE = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        platforms_module = types.ModuleType("sglang.multimodal_gen.runtime.platforms")
        platforms_module.current_platform = SimpleNamespace(
            device_type="cpu",
            is_mps=lambda: False,
            is_amp_supported=lambda: True,
        )
        for package_name in package_names:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
        sys.modules["sglang.multimodal_gen.runtime.platforms"] = platforms_module
        sys.modules["sglang.multimodal_gen.utils"] = utils_module

        precision_path = (
            Path(__file__).resolve().parents[2] / "runtime/utils/precision.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_diffusion_precision_under_test", precision_path
        )
        precision = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = precision
        spec.loader.exec_module(precision)
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

    return precision


precision = _load_precision_module()
align_tensor_to_module_dtype = precision.align_tensor_to_module_dtype
autocast_context = precision.autocast_context
autocast_enabled = precision.autocast_enabled
get_module_dtype = precision.get_module_dtype
precision_to_dtype = precision.precision_to_dtype
resolve_component_precision = precision.resolve_component_precision
resolve_precision = precision.resolve_precision
temporary_module_dtype = precision.temporary_module_dtype


class _DtypedNoParameterModule(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype


class _ParameterDtypeWinsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))


class _FakePlatform:
    def __init__(self, device_type: str, *, is_mps: bool, amp_supported: bool):
        self.device_type = device_type
        self._is_mps = is_mps
        self._amp_supported = amp_supported

    def is_mps(self):
        return self._is_mps

    def is_amp_supported(self):
        return self._amp_supported


class TestDiffusionPrecisionConsistency(unittest.TestCase):
    def _server_args(self, **overrides):
        config = {
            "vae_precision": "fp16",
            "audio_vae_precision": "bf16",
            "dit_precision": "fp32",
            "image_encoder_precision": "fp16",
            "text_encoder_precisions": ["fp16", "bf16"],
        }
        config.update(overrides)
        return SimpleNamespace(pipeline_config=SimpleNamespace(**config))

    def test_precision_lookup(self):
        server_args = self._server_args()

        self.assertEqual(
            resolve_precision(server_args, "vae", precision_attr="vae_precision"),
            torch.float16,
        )
        self.assertEqual(
            resolve_precision(server_args, "dit", precision_attr="dit_precision"),
            torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported vae_precision"):
            resolve_precision(self._server_args(vae_precision="fp8"), "vae_precision")
        with self.assertRaisesRegex(ValueError, "Unsupported custom_precision"):
            precision_to_dtype("fp8", "custom_precision")

    def test_component_precision_mapping(self):
        server_args = self._server_args()
        expected = {
            "vae": torch.float16,
            "video_vae": torch.float16,
            "audio_vae": torch.bfloat16,
            "vocoder": torch.bfloat16,
            "transformer": torch.float32,
            "transformer_2": torch.float32,
            "audio_dit": torch.float32,
            "video_dit": torch.float32,
            "connectors": torch.float32,
            "dual_tower_bridge": torch.float32,
            "image_encoder": torch.float16,
            "text_encoder": torch.float16,
            "text_encoder_2": torch.bfloat16,
        }

        for module_name, expected_dtype in expected.items():
            self.assertEqual(
                resolve_component_precision(server_args, module_name),
                expected_dtype,
                module_name,
            )

        self.assertIsNone(resolve_component_precision(SimpleNamespace(), "vae"))
        self.assertIsNone(
            resolve_component_precision(server_args, "unregistered_component")
        )
        self.assertIsNone(
            resolve_component_precision(
                self._server_args(text_encoder_precisions=[]), "text_encoder"
            )
        )

    def test_autocast_and_dtype_alignment(self):
        self.assertTrue(autocast_enabled(torch.float16, disable_autocast=False))
        self.assertTrue(autocast_enabled(torch.bfloat16, disable_autocast=False))
        self.assertFalse(autocast_enabled(torch.float32, disable_autocast=False))
        self.assertFalse(autocast_enabled(torch.float16, disable_autocast=True))

        module = _ParameterDtypeWinsModule()
        self.assertEqual(get_module_dtype(module), torch.float16)
        aligned = align_tensor_to_module_dtype(
            torch.ones(1, dtype=torch.float32), module
        )
        self.assertEqual(aligned.dtype, torch.float16)

        module_without_parameters = _DtypedNoParameterModule(torch.bfloat16)
        self.assertEqual(get_module_dtype(module_without_parameters), torch.bfloat16)
        tokens = torch.ones(2, dtype=torch.long)
        aligned_tokens = align_tensor_to_module_dtype(tokens, module_without_parameters)
        self.assertEqual(aligned_tokens.dtype, torch.long)

    def test_autocast_context_honors_explicit_override(self):
        original_platform = precision.current_platform
        try:
            precision.current_platform = _FakePlatform(
                "cpu", is_mps=False, amp_supported=True
            )
            disabled_context = autocast_context(
                torch.bfloat16, disable_autocast=False, enabled=False
            )
            self.assertNotIsInstance(disabled_context, nullcontext)

            precision.current_platform = _FakePlatform(
                "mps", is_mps=True, amp_supported=False
            )
            mps_disabled_context = autocast_context(
                torch.bfloat16, disable_autocast=False, enabled=False
            )
            self.assertIsInstance(mps_disabled_context, nullcontext)
        finally:
            precision.current_platform = original_platform

    def test_temporary_module_dtype(self):
        module = torch.nn.Linear(2, 2).to(dtype=torch.float32)

        with temporary_module_dtype(module, torch.bfloat16):
            self.assertEqual(module.weight.dtype, torch.bfloat16)

        self.assertEqual(module.weight.dtype, torch.float32)

        with temporary_module_dtype(module, torch.float16, enabled=False) as casted:
            self.assertIs(casted, module)
            self.assertEqual(module.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
