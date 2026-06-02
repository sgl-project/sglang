import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_precision_module():
    """Load precision.py for both CI and lightweight local unittest runs.

    Prefer the normal SGLang package import. If optional runtime dependencies are
    missing in a lightweight local venv, fall back to loading the leaf module
    directly and stubbing only the precision string mapping it depends on.
    """

    try:
        from sglang.multimodal_gen.runtime import precision

        return precision
    except ModuleNotFoundError:
        for module_name in list(sys.modules):
            if module_name == "sglang" or module_name.startswith("sglang."):
                del sys.modules[module_name]

    utils_module = types.ModuleType("sglang.multimodal_gen.utils")
    utils_module.PRECISION_TO_TYPE = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    for package_name in (
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
    ):
        package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
        package.__path__ = []
    sys.modules["sglang.multimodal_gen.utils"] = utils_module

    precision_path = (
        Path(__file__).resolve().parents[1]
        / "python/sglang/multimodal_gen/runtime/precision.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sglang.multimodal_gen.runtime.precision", precision_path
    )
    precision = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = precision
    spec.loader.exec_module(precision)
    return precision


precision = _load_precision_module()
align_tensor_to_module_dtype = precision.align_tensor_to_module_dtype
precision_cache_key = precision.precision_cache_key
resolve_component_precision = precision.resolve_component_precision
resolve_precision = precision.resolve_precision
temporary_module_dtype = precision.temporary_module_dtype


class _ReplacingMixedDtypeModule(torch.nn.Module):
    """Module that changes Parameter identities during `.to()`.

    This makes `temporary_module_dtype` restoration deterministic to test: an
    implementation keyed by `id(param)` would fail because the names remain
    stable while Parameter objects are replaced.
    """

    def __init__(self):
        super().__init__()
        self.fp16 = torch.nn.Linear(2, 2).to(dtype=torch.float16)
        self.fp32 = torch.nn.LayerNorm(2).to(dtype=torch.float32)
        self.register_buffer("scale", torch.ones(2, dtype=torch.float32))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for child in (self.fp16, self.fp32):
            for name, param in list(child.named_parameters(recurse=False)):
                setattr(
                    child,
                    name,
                    torch.nn.Parameter(
                        param.detach().clone(), requires_grad=param.requires_grad
                    ),
                )
        self.scale = self.scale.detach().clone()
        return self


class TestDiffusionPrecisionConsistency(unittest.TestCase):
    def test_precision_consistency_contract(self):
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                vae_precision="fp16",
                audio_vae_precision="bf16",
                dit_precision="fp32",
                image_encoder_precision="fp16",
                text_encoder_precisions=["fp16", "bf16"],
            )
        )

        with self.subTest("user policy and execution constraint are explicit"):
            vae_spec = resolve_precision(
                server_args, "vae", precision_attr="vae_precision"
            )
            constraint_spec = resolve_precision(
                server_args,
                "denoising",
                constraint_dtype=torch.bfloat16,
                constraint_reason="kernel validated on bf16",
            )

            self.assertEqual(vae_spec.dtype, torch.float16)
            self.assertTrue(vae_spec.is_user_policy)
            self.assertEqual(
                vae_spec.reason, "server_args.pipeline_config.vae_precision"
            )
            self.assertEqual(constraint_spec.dtype, torch.bfloat16)
            self.assertFalse(constraint_spec.is_user_policy)
            self.assertEqual(constraint_spec.reason, "kernel validated on bf16")

        with self.subTest("component names map to centralized precision policy"):
            expected = {
                "vae": torch.float16,
                "audio_vae": torch.bfloat16,
                "vocoder": torch.bfloat16,
                "transformer": torch.float32,
                "transformer_2": torch.float32,
                "connectors": torch.float32,
                "dual_tower_bridge": torch.float32,
                "image_encoder": torch.float16,
                "text_encoder": torch.float16,
                "text_encoder_2": torch.bfloat16,
            }

            for module_name, expected_dtype in expected.items():
                spec = resolve_component_precision(server_args, module_name)
                self.assertIsNotNone(spec)
                self.assertEqual(spec.dtype, expected_dtype, module_name)
                self.assertTrue(spec.is_user_policy, module_name)

            for invalid_module_name in ("text_encoder_0", "text_encoder_3", "text_encoder_20"):
                with self.assertRaises(ValueError, msg=invalid_module_name):
                    resolve_component_precision(server_args, invalid_module_name)

        with self.subTest("tensor/module alignment prevents dtype mismatch safely"):
            module = torch.nn.Conv2d(3, 3, kernel_size=1).to(dtype=torch.float16)
            float_input = torch.ones((1, 3, 4, 4), dtype=torch.float32)
            int_input = torch.ones((1, 3), dtype=torch.long)

            aligned_float = align_tensor_to_module_dtype(float_input, module)
            aligned_int = align_tensor_to_module_dtype(int_input, module)

            self.assertEqual(aligned_float.dtype, torch.float16)
            self.assertEqual(aligned_float.device, next(module.parameters()).device)
            self.assertEqual(aligned_int.dtype, torch.long)
            self.assertEqual(aligned_int.device, next(module.parameters()).device)

        with self.subTest("temporary module dtype restores mixed dtypes by name"):
            module = _ReplacingMixedDtypeModule()

            with temporary_module_dtype(module, torch.bfloat16):
                self.assertEqual(module.fp16.weight.dtype, torch.bfloat16)
                self.assertEqual(module.fp32.weight.dtype, torch.bfloat16)
                self.assertEqual(module.scale.dtype, torch.bfloat16)

            self.assertEqual(module.fp16.weight.dtype, torch.float16)
            self.assertEqual(module.fp32.weight.dtype, torch.float32)
            self.assertEqual(module.scale.dtype, torch.float32)

        with self.subTest("cache key separates same path by device and dtype"):
            fp16_key = precision_cache_key("model.pth", "cpu", torch.float16)
            fp32_key = precision_cache_key("model.pth", "cpu", torch.float32)
            cuda_key = precision_cache_key("model.pth", "cuda:0", torch.float16)

            self.assertNotEqual(fp16_key, fp32_key)
            self.assertNotEqual(fp16_key, cuda_key)


if __name__ == "__main__":
    unittest.main()
