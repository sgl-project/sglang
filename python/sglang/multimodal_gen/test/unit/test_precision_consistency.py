import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_precision_module():
    """Load precision.py for both CI and lightweight local unittest runs.

    Load the leaf module directly and stub only the precision string mapping it
    depends on. Importing ``sglang.multimodal_gen`` normally pulls in optional
    runtime dependencies such as diffusers/torchao/flashinfer, which makes this
    focused precision contract test sensitive to unrelated environment issues.
    """

    stub_names = (
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
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
        for package_name in stub_names[:-1]:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
        sys.modules["sglang.multimodal_gen.utils"] = utils_module

        precision_path = Path(__file__).resolve().parents[2] / "runtime/precision.py"
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
if not hasattr(precision, "logger"):
    precision.logger = logging.getLogger(precision.__name__)
align_tensor_to_module_dtype = precision.align_tensor_to_module_dtype
autocast_enabled = precision.autocast_enabled
get_module_dtype = precision.get_module_dtype
precision_from_string = precision.precision_from_string
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


class _DtypedNoParameterModule(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype


class _ParameterDtypeWinsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))


class _RecordingVAE(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        self.seen_input_dtype = None
        self.seen_module_dtype = None

    def encode(self, tensor: torch.Tensor):
        self._record_call(tensor)
        return SimpleNamespace(mean=tensor)

    def decode(self, tensor: torch.Tensor):
        self._record_call(tensor)
        return tensor

    def _record_call(self, tensor: torch.Tensor):
        self.seen_input_dtype = tensor.dtype
        self.seen_module_dtype = next(self.parameters()).dtype


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

    def test_precision_consistency_contract(self):
        server_args = self._server_args()

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
                spec = resolve_component_precision(server_args, module_name)
                self.assertIsNotNone(spec)
                self.assertEqual(spec.dtype, expected_dtype, module_name)
                self.assertTrue(spec.is_user_policy, module_name)

            for invalid_module_name in (
                "text_encoder_0",
                "text_encoder_3",
                "text_encoder_20",
            ):
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
            torch_device_key = precision_cache_key(
                "model.pth", torch.device("cpu"), torch.float16
            )

            self.assertNotEqual(fp16_key, fp32_key)
            self.assertNotEqual(fp16_key, cuda_key)
            self.assertEqual(fp16_key, torch_device_key)

        with self.subTest("unquantized DiT dtype comes from dit_precision policy"):
            expected = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            for dit_precision, expected_dtype in expected.items():
                spec = resolve_precision(
                    self._server_args(dit_precision=dit_precision),
                    "dit",
                    precision_attr="dit_precision",
                )
                self.assertEqual(spec.dtype, expected_dtype)
                self.assertTrue(spec.is_user_policy)

    def test_component_precision_edge_cases(self):
        with self.subTest("component resolution is optional without pipeline config"):
            self.assertIsNone(resolve_component_precision(SimpleNamespace(), "vae"))

        with self.subTest("known component without its config attr falls back safely"):
            missing_vae_config = SimpleNamespace(
                pipeline_config=SimpleNamespace(
                    audio_vae_precision="bf16",
                    dit_precision="fp32",
                    image_encoder_precision="fp16",
                    text_encoder_precisions=["fp16"],
                )
            )
            self.assertIsNone(resolve_component_precision(missing_vae_config, "vae"))

        with self.subTest("unknown model components warn once and do not crash"):
            component_name = "unregistered_component_for_precision_test"
            with self.assertLogs(precision.logger, level="WARNING") as logs:
                self.assertIsNone(
                    resolve_component_precision(self._server_args(), component_name)
                )
                self.assertIsNone(
                    resolve_component_precision(self._server_args(), component_name)
                )
            self.assertEqual(len(logs.output), 1)
            self.assertIn(component_name, logs.output[0])

        with self.subTest("components explicitly without precision policy stay quiet"):
            for component_name in (
                "image_processor",
                "processor",
                "scheduler",
                "tokenizer",
                "vision_language_encoder",
            ):
                self.assertIsNone(
                    resolve_component_precision(self._server_args(), component_name)
                )

        with self.subTest("unsupported precision strings fail loudly"):
            with self.assertRaisesRegex(ValueError, "Unsupported vae_precision"):
                resolve_precision(self._server_args(vae_precision="fp8"), "vae")
            with self.assertRaisesRegex(ValueError, "Unsupported precision"):
                precision_from_string("vae", "fp8")
            with self.assertRaisesRegex(ValueError, r"text_encoder_precisions\[0\]"):
                resolve_component_precision(
                    self._server_args(text_encoder_precisions=["fp8"]),
                    "text_encoder",
                )

    def test_text_encoder_precision_policy_edge_cases(self):
        with self.subTest("numbered text encoders are one-indexed"):
            server_args = self._server_args(
                text_encoder_precisions=["fp16", "bf16", "fp32"]
            )

            first_encoder = resolve_component_precision(server_args, "text_encoder_1")
            default_encoder = resolve_component_precision(server_args, "text_encoder")
            third_encoder = resolve_component_precision(server_args, "text_encoder_3")
            self.assertEqual(first_encoder.dtype, torch.float16)
            self.assertEqual(default_encoder.dtype, torch.float16)
            self.assertEqual(third_encoder.dtype, torch.float32)
            self.assertEqual(
                first_encoder.reason,
                "server_args.pipeline_config.text_encoder_precisions[0]",
            )
            self.assertEqual(
                third_encoder.reason,
                "server_args.pipeline_config.text_encoder_precisions[2]",
            )

        with self.subTest("text encoder policy is optional when no precisions exist"):
            for precisions in (None, []):
                server_args = self._server_args(text_encoder_precisions=precisions)
                self.assertIsNone(
                    resolve_component_precision(server_args, "text_encoder")
                )
                self.assertIsNone(
                    resolve_component_precision(server_args, "text_encoder_99")
                )

        with self.subTest("malformed text encoder component names fail loudly"):
            server_args = self._server_args(text_encoder_precisions=["fp16"])
            for module_name in (
                "text_encoder_",
                "text_encoder_0",
                "text_encoder_00",
                "text_encoder_aux",
                "text_encoder_two",
            ):
                with self.assertRaises(ValueError, msg=module_name):
                    resolve_component_precision(server_args, module_name)

        with self.subTest("precision_from_string preserves explicit reason"):
            spec = precision_from_string(
                "standalone_component", "bf16", reason="test selected bf16"
            )
            self.assertEqual(spec.dtype, torch.bfloat16)
            self.assertTrue(spec.is_user_policy)
            self.assertEqual(spec.reason, "test selected bf16")

    def test_autocast_and_dtype_alignment_contract(self):
        with self.subTest("autocast is enabled only for non-fp32 user dtypes"):
            self.assertTrue(autocast_enabled(torch.float16, disable_autocast=False))
            self.assertTrue(autocast_enabled(torch.bfloat16, disable_autocast=False))
            self.assertFalse(autocast_enabled(torch.float32, disable_autocast=False))
            self.assertFalse(autocast_enabled(torch.float16, disable_autocast=True))

        with self.subTest("parameter dtype takes precedence over module dtype attr"):
            module = _ParameterDtypeWinsModule()
            self.assertEqual(get_module_dtype(module), torch.float16)
            aligned = align_tensor_to_module_dtype(
                torch.ones(1, dtype=torch.float32), module
            )
            self.assertEqual(aligned.dtype, torch.float16)

        with self.subTest("module dtype attr is used when there are no parameters"):
            module = _DtypedNoParameterModule(torch.bfloat16)
            self.assertEqual(get_module_dtype(module), torch.bfloat16)
            aligned = align_tensor_to_module_dtype(
                torch.ones(1, dtype=torch.float32), module
            )
            self.assertEqual(aligned.dtype, torch.bfloat16)

        with self.subTest("non-floating tensors are moved but never dtype-cast"):
            module = _DtypedNoParameterModule(torch.float16)
            tokens = torch.ones(2, dtype=torch.long)
            aligned = align_tensor_to_module_dtype(tokens, module)
            self.assertEqual(aligned.dtype, torch.long)

        with self.subTest("complex tensors follow module dtype policy"):
            module = _DtypedNoParameterModule(torch.float32)
            values = torch.ones(2, dtype=torch.complex64)
            with self.assertWarns(UserWarning):
                aligned = align_tensor_to_module_dtype(values, module)
            self.assertEqual(aligned.dtype, torch.float32)

        with self.subTest(
            "explicit device and default dtype cover parameterless modules"
        ):
            module = torch.nn.Module()
            values = torch.ones(2, dtype=torch.float16)
            aligned = align_tensor_to_module_dtype(
                values, module, device="cpu", default_dtype=torch.bfloat16
            )
            self.assertEqual(aligned.dtype, torch.bfloat16)
            self.assertEqual(aligned.device.type, "cpu")

    def test_stage_like_vae_precision_transitions(self):
        def run_vae_call(
            *,
            method_name: str,
            initial_module_dtype: torch.dtype,
            configured_precision: str,
        ):
            vae = _RecordingVAE(initial_module_dtype)
            vae_precision = resolve_precision(
                self._server_args(vae_precision=configured_precision),
                "vae",
                precision_attr="vae_precision",
            )
            vae_autocast_enabled = autocast_enabled(
                vae_precision.dtype, disable_autocast=True
            )
            tensor = torch.ones(1, 1, dtype=torch.float32)
            if not vae_autocast_enabled:
                tensor = tensor.to(vae_precision.dtype)
            should_cast_vae = vae_precision.is_user_policy and not vae_autocast_enabled

            with temporary_module_dtype(
                vae, vae_precision.dtype, enabled=should_cast_vae
            ) as casted_vae:
                getattr(casted_vae, method_name)(tensor)

            return vae

        with self.subTest("decode path casts fp32 module/input to configured bf16"):
            vae = run_vae_call(
                method_name="decode",
                initial_module_dtype=torch.float32,
                configured_precision="bf16",
            )
            self.assertEqual(vae.seen_input_dtype, torch.bfloat16)
            self.assertEqual(vae.seen_module_dtype, torch.bfloat16)
            self.assertEqual(vae.weight.dtype, torch.float32)

        with self.subTest("encode path casts fp16 module/input to configured fp32"):
            vae = run_vae_call(
                method_name="encode",
                initial_module_dtype=torch.float16,
                configured_precision="fp32",
            )
            self.assertEqual(vae.seen_input_dtype, torch.float32)
            self.assertEqual(vae.seen_module_dtype, torch.float32)
            self.assertEqual(vae.weight.dtype, torch.float16)

    def test_temporary_module_dtype_disable_and_restore_contract(self):
        with self.subTest("disabled context does not mutate dtype"):
            module = torch.nn.Linear(2, 2).to(dtype=torch.float32)
            with temporary_module_dtype(module, torch.float16, enabled=False) as casted:
                self.assertIs(casted, module)
                self.assertEqual(module.weight.dtype, torch.float32)
            self.assertEqual(module.weight.dtype, torch.float32)

        with self.subTest("gradients are restored with parameters"):
            module = torch.nn.Linear(2, 2).to(dtype=torch.float32)
            module.weight.grad = torch.ones_like(module.weight)
            module.bias.grad = torch.ones_like(module.bias)

            with temporary_module_dtype(module, torch.bfloat16):
                self.assertEqual(module.weight.dtype, torch.bfloat16)
                self.assertEqual(module.weight.grad.dtype, torch.bfloat16)

            self.assertEqual(module.weight.dtype, torch.float32)
            self.assertEqual(module.bias.dtype, torch.float32)
            self.assertEqual(module.weight.grad.dtype, torch.float32)
            self.assertEqual(module.bias.grad.dtype, torch.float32)

        with self.subTest("restore_dtype intentionally collapses module dtype"):
            module = _ReplacingMixedDtypeModule()
            with temporary_module_dtype(
                module, torch.bfloat16, restore_dtype=torch.float32
            ):
                self.assertEqual(module.fp16.weight.dtype, torch.bfloat16)
            self.assertEqual(module.fp16.weight.dtype, torch.float32)
            self.assertEqual(module.fp32.weight.dtype, torch.float32)

        with self.subTest("integer buffers survive temporary dtype changes"):
            module = torch.nn.Module()
            module.register_buffer("token_ids", torch.ones(3, dtype=torch.long))

            with temporary_module_dtype(module, torch.bfloat16):
                self.assertEqual(module.token_ids.dtype, torch.long)

            self.assertEqual(module.token_ids.dtype, torch.long)

        with self.subTest("floating buffers restore by name alongside integer buffers"):
            module = torch.nn.Module()
            module.register_buffer("scale", torch.ones(3, dtype=torch.float32))
            module.register_buffer("token_ids", torch.ones(3, dtype=torch.long))

            with temporary_module_dtype(module, torch.float16):
                self.assertEqual(module.scale.dtype, torch.float16)
                self.assertEqual(module.token_ids.dtype, torch.long)

            self.assertEqual(module.scale.dtype, torch.float32)
            self.assertEqual(module.token_ids.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
