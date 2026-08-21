"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

partial_json_parser = types.ModuleType("partial_json_parser")
partial_json_parser_core = types.ModuleType("partial_json_parser.core")
partial_json_parser_exceptions = types.ModuleType("partial_json_parser.core.exceptions")
partial_json_parser_options = types.ModuleType("partial_json_parser.core.options")


class _MalformedJSON(Exception):
    pass


class _Allow:
    STR = 1
    OBJ = 2
    ARR = 4
    ALL = STR | OBJ | ARR


def _loads(input_str, _flags=None):
    return json.loads(input_str)


partial_json_parser_exceptions.MalformedJSON = _MalformedJSON
partial_json_parser_options.Allow = _Allow
partial_json_parser.loads = _loads
sys.modules.setdefault("partial_json_parser", partial_json_parser)
sys.modules.setdefault("partial_json_parser.core", partial_json_parser_core)
sys.modules.setdefault(
    "partial_json_parser.core.exceptions", partial_json_parser_exceptions
)
sys.modules.setdefault("partial_json_parser.core.options", partial_json_parser_options)

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import (
    ComfyFp8Config,
    ComfyFullPrecisionFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
    _prepare_nvfp4_weight_bytes,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    _default_quantized_attention_backend,
    _resolve_checkpoint_load_device,
    _warn_if_expected_param_dtype_missing,
)
from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
    inspect_minimax_h3_safetensors,
    resolve_minimax_h3_checkpoint_quantization,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    TransformerQuantLoadSpec,
    _filter_duplicate_precision_variant_safetensors,
    _Flux2Nvfp4FallbackAdapter,
    _needs_device_weight_postprocess,
    _resolve_quant_config,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.models.dits.flux import FluxSingleTransformerBlock
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    _resolve_quant_method_name,
    build_nvfp4_config_from_safetensors_list,
    get_quant_config,
)
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _updated_quant_config,
)


class _FakeFluxTransformer:
    pass


class _FakeQuantConfig:
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"


def _make_quant_config(name: str, **attrs):
    cls = type(
        f"_Fake{name.title().replace('_', '')}QuantConfig",
        (),
        {"get_name": classmethod(lambda cls: name)},
    )
    quant_config = cls()
    for attr_name, attr_value in attrs.items():
        setattr(quant_config, attr_name, attr_value)
    return quant_config


class TestTransformerQuantHelpers(unittest.TestCase):
    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(param_names_mapping={})
                ),
            ),
            nunchaku_config=None,
            quantization=None,
            quantization_ignored_layers=None,
            revision="test-revision",
            tp_size=1,
            dit_cpu_offload=False,
            direct_gpu_weight_loading=False,
            text_encoder_cpu_offload=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_modelopt_fp4_uses_fa_by_default_on_blackwell(self):
        quant_spec = TransformerQuantLoadSpec([], _FakeQuantConfig(), None, None)
        server_args = SimpleNamespace(attention_backend=None)

        with (
            patch.object(
                transformer_loader.current_platform, "is_blackwell", return_value=True
            ),
            patch.object(
                transformer_loader,
                "get_global_forced_attn_backend",
                return_value=None,
            ),
            patch.object(
                transformer_loader,
                "get_component_forced_attn_backend",
                return_value=None,
            ),
        ):
            backend = _default_quantized_attention_backend(quant_spec, server_args)

        self.assertEqual(backend, AttentionBackendEnum.FA)

    def test_modelopt_fp4_preserves_explicit_attention_backend(self):
        quant_spec = TransformerQuantLoadSpec([], _FakeQuantConfig(), None, None)
        server_args = SimpleNamespace(attention_backend="dynamic_cudnn_sdpa")

        with patch.object(
            transformer_loader.current_platform, "is_blackwell", return_value=True
        ):
            backend = _default_quantized_attention_backend(quant_spec, server_args)

        self.assertIsNone(backend)

    def test_resolve_transformer_safetensors_to_load_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [f.name])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.hf_hub_download",
        return_value="/cache/model.safetensors",
    )
    def test_resolve_transformer_safetensors_to_load_uses_hf_file_reference(
        self, mock_download
    ):
        filename = "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
        references = (
            (
                f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/{filename}",
                "main",
            ),
            (f"Comfy-Org/MiniMax-H3/{filename}", "test-revision"),
        )

        for reference, revision in references:
            with self.subTest(reference=reference):
                server_args = self._make_server_args(transformer_weights_path=reference)
                with patch(
                    "os.path.isfile",
                    side_effect=lambda path: path == "/cache/model.safetensors",
                ):
                    self.assertEqual(
                        resolve_transformer_safetensors_to_load(
                            server_args, "/unused/component/path"
                        ),
                        ["/cache/model.safetensors"],
                    )
                mock_download.assert_called_once_with(
                    repo_id="Comfy-Org/MiniMax-H3",
                    filename=filename,
                    revision=revision,
                )
                mock_download.reset_mock()

    def test_inspect_minimax_h3_safetensors_detects_curve_and_comfy_format(self):
        marker = json.dumps({"format": "int8_tensorwise", "convrot": True}).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(
                {
                    "adaln_t_table": torch.zeros((1025, 8)),
                    "blocks.0.mlp.fc1.comfy_quant": torch.tensor(
                        list(marker), dtype=torch.uint8
                    ),
                },
                f.name,
            )

            curve_shape, comfy_quant = inspect_minimax_h3_safetensors([f.name])

        self.assertEqual(curve_shape, (1025, 8))
        self.assertEqual(comfy_quant["blocks.0.mlp.fc1"]["format"], "int8_tensorwise")

    def test_inspect_minimax_h3_fp8_validates_required_scales(self):
        marker = torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8)
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(
                {
                    "blocks.0.mlp.fc1.weight": torch.ones(
                        (2, 2), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.mlp.fc1.weight_scale": torch.tensor(0.5),
                    "blocks.0.mlp.fc1.input_scale": torch.tensor(0.25),
                    "blocks.0.mlp.fc1.comfy_quant": marker,
                },
                f.name,
            )

            _, layer_markers = inspect_minimax_h3_safetensors([f.name])

        self.assertEqual(layer_markers["blocks.0.mlp.fc1"], {"format": "float8_e4m3fn"})

    def test_minimax_h3_comfy_int8_fails_before_weight_loading(self):
        with self.assertRaisesRegex(NotImplementedError, "regular-Hadamard"):
            resolve_minimax_h3_checkpoint_quantization(
                {
                    "blocks.0.mlp.fc1": {
                        "format": "int8_tensorwise",
                        "convrot": True,
                    }
                }
            )

    def test_minimax_h3_comfy_fp8_resolves_per_layer_dispatch(self):
        config = resolve_minimax_h3_checkpoint_quantization(
            {
                "blocks.0.attn.qkv_proj": {"format": "float8_e4m3fn"},
                "blocks.0.mlp.fc2": {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": True,
                },
            }
        )

        self.assertIsInstance(config, ComfyFp8Config)
        layer = LinearBase(input_size=1, output_size=1)
        self.assertIsInstance(
            config.get_quant_method(layer, "blocks.0.mlp.fc2"),
            ComfyFullPrecisionFp8LinearMethod,
        )
        self.assertIsInstance(
            config.get_quant_method(layer, "blocks.0.attn.qkv_proj"),
            Fp8LinearMethod,
        )
        self.assertIsInstance(
            config.get_quant_method(layer, "unmarked"),
            UnquantizedLinearMethod,
        )

    def test_comfy_full_precision_fp8_dequantizes_before_linear(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.tensor([[2.0, -4.0]], dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=False
        )

        output = ComfyFullPrecisionFp8LinearMethod().apply(
            layer, torch.tensor([[3.0, 1.0]])
        )

        torch.testing.assert_close(output, torch.tensor([[1.0]]))

    def test_checkpoint_quantization_metadata_drives_load_spec(self):
        config = ComfyFp8Config({})
        server_args = self._make_server_args()

        spec = resolve_transformer_quant_load_spec(
            hf_config={},
            server_args=server_args,
            safetensors_list=["model.safetensors"],
            component_model_path="/unused/component/path",
            model_cls=_FakeFluxTransformer,
            cls_name=_FakeFluxTransformer.__name__,
            checkpoint_quant_config=config,
        )

        self.assertIs(spec.quant_config, config)
        self.assertTrue(spec.is_comfy_fp8)
        self.assertTrue(spec.needs_device_weight_postprocess)

    def test_checkpoint_quantization_rejects_explicit_quantization(self):
        server_args = self._make_server_args(quantization="fp8")

        with self.assertRaisesRegex(ValueError, "per-layer metadata"):
            resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=["model.safetensors"],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
                checkpoint_quant_config=ComfyFp8Config({}),
            )

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_safetensors_to_load_prefers_mixed_export(
        self, _mock_download
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed = f"{tmpdir}/flux2-dev-nvfp4-mixed.safetensors"
            full = f"{tmpdir}/flux2-dev-nvfp4.safetensors"
            open(mixed, "a").close()
            open(full, "a").close()

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [mixed])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.snapshot_download",
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
    )
    def test_resolve_transformer_safetensors_to_load_refreshes_empty_cached_repo(
        self, mock_download_model, mock_snapshot_download
    ):
        with tempfile.TemporaryDirectory() as cached_dir:
            repo_id = "black-forest-labs/FLUX.2-dev-NVFP4"
            mixed = os.path.join(cached_dir, "flux2-dev-nvfp4-mixed.safetensors")
            mock_download_model.return_value = cached_dir

            def _snapshot_download(**_kwargs):
                open(mixed, "a").close()
                return cached_dir

            mock_snapshot_download.side_effect = _snapshot_download

            server_args = self._make_server_args(transformer_weights_path=repo_id)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [mixed])
        mock_snapshot_download.assert_called_once()

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(
            resolved,
            [
                "/tmp/transformer/diffusion_pytorch_model.safetensors",
                "/tmp/transformer/other.safetensors",
            ],
        )

    def test_filter_transformer_precision_variants_keeps_precision_only_family(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.bf16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(resolved, files)

    def test_weight_load_plan_defers_cpu_offload_for_device_postprocess(self):
        device = torch.device("cuda:0")

        plan = WeightLoadPlan.for_component(
            checkpoint_load_device=device,
            needs_device_weight_postprocess=True,
            component_starts_on_cpu=True,
        )

        self.assertEqual(plan.checkpoint_load_device, device)
        self.assertEqual(plan.weight_postprocess_device, device)
        self.assertTrue(plan.defer_cpu_placement)
        self.assertFalse(plan.load_full_state_dict_on_device)

    def test_weight_load_plan_can_keep_full_state_dict_on_device(self):
        plan = WeightLoadPlan.for_component(
            checkpoint_load_device=torch.device("cuda:0"),
            needs_device_weight_postprocess=False,
            component_starts_on_cpu=False,
            load_full_state_dict_on_device=True,
        )

        self.assertTrue(plan.load_full_state_dict_on_device)

    def test_unquantized_cpu_offload_loads_checkpoint_on_cpu(self):
        device = _resolve_checkpoint_load_device(
            torch.device("cuda:0"),
            component_starts_on_cpu=True,
            runtime_quant_config=None,
        )

        self.assertEqual(device, torch.device("cpu"))

    def test_quantized_cpu_offload_keeps_checkpoint_on_runtime_device(self):
        runtime_device = torch.device("cuda:0")
        device = _resolve_checkpoint_load_device(
            runtime_device,
            component_starts_on_cpu=True,
            runtime_quant_config=object(),
        )

        self.assertEqual(device, runtime_device)

    def test_gguf_cpu_offload_loads_packed_checkpoint_on_cpu(self):
        device = _resolve_checkpoint_load_device(
            torch.device("cuda:0"),
            component_starts_on_cpu=True,
            runtime_quant_config=object(),
            quantized_cpu_load_supported=True,
        )

        self.assertEqual(device, torch.device("cpu"))

    def test_resident_transformer_loads_checkpoint_on_runtime_device(self):
        runtime_device = torch.device("cuda:0")
        device = _resolve_checkpoint_load_device(
            runtime_device,
            component_starts_on_cpu=False,
            runtime_quant_config=None,
        )

        self.assertEqual(device, runtime_device)

    def test_mixed_model_with_expected_dtype_does_not_warn(self):
        model = torch.nn.Module()
        model.fp32 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        model.bf16 = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        with patch.object(transformer_loader.logger, "warning") as warning:
            _warn_if_expected_param_dtype_missing(model, torch.bfloat16)

        warning.assert_not_called()

    def test_model_without_expected_dtype_warns(self):
        model = torch.nn.Linear(1, 1, dtype=torch.float32)

        with patch.object(transformer_loader.logger, "warning") as warning:
            _warn_if_expected_param_dtype_missing(model, torch.bfloat16)

        warning.assert_called_once()

    def test_modelopt_fp8_serialized_checkpoint_needs_device_postprocess(self):
        self.assertTrue(
            _needs_device_weight_postprocess(
                ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
            )
        )

    def test_online_fp8_needs_device_weight_postprocess(self):
        self.assertTrue(_needs_device_weight_postprocess(Fp8Config()))
        self.assertFalse(
            _needs_device_weight_postprocess(
                Fp8Config(is_checkpoint_fp8_serialized=True)
            )
        )
        self.assertTrue(_needs_device_weight_postprocess(_make_quant_config("mxfp8")))
        self.assertFalse(
            _needs_device_weight_postprocess(
                _make_quant_config("mxfp8", is_checkpoint_fp8_serialized=True)
            )
        )
        self.assertTrue(
            _needs_device_weight_postprocess(_make_quant_config("mxfp4_npu"))
        )
        self.assertFalse(
            _needs_device_weight_postprocess(
                _make_quant_config("mxfp4_npu", is_checkpoint_mxfp4_npu_serialized=True)
            )
        )

    def test_comfy_fp8_needs_device_weight_postprocess(self):
        self.assertTrue(
            _needs_device_weight_postprocess(_make_quant_config("comfy_fp8"))
        )

    def test_online_fp8_receives_cli_ignored_layer_patterns(self):
        ignored_layers = ["blocks.0.attn.out_proj", "condition_proj"]
        server_args = self._make_server_args(
            quantization="fp8",
            quantization_ignored_layers=ignored_layers,
        )

        quant_config = _resolve_quant_config(
            hf_config={},
            server_args=server_args,
            safetensors_list=[],
            component_model_path="/unused/component/path",
        )

        self.assertIsInstance(quant_config, Fp8Config)
        self.assertEqual(quant_config.ignored_layers, ignored_layers)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        _mock_download,
        mock_metadata,
        _mock_quant_metadata,
        mock_maybe_download,
        _mock_nvfp4,
    ):
        mock_maybe_download.side_effect = AssertionError(
            "local safetensors path should not trigger maybe_download_model"
        )
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            nunchaku_config = NunchakuConfig(transformer_weights_path=f.name)
            server_args = self._make_server_args(
                transformer_weights_path=nunchaku_config.transformer_weights_path,
                nunchaku_config=nunchaku_config,
            )

            spec = resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=[nunchaku_config.transformer_weights_path],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
            )

        self.assertIsNone(spec.quant_config)
        self.assertIs(spec.nunchaku_config, nunchaku_config)
        self.assertIsNone(spec.param_dtype)
        self.assertEqual(len(spec.post_load_hooks), 1)
        self.assertIs(nunchaku_config.model_cls, _FakeFluxTransformer)
        mock_maybe_download.assert_not_called()

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads(self):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4-mixed.safetensors",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)

    def test_prepare_nvfp4_weight_bytes_swaps_nibbles(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=True)

        self.assertEqual(prepared.tolist(), [[0xBA, 0x01]])

    def test_prepare_nvfp4_weight_bytes_can_skip_nibble_swap(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=False)

        self.assertEqual(prepared.tolist(), [[0xAB, 0x10]])

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_flat_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "swap_weight_nibbles": False,
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_nested_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "swap_weight_nibbles": False,
                },
                "config_groups": {"default": {"weights": {"group_size": 16}}},
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_bitsandbytes_quant_config_resolves_from_hf_config(self):
        config = get_quant_config(
            {
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_quant_storage": "uint8",
                }
            },
            "/unused/component/path",
        )

        self.assertEqual(config.get_name(), "bitsandbytes")
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")

    def test_fp8_quant_config_resolves_from_text_config(self):
        config = get_quant_config(
            {
                "text_config": {
                    "quantization_config": {
                        "quant_method": "fp8",
                        "activation_scheme": "dynamic",
                    }
                }
            },
            "/unused/component/path",
        )

        self.assertIsInstance(config, Fp8Config)
        self.assertTrue(config.is_checkpoint_fp8_serialized)

    def test_bitsandbytes_quant_config_resolves_from_compression_config(self):
        config = get_quant_config(
            {
                "compression_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_storage": "uint8",
                }
            },
            "/unused/component/path",
        )

        self.assertEqual(config.get_name(), "bitsandbytes")
        self.assertTrue(config.load_in_4bit)

    def test_nvfp4_safetensors_inference_ignores_fp8_fallback_scales(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(
                {
                    "fallback.weight": torch.empty(
                        (4, 4),
                        dtype=torch.float8_e4m3fn,
                    ),
                    "fallback.weight_scale": torch.tensor(1.0, dtype=torch.float32),
                    "layers.0.attention.qkv.weight": torch.zeros(
                        (32, 8),
                        dtype=torch.uint8,
                    ),
                    "layers.0.attention.qkv.weight_scale": torch.empty(
                        (32, 1),
                        dtype=torch.float8_e4m3fn,
                    ),
                    "layers.0.attention.qkv.weight_scale_2": torch.tensor(
                        1.0,
                        dtype=torch.float32,
                    ),
                },
                f.name,
            )

            config = build_nvfp4_config_from_safetensors_list([f.name])

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.group_size, 16)
        self.assertIn("fallback", config.exclude_modules)
        self.assertNotIn("layers.0.attention.qkv", config.exclude_modules)
        self.assertEqual(config.checkpoint_weight_scale_layout, "linear")
        self.assertFalse(config.swap_weight_nibbles)

    def test_nvfp4_safetensors_inference_uses_comfy_checkpoint_layout(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(
                {
                    "fallback.weight": torch.empty(
                        (4, 4),
                        dtype=torch.float8_e4m3fn,
                    ),
                    "fallback.weight_scale": torch.tensor(1.0, dtype=torch.float32),
                    "fallback.comfy_quant": torch.tensor(
                        list(b'{"format":"float8_e4m3fn"}'),
                        dtype=torch.uint8,
                    ),
                    "layers.0.attention.qkv.weight": torch.zeros(
                        (32, 8),
                        dtype=torch.uint8,
                    ),
                    "layers.0.attention.qkv.weight_scale": torch.empty(
                        (32, 1),
                        dtype=torch.float8_e4m3fn,
                    ),
                    "layers.0.attention.qkv.weight_scale_2": torch.tensor(
                        1.0,
                        dtype=torch.float32,
                    ),
                    "layers.0.attention.qkv.comfy_quant": torch.tensor(
                        list(b'{"format":"nvfp4"}'),
                        dtype=torch.uint8,
                    ),
                },
                f.name,
            )

            config = build_nvfp4_config_from_safetensors_list([f.name])

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.group_size, 16)
        self.assertIn("fallback", config.exclude_modules)
        self.assertNotIn("layers.0.attention.qkv", config.exclude_modules)
        self.assertEqual(config.checkpoint_weight_scale_layout, "swizzled")
        self.assertTrue(config.swap_weight_nibbles)

    def test_builder_adds_diffusers_quant_type_for_nvfp4(self):
        updated = _updated_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                }
            },
            fallback_patterns=["single_transformer_blocks.*.proj_mlp*"],
            swap_weight_nibbles=False,
        )

        self.assertEqual(updated["quantization_config"]["quant_type"], "NVFP4")
        self.assertEqual(
            updated["quantization_config"]["ignore"],
            ["single_transformer_blocks.*.proj_mlp*"],
        )

    def test_modelopt_fp8_hf_config_uses_general_modelopt_fp8(self):
        config = get_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": ["vae2llm", "llm2vae"],
                }
            },
            "/unused/component/path",
            quant_ignore_remap={"vae2llm": "proj_in", "llm2vae": "proj_out"},
        )

        self.assertIsInstance(config, ModelOptFp8Config)
        self.assertEqual(config.exclude_modules, ["proj_in", "proj_out"])

    def test_modelopt_fp8_explicit_config_uses_general_modelopt_fp8(self):
        config = get_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt_fp8",
                    "quant_algo": "FP8",
                    "ignore": ["proj_out"],
                }
            },
            "/unused/component/path",
        )

        self.assertIsInstance(config, ModelOptFp8Config)
        self.assertEqual(config.exclude_modules, ["proj_out"])

    def test_modelopt_checkpoint_algorithm_admission(self):
        cases = [
            ("modelopt", "FP8", {"ignore": []}, ModelOptFp8Config, None),
            ("modelopt_fp8", "FP8", {"ignore": []}, ModelOptFp8Config, None),
            (
                "modelopt",
                "NVFP4",
                {"group_size": 16, "ignore": []},
                ModelOptFp4Config,
                None,
            ),
            (
                "modelopt_fp4",
                "NVFP4",
                {"group_size": 16, "ignore": []},
                ModelOptFp4Config,
                None,
            ),
            ("modelopt", "MXFP8", {}, None, "maps to 'mxfp8'"),
            ("modelopt", "FP4", {}, None, "maps to 'modelopt_fp4'"),
            ("modelopt", "NVFP4_AWQ", {}, None, "maps to 'modelopt_fp4'"),
            ("modelopt", "W4A16_NVFP4", {}, None, "maps to 'modelopt_fp4'"),
            (
                "modelopt",
                "MIXED_PRECISION",
                {},
                None,
                "mixed precision is not supported",
            ),
            (
                "modelopt",
                "FP8_FAKE",
                {},
                None,
                "Unsupported ModelOpt quant_algo for diffusion: FP8_FAKE",
            ),
            (
                "modelopt_fp8",
                "MXFP8",
                {},
                None,
                "declares quant_method='modelopt_fp8'.*maps to 'mxfp8'",
            ),
            (
                "modelopt_fp4",
                "FP8",
                {},
                None,
                "declares quant_method='modelopt_fp4'.*maps to 'modelopt_fp8'",
            ),
        ]
        for (
            quant_method,
            quant_algo,
            extra_metadata,
            expected_type,
            expected_error,
        ) in cases:
            with self.subTest(quant_algo=quant_algo):
                metadata = {
                    "quant_method": quant_method,
                    "quant_algo": quant_algo,
                    **extra_metadata,
                }
                if expected_error is not None:
                    with self.assertRaisesRegex(ValueError, expected_error):
                        get_quant_config(
                            {"quantization_config": metadata},
                            "/unused/component/path",
                        )
                else:
                    config = get_quant_config(
                        {"quantization_config": metadata},
                        "/unused/component/path",
                    )
                    self.assertIsInstance(config, expected_type)

    def test_explicit_modelopt_method_without_algorithm_is_preserved(self):
        for quant_method in ("modelopt_fp8", "modelopt_fp4"):
            with self.subTest(quant_method=quant_method):
                self.assertEqual(
                    _resolve_quant_method_name({"quant_method": quant_method}),
                    quant_method,
                )

    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    @patch(
        "sglang.multimodal_gen.runtime.models.dits.flux.get_tp_world_size",
        return_value=1,
    )
    def test_flux_single_transformer_block_modelopt_excludes_use_full_prefix(
        self,
        _mock_tp_world_size,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
    ):
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "single_transformer_blocks.*.proj_mlp*",
                "single_transformer_blocks.*.proj_out*",
                "single_transformer_blocks.*.attn.to_q",
            ],
        )

        with patch(
            "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant.current_platform.get_device_capability",
            return_value=DeviceCapability(10, 0),
        ):
            block = FluxSingleTransformerBlock(
                dim=64,
                num_attention_heads=4,
                attention_head_dim=16,
                mlp_ratio=2.0,
                quant_config=quant_config,
                prefix="single_transformer_blocks.0",
            )

        self.assertEqual(block.proj_mlp.prefix, "single_transformer_blocks.0.proj_mlp")
        self.assertEqual(block.proj_out.prefix, "single_transformer_blocks.0.proj_out")
        self.assertEqual(
            block.attn.to_q.prefix, "single_transformer_blocks.0.attn.to_q"
        )
        self.assertIsInstance(block.proj_mlp.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.proj_out.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
