"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
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
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.auto_round import (
    AutoRoundConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import (
    ComfyFp8Config,
    ComfyFullPrecisionFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8 import (
    KitchenInt8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
    _prepare_nvfp4_weight_bytes,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
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
    _Flux2Nvfp4FallbackAdapter,
    _needs_device_weight_postprocess,
    _resolve_quant_config,
    _resolve_weight_override_quantization,
    resolve_transformer_checkpoint_files,
    resolve_transformer_quant_load_spec,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.models.dits.flux import FluxSingleTransformerBlock
from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    Flux2Transformer2DModel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    _resolve_quant_method_name,
    build_nvfp4_config_from_safetensors_list,
    get_quant_config,
)
from sglang.multimodal_gen.runtime.weights.source import (
    filter_duplicate_precision_variant_safetensors,
)
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _updated_quant_config,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.srt.layers.quantization.bitsandbytes import (
    BitsAndBytesConfig as SRTBitsAndBytesConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config as SRTFp8Config
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod as SRTFp8LinearMethod


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
    def test_modelopt_fp8_packed_cutlass_preserves_checkpoint_shard_scales(self):
        method = ModelOptFp8LinearMethod(
            ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
        )
        method.cutlass_fp8_supported = True
        layer = torch.nn.Module()
        layer.logical_widths = [2, 2, 2]
        weight = (
            torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.float8_e4m3fn)
        )
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "input_scale",
            torch.nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=False),
        )

        method.process_weights_after_loading(layer)

        torch.testing.assert_close(layer.weight, weight.t(), rtol=0, atol=0)
        torch.testing.assert_close(
            layer.weight_scale,
            torch.tensor([[0.1], [0.1], [0.2], [0.2], [0.3], [0.3]]),
        )
        torch.testing.assert_close(layer.input_scale, torch.tensor(1.0))

    def test_modelopt_fp8_packed_cutlass_requantizes_incomplete_shard_scales(self):
        method = ModelOptFp8LinearMethod(
            ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
        )
        method.cutlass_fp8_supported = True
        layer = torch.nn.Module()
        layer.logical_widths = [2, 2, 2]
        weight = (
            torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.float8_e4m3fn)
        )
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.tensor(
                    [0.1, torch.finfo(torch.float32).min, 0.3],
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "input_scale",
            torch.nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=False),
        )

        with patch(
            "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant."
            "requantize_with_max_scale",
            return_value=(torch.tensor(0.3), weight.clone()),
        ) as requantize:
            method.process_weights_after_loading(layer)

        requantize.assert_called_once()
        torch.testing.assert_close(layer.weight, weight.t(), rtol=0, atol=0)
        torch.testing.assert_close(
            layer.weight_scale, torch.full((6, 1), 0.3), rtol=0, atol=0
        )

    def test_modelopt_packed_layer_requires_consistent_shard_precision(self):
        prefix = "blocks.0.attn.to_qkv"
        mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
        layer = LinearBase(input_size=16, output_size=48)

        quantized = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            packed_modules_mapping=mapping,
        )
        self.assertIsInstance(
            quantized.get_quant_method(layer, prefix), ModelOptFp8LinearMethod
        )

        excluded = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=[
                "blocks.0.attn.to_q",
                "blocks.0.attn.to_k",
                "blocks.0.attn.to_v",
            ],
            packed_modules_mapping=mapping,
        )
        self.assertIsInstance(
            excluded.get_quant_method(layer, prefix), UnquantizedLinearMethod
        )

        partial = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=["blocks.0.attn.to_q"],
            packed_modules_mapping=mapping,
        )
        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            partial.get_quant_method(layer, prefix)

    def test_qwen_modelopt_fp8_qkv_checkpoint_tensors_are_merged(self):
        mapping = get_param_names_mapping(
            QwenImageTransformer2DModel.get_param_names_mapping_for_quant_config(
                ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
            )
        )
        prefix = "transformer_blocks.0.attn"
        source = {}
        for shard_id, shard_name in enumerate(("q", "k", "v")):
            source[f"{prefix}.to_{shard_name}.weight"] = torch.full(
                (2, 3), shard_id + 1, dtype=torch.float8_e4m3fn
            )
            source[f"{prefix}.to_{shard_name}.bias"] = torch.full(
                (2,), shard_id + 1, dtype=torch.bfloat16
            )
            source[f"{prefix}.to_{shard_name}.weight_scale"] = torch.tensor(
                [0.1 * (shard_id + 1)], dtype=torch.float32
            )
            source[f"{prefix}.to_{shard_name}.input_scale"] = torch.tensor(
                [0.2 * (shard_id + 1)], dtype=torch.float32
            )

        merged, _ = hf_to_custom_state_dict(source, mapping)

        self.assertEqual(merged[f"{prefix}.to_qkv.weight"].shape, (6, 3))
        self.assertEqual(merged[f"{prefix}.to_qkv.bias"].shape, (6,))
        torch.testing.assert_close(
            merged[f"{prefix}.to_qkv.weight_scale"],
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        )
        torch.testing.assert_close(
            merged[f"{prefix}.to_qkv.input_scale"],
            torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32),
        )
        self.assertEqual(
            QwenImageTransformer2DModel.packed_modules_mapping["to_qkv"],
            ["to_q", "to_k", "to_v"],
        )

    def test_qwen_non_fp8_qkv_checkpoint_tensors_are_not_merged(self):
        prefix = "transformer_blocks.0.attn"
        source_name = f"{prefix}.to_q.weight"
        for quant_config in (None, ModelOptFp4Config()):
            with self.subTest(quant_config=quant_config):
                mapping = get_param_names_mapping(
                    QwenImageTransformer2DModel.get_param_names_mapping_for_quant_config(
                        quant_config
                    )
                )
                target_name, merge_index, total_shards = mapping(source_name)
                self.assertEqual(target_name, source_name)
                self.assertIsNone(merge_index)
                self.assertIsNone(total_shards)

    def test_flux2_modelopt_fp8_qkv_checkpoint_tensors_are_merged(self):
        mapping = get_param_names_mapping(Flux2Transformer2DModel.param_names_mapping)
        prefix = "transformer_blocks.0.attn"
        source = {}
        for projection_prefix in ("to_", "add_"):
            source_names = (
                ("q", "k", "v")
                if projection_prefix == "to_"
                else ("q_proj", "k_proj", "v_proj")
            )
            for shard_id, shard_name in enumerate(source_names):
                name = f"{prefix}.{projection_prefix}{shard_name}"
                source[f"{name}.weight"] = torch.full(
                    (2, 3), shard_id + 1, dtype=torch.float8_e4m3fn
                )
                source[f"{name}.weight_scale"] = torch.tensor(
                    [0.1 * (shard_id + 1)], dtype=torch.float32
                )
                source[f"{name}.input_scale"] = torch.tensor([0.2], dtype=torch.float32)

        merged, _ = hf_to_custom_state_dict(source, mapping)

        for target in ("to_qkv", "to_added_qkv"):
            self.assertEqual(merged[f"{prefix}.{target}.weight"].shape, (6, 3))
            torch.testing.assert_close(
                merged[f"{prefix}.{target}.weight_scale"],
                torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            )
            torch.testing.assert_close(
                merged[f"{prefix}.{target}.input_scale"],
                torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32),
            )
        self.assertEqual(
            Flux2Transformer2DModel.packed_modules_mapping["to_qkv"],
            ["to_q", "to_k", "to_v"],
        )

        # On an unfused model (BF16, Hopper FP8, or TP>1), the source
        # projection names are valid model parameters and the loader must keep
        # them separate rather than producing a nonexistent packed target.
        unmerged, _ = hf_to_custom_state_dict(
            source,
            mapping,
            valid_target_names=set(source),
        )
        self.assertEqual(set(unmerged), set(source))
        for name, tensor in source.items():
            torch.testing.assert_close(unmerged[name], tensor)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    def test_weight_override_uses_adjacent_quantization_config(self, _build_nvfp4):
        with tempfile.TemporaryDirectory() as directory:
            weights = f"{directory}/model.safetensors"
            save_file({"block.weight": torch.ones((2, 2))}, weights)
            with open(f"{directory}/config.json", "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "activation_scheme": "dynamic",
                        }
                    },
                    stream,
                )
            server_args = self._make_server_args(transformer_weights_path=weights)

            quant_config = _resolve_quant_config(
                hf_config={
                    "quantization_config": {
                        "quant_method": "fp8",
                        "activation_scheme": "static",
                    }
                },
                server_args=server_args,
                safetensors_list=[weights],
                component_model_path="/base",
            )

        self.assertIsInstance(quant_config, Fp8Config)
        self.assertEqual(quant_config.activation_scheme, "dynamic")
        self.assertTrue(quant_config.is_checkpoint_fp8_serialized)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    def test_unquantized_weight_override_does_not_inherit_base_config(
        self, _build_nvfp4
    ):
        with tempfile.TemporaryDirectory() as directory:
            weights = f"{directory}/model.safetensors"
            save_file({"block.weight": torch.ones((2, 2))}, weights)
            server_args = self._make_server_args(transformer_weights_path=weights)

            quant_config = _resolve_quant_config(
                hf_config={"quantization_config": {"quant_method": "fp8"}},
                server_args=server_args,
                safetensors_list=[weights],
                component_model_path="/base",
            )

        self.assertIsNone(quant_config)

    def test_weight_override_defers_header_without_quant_method_to_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = f"{directory}/model.safetensors"
            save_file(
                {"block.weight": torch.ones((2, 2))},
                weights,
                metadata={"quantization_config": json.dumps({"quant_algo": "NVFP4"})},
            )

            quant_config, declared = _resolve_weight_override_quantization(
                [weights], {}, {}
            )

        self.assertIsNone(quant_config)
        self.assertTrue(declared)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    def test_declared_weight_override_rejects_online_quantization(self, _build_nvfp4):
        with tempfile.TemporaryDirectory() as directory:
            weights = f"{directory}/model.safetensors"
            save_file(
                {"block.weight": torch.ones((2, 2))},
                weights,
                metadata={
                    "quantization_config": json.dumps(
                        {"quant_method": "fp8", "activation_scheme": "dynamic"}
                    )
                },
            )
            server_args = self._make_server_args(
                transformer_weights_path=weights,
                quantization="fp8",
            )

            with self.assertRaisesRegex(ValueError, "online --quantization"):
                _resolve_quant_config(
                    hf_config={},
                    server_args=server_args,
                    safetensors_list=[weights],
                    component_model_path="/base",
                )

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    def test_undeclared_quantized_weight_override_fails_closed(self, _build_nvfp4):
        with tempfile.TemporaryDirectory() as directory:
            weights = f"{directory}/model.safetensors"
            save_file(
                {
                    "block.weight": torch.ones((2, 2)),
                    "block.weight_scale": torch.ones(2),
                },
                weights,
            )
            server_args = self._make_server_args(transformer_weights_path=weights)

            with self.assertRaisesRegex(ValueError, "no supported native"):
                _resolve_quant_config(
                    hf_config={},
                    server_args=server_args,
                    safetensors_list=[weights],
                    component_model_path="/base",
                )

    def test_autoround_config_is_inferred_and_remapped_to_native_prefixes(self):
        layer_config = {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "data_type": "int",
            "act_bits": 16,
        }
        metadata = {
            "quant_method": "auto-round",
            "packing_format": "auto_round:auto_gptq",
            **layer_config,
            "block_name_to_quantize": "transformer_blocks",
            "extra_config": {
                "context_embedder": {**layer_config, "bits": 16},
                **{
                    f"transformer_blocks.0.attn.to_{shard}": layer_config
                    for shard in ("q", "k", "v")
                },
            },
        }

        config = get_quant_config(
            {"quantization_config": metadata}, "/unused/component/path"
        )
        self.assertIsInstance(config, AutoRoundConfig)
        config.remap_checkpoint_prefixes(MiniMaxH3DiTModel.param_names_mapping)
        self.assertEqual(
            config.srt_config.get_layer_config(object(), "condition_proj")[0], 16
        )
        self.assertEqual(
            config.srt_config.get_layer_config(object(), "blocks.0.attn.qkv_proj")[0],
            4,
        )

    def test_mps_layerwise_load_uses_residency_api(self):
        server_args = SimpleNamespace(
            should_configure_layerwise_offload_for_lazy_component=lambda name: (
                name == "transformer"
            )
        )

        with patch.object(
            transformer_loader.current_platform, "is_mps", return_value=True
        ):
            self.assertEqual(
                TransformerLoader().customized_load_kwargs_for_component(
                    server_args, "transformer"
                ),
                {"cpu_offload_flag": True},
            )
            self.assertEqual(
                TransformerLoader().customized_load_kwargs_for_component(
                    server_args, "audio_dit"
                ),
                {},
            )

    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(
                        param_names_mapping={},
                        reverse_param_names_mapping={},
                        quant_ignore_remap={},
                    )
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

    def test_resolve_transformer_checkpoint_files_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            with (
                patch(
                    "sglang.multimodal_gen.runtime.weights.source.HfApi.model_info"
                ) as model_info,
                patch(
                    "sglang.multimodal_gen.runtime.weights.source.hf_hub_download"
                ) as download,
            ):
                resolved = resolve_transformer_checkpoint_files(
                    server_args, "/unused/component/path"
                )

        self.assertEqual(resolved.safetensors, (f.name,))
        self.assertIsNone(resolved.config_path)
        model_info.assert_not_called()
        download.assert_not_called()

    def test_resolve_transformer_checkpoint_files_uses_one_hf_revision(self):
        filename = "weights/model.safetensors"
        references = (
            (
                f"https://huggingface.co/owner/repo/resolve/main/{filename}",
                "main",
            ),
            (f"owner/repo/{filename}", "test-revision"),
        )

        for reference, revision in references:
            with self.subTest(reference=reference):
                server_args = self._make_server_args(
                    transformer_weights_path=reference,
                    revision=revision,
                )
                model_info = SimpleNamespace(
                    sha="immutable-sha",
                    siblings=[
                        SimpleNamespace(rfilename=filename),
                        SimpleNamespace(rfilename="weights/config.json"),
                    ],
                )

                def download(*, filename, **_kwargs):
                    return f"/cache/{filename.rsplit('/', 1)[-1]}"

                with (
                    patch(
                        "sglang.multimodal_gen.runtime.weights.source.HfApi.model_info",
                        return_value=model_info,
                    ) as model_info_call,
                    patch(
                        "sglang.multimodal_gen.runtime.weights.source.hf_hub_download",
                        side_effect=download,
                    ) as download,
                ):
                    resolved = resolve_transformer_checkpoint_files(
                        server_args, "/unused/component/path"
                    )
                self.assertEqual(resolved.safetensors, ("/cache/model.safetensors",))
                self.assertEqual(resolved.config_path, "/cache/config.json")
                model_info_call.assert_called_once_with("owner/repo", revision=revision)
                self.assertEqual(
                    {call.kwargs["filename"] for call in download.call_args_list},
                    {filename, "weights/config.json"},
                )
                self.assertTrue(
                    all(
                        call.kwargs["revision"] == "immutable-sha"
                        for call in download.call_args_list
                    )
                )

    def test_inspect_minimax_h3_safetensors_detects_curve_and_comfy_format(self):
        marker = json.dumps(
            {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            }
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(
                {
                    "adaln_t_table": torch.zeros((1025, 8)),
                    "blocks.0.mlp.fc1.weight": torch.ones((2, 256), dtype=torch.int8),
                    "blocks.0.mlp.fc1.weight_scale": torch.ones((2, 1)),
                    "blocks.0.mlp.fc1.comfy_quant": torch.tensor(
                        list(marker), dtype=torch.uint8
                    ),
                },
                f.name,
            )

            curve_shape, comfy_quant = inspect_minimax_h3_safetensors([f.name])

        self.assertEqual(curve_shape, (1025, 8))
        self.assertEqual(comfy_quant["blocks.0.mlp.fc1"]["format"], "int8_tensorwise")

    def test_inspect_minimax_h3_fp8_detects_static_activation_scale(self):
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

        self.assertEqual(
            layer_markers["blocks.0.mlp.fc1"],
            {"format": "float8_e4m3fn", "_activation_scheme": "static"},
        )

    def test_inspect_minimax_h3_fp8_without_input_scale_uses_dynamic_activation(self):
        marker = torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8)
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(
                {
                    "blocks.0.mlp.fc1.weight": torch.ones(
                        (2, 2), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.mlp.fc1.weight_scale": torch.tensor(0.5),
                    "blocks.0.mlp.fc1.comfy_quant": marker,
                },
                f.name,
            )

            _, layer_markers = inspect_minimax_h3_safetensors([f.name])

        self.assertEqual(
            layer_markers["blocks.0.mlp.fc1"],
            {"format": "float8_e4m3fn", "_activation_scheme": "dynamic"},
        )

    def test_minimax_h3_comfy_int8_resolves_serialized_kitchen(self):
        config = resolve_minimax_h3_checkpoint_quantization(
            {
                "blocks.0.mlp.fc1": {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }
            }
        )

        self.assertIsInstance(config, KitchenInt8Config)
        self.assertTrue(config.is_checkpoint_int8_serialized)
        self.assertTrue(config.checkpoint_uses_native_qkv_layout)
        self.assertFalse(KitchenInt8Config().checkpoint_uses_native_qkv_layout)
        self.assertFalse(_needs_device_weight_postprocess(config))
        self.assertTrue(config.supports_input_partition("blocks.0.mlp.fc1", 6400))
        self.assertFalse(config.supports_input_partition("blocks.0.mlp.fc1", 3200))

    def test_minimax_h3_w4a8_metadata_resolves_serialized_kitchen(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "layers": {
                        "blocks.0.mlp.fc1": {
                            "format": "asym_w4a8_int8",
                            "convrot": True,
                            "group_size": 16,
                            "convrot_groupsize": 256,
                        }
                    }
                }
            )
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "blocks.0.mlp.fc1.weight": torch.ones((2, 128), dtype=torch.int8),
                    "blocks.0.mlp.fc1.weight_s_rel": torch.ones(
                        (2, 16), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.mlp.fc1.weight_s_channel": torch.ones(2),
                    "blocks.0.mlp.fc1.weight_codebook": torch.ones(16),
                },
                checkpoint.name,
                metadata=metadata,
            )

            _, markers = inspect_minimax_h3_safetensors([checkpoint.name])

        config = resolve_minimax_h3_checkpoint_quantization(markers)
        self.assertIsInstance(config, KitchenW4A8Config)
        self.assertTrue(markers["blocks.0.mlp.fc1"]["_has_codebook"])
        self.assertTrue(config.supports_input_partition("blocks.0.mlp.fc1", 256))
        self.assertFalse(config.supports_input_partition("blocks.0.mlp.fc1", 128))

    @patch(
        "sglang.multimodal_gen.runtime.layers.quantization.kitchen_w4a8."
        "w4a8_int8_linear",
        new=object(),
    )
    def test_serialized_w4a8_constructs_packed_weights_and_scales(self):
        config = KitchenW4A8Config(
            {
                "proj": {
                    "format": "asym_w4a8_int8",
                    "convrot": True,
                    "group_size": 16,
                    "convrot_groupsize": 256,
                    "_has_codebook": True,
                    "_has_correction": False,
                }
            }
        )
        layer = ReplicatedLinear(
            256,
            3,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="proj",
        )

        self.assertEqual(layer.weight.shape, (3, 128))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight_s_rel.shape, (3, 16))
        self.assertEqual(layer.weight_s_rel.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_s_channel.shape, (3,))
        self.assertEqual(layer.weight_codebook.shape, (16,))
        self.assertIsNone(layer.weight_correction)

    def test_minimax_h3_w4a4_marker_resolves_packed_kitchen(self):
        marker = json.dumps(
            {
                "format": "convrot_w4a4",
                "convrot_groupsize": 256,
                "linear_dtype": "int8",
            }
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "blocks.0.mlp.fc1.weight": torch.ones((2, 128), dtype=torch.int8),
                    "blocks.0.mlp.fc1.weight_scale": torch.ones(2),
                    "blocks.0.mlp.fc1.comfy_quant": torch.tensor(
                        list(marker), dtype=torch.uint8
                    ),
                },
                checkpoint.name,
            )

            _, markers = inspect_minimax_h3_safetensors([checkpoint.name])

        config = resolve_minimax_h3_checkpoint_quantization(markers)
        self.assertIsInstance(config, KitchenW4A4Config)
        self.assertTrue(config.supports_input_partition("blocks.0.mlp.fc1", 256))
        self.assertFalse(config.supports_input_partition("blocks.0.mlp.fc1", 128))
        self.assertFalse(_needs_device_weight_postprocess(config))

    @patch(
        "sglang.multimodal_gen.runtime.layers.quantization.kitchen_w4a4."
        "convrot_w4a4_linear",
        new=object(),
    )
    def test_serialized_w4a4_constructs_packed_weight_and_row_scale(self):
        config = KitchenW4A4Config(
            {
                "proj": {
                    "format": "convrot_w4a4",
                    "convrot_groupsize": 256,
                }
            }
        )
        layer = ReplicatedLinear(
            256,
            3,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="proj",
        )

        self.assertEqual(layer.weight.shape, (3, 128))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight_scale.shape, (3,))
        self.assertEqual(layer.weight_scale.dtype, torch.float32)

    def test_mixed_w4a4_int8_dispatches_each_serialized_layer(self):
        markers = {
            "w4a4": {
                "format": "convrot_w4a4",
                "convrot_groupsize": 256,
                "linear_dtype": "int8",
            },
            "int8": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            },
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.kitchen_w4a4."
                "convrot_w4a4_linear",
                new=object(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8."
                "_load_comfy_kitchen"
            ),
        ):
            config = resolve_minimax_h3_checkpoint_quantization(markers)
            w4a4 = ReplicatedLinear(
                256,
                3,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="w4a4",
            )
            int8 = ReplicatedLinear(
                256,
                3,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="int8",
            )

        self.assertIsInstance(config, KitchenW4A4Config)
        self.assertEqual(w4a4.weight.shape, (3, 128))
        self.assertEqual(int8.weight.shape, (3, 256))
        self.assertEqual(set(config.selected), {"w4a4", "int8"})

    @patch(
        "sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8."
        "_load_comfy_kitchen"
    )
    def test_serialized_kitchen_constructs_int8_weight_and_row_scale(self, _load):
        config = KitchenInt8Config(
            layer_markers={
                "proj": {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }
            }
        )

        layer = ReplicatedLinear(
            256,
            3,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="proj",
        )

        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight.shape, (3, 256))
        self.assertEqual(layer.weight_scale.dtype, torch.float32)
        self.assertEqual(layer.weight_scale.shape, (3, 1))

    def test_serialized_kitchen_rejects_non_convrot_marker(self):
        with self.assertRaisesRegex(ValueError, "convrot=true"):
            resolve_minimax_h3_checkpoint_quantization(
                {
                    "blocks.0.mlp.fc1": {
                        "format": "int8_tensorwise",
                        "convrot": False,
                        "convrot_groupsize": 256,
                    }
                }
            )

    def test_minimax_h3_comfy_fp8_resolves_per_layer_dispatch(self):
        config = resolve_minimax_h3_checkpoint_quantization(
            {
                "blocks.0.attn.qkv_proj": {
                    "format": "float8_e4m3fn",
                    "_activation_scheme": "dynamic",
                },
                "blocks.0.mlp.fc2": {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": True,
                },
            }
        )

        self.assertIsInstance(config, ComfyFp8Config)
        self.assertTrue(config.checkpoint_uses_native_qkv_layout)
        layer = LinearBase(input_size=1, output_size=1)
        self.assertIsInstance(
            config.get_quant_method(layer, "blocks.0.mlp.fc2"),
            ComfyFullPrecisionFp8LinearMethod,
        )
        fp8_method = config.get_quant_method(layer, "blocks.0.attn.qkv_proj")
        self.assertIsInstance(fp8_method, Fp8LinearMethod)
        self.assertEqual(fp8_method.quant_config.activation_scheme, "dynamic")
        self.assertIsInstance(
            config.get_quant_method(layer, "unmarked"),
            UnquantizedLinearMethod,
        )

    def test_minimax_h3_global_mxfp8_metadata_selects_srt_per_layer(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "blocks.0.attn.out_proj.weight": torch.ones(
                        (32, 64), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.attn.out_proj.weight_scale": torch.ones(
                        (32, 2), dtype=torch.uint8
                    ),
                    "token_refiner.blocks.0.attn.out_proj.weight": torch.ones(
                        (32, 64), dtype=torch.bfloat16
                    ),
                },
                checkpoint.name,
                metadata={"quant_format": "mxfp8"},
            )
            _, layer_markers = inspect_minimax_h3_safetensors([checkpoint.name])
            config = resolve_minimax_h3_checkpoint_quantization(layer_markers)

        self.assertIsInstance(config, MXFP8Config)
        self.assertIsInstance(config, SRTFp8Config)
        layer = LinearBase(input_size=64, output_size=32)
        self.assertIsInstance(
            config.get_quant_method(layer, "blocks.0.attn.out_proj"),
            SRTFp8LinearMethod,
        )
        self.assertIsInstance(
            config.get_quant_method(layer, "token_refiner.blocks.0.attn.out_proj"),
            UnquantizedLinearMethod,
        )

    def test_mxfp8_npu_selects_srt_linear_method(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.mxfp8.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.mxfp8.current_platform.is_npu",
                return_value=True,
            ),
        ):
            config = MXFP8Config()
            method = config.get_quant_method(
                LinearBase(input_size=64, output_size=32),
                "blocks.0.attn.out_proj",
            )

        self.assertIsInstance(method, NPUMXFP8LinearMethod)

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

    def test_resolve_transformer_override_prefers_single_mixed_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed = f"{tmpdir}/flux2-dev-nvfp4-mixed.safetensors"
            full = f"{tmpdir}/flux2-dev-nvfp4.safetensors"
            open(mixed, "a").close()
            open(full, "a").close()

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            resolved = resolve_transformer_checkpoint_files(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved.safetensors, (mixed,))

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = filter_duplicate_precision_variant_safetensors(files)

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

        resolved = filter_duplicate_precision_variant_safetensors(files)

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

    def test_modelopt_fp8_always_needs_device_weight_postprocess(self):
        # Serialized checkpoints still transpose weights and may requantize
        # packed shards through scaled_fp8_quant() on the runtime device.
        self.assertTrue(
            _needs_device_weight_postprocess(
                ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
            )
        )
        self.assertTrue(
            _needs_device_weight_postprocess(
                ModelOptFp8Config(is_checkpoint_fp8_serialized=False)
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
        self.assertTrue(
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
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        mock_metadata,
        _mock_quant_metadata,
        _mock_nvfp4,
    ):
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file({"block.weight": torch.ones((2, 2))}, f.name)
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
        self.assertIsInstance(config, SRTBitsAndBytesConfig)
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
        self.assertIsInstance(config, SRTFp8Config)
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
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {
                        "layers.0.attention.qkv": {"format": "nvfp4"},
                    },
                }
            )
        }
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
                metadata=metadata,
            )

            config = build_nvfp4_config_from_safetensors_list([f.name])

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.group_size, 16)
        self.assertIn("fallback", config.exclude_modules)
        self.assertNotIn("layers.0.attention.qkv", config.exclude_modules)
        self.assertEqual(config.checkpoint_weight_scale_layout, "linear")
        self.assertFalse(config.swap_weight_nibbles)
        self.assertFalse(config.checkpoint_uses_comfy_quantization)

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
        self.assertTrue(config.checkpoint_uses_comfy_quantization)
        self.assertFalse(config.checkpoint_uses_native_qkv_layout)
        spec = TransformerQuantLoadSpec(
            safetensors_list=[f.name],
            quant_config=config,
            nunchaku_config=None,
            param_dtype=None,
        )
        self.assertTrue(spec.uses_comfy_layer_markers)

    def test_minimax_h3_comfy_nvfp4_resolves_modelopt_backend(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {
                        "blocks.0.attn.qkv_proj": {"format": "nvfp4"},
                    },
                }
            )
        }
        with (
            tempfile.NamedTemporaryFile(suffix=".safetensors") as quantized,
            tempfile.NamedTemporaryFile(suffix=".safetensors") as fallback,
        ):
            save_file(
                {
                    "blocks.0.attn.qkv_proj.weight": torch.zeros(
                        (32, 8), dtype=torch.uint8
                    ),
                    "blocks.0.attn.qkv_proj.weight_scale": torch.ones(
                        (32, 1), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.attn.qkv_proj.weight_scale_2": torch.tensor(1.0),
                },
                quantized.name,
                metadata=metadata,
            )
            save_file(
                {"blocks.0.mlp.fc1.weight": torch.ones((2, 2))},
                fallback.name,
            )
            checkpoint_files = [quantized.name, fallback.name]
            _, markers = inspect_minimax_h3_safetensors(checkpoint_files)
            config = resolve_minimax_h3_checkpoint_quantization(
                markers,
                checkpoint_files,
            )

        self.assertIsInstance(config, ModelOptFp4Config)
        self.assertEqual(config.group_size, 16)
        self.assertIn("blocks.0.mlp.fc1", config.exclude_modules)
        self.assertTrue(config.checkpoint_uses_comfy_quantization)
        self.assertTrue(config.checkpoint_uses_native_qkv_layout)
        self.assertEqual(config.checkpoint_weight_scale_layout, "swizzled")
        self.assertTrue(config.swap_weight_nibbles)

    def test_minimax_h3_mixed_nvfp4_companions_dispatch_each_layer(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {
                        "blocks.0.attn.qkv_proj": {"format": "nvfp4"},
                        "blocks.0.attn.out_proj": {
                            "format": "int8_tensorwise",
                            "convrot": True,
                            "convrot_groupsize": 256,
                        },
                        "blocks.0.mlp.fc1": {"format": "float8_e4m3fn"},
                    },
                }
            )
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "blocks.0.attn.qkv_proj.weight": torch.zeros(
                        (32, 8), dtype=torch.uint8
                    ),
                    "blocks.0.attn.qkv_proj.weight_scale": torch.ones(
                        (32, 1), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.attn.qkv_proj.weight_scale_2": torch.tensor(1.0),
                    "blocks.0.attn.out_proj.weight": torch.zeros(
                        (32, 256), dtype=torch.int8
                    ),
                    "blocks.0.attn.out_proj.weight_scale": torch.ones((32, 1)),
                    "blocks.0.mlp.fc1.weight": torch.ones(
                        (32, 64), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.mlp.fc1.weight_scale": torch.tensor(1.0),
                },
                checkpoint.name,
                metadata=metadata,
            )
            _, markers = inspect_minimax_h3_safetensors([checkpoint.name])
            config = resolve_minimax_h3_checkpoint_quantization(
                markers,
                [checkpoint.name],
            )

        self.assertIsInstance(config, ModelOptFp4Config)
        with patch(
            "sglang.multimodal_gen.runtime.layers.quantization."
            "modelopt_quant.current_platform.get_device_capability",
            return_value=DeviceCapability(10, 0),
        ):
            self.assertIsInstance(
                config.get_quant_method(
                    LinearBase(input_size=16, output_size=32),
                    "blocks.0.attn.qkv_proj",
                ),
                ModelOptFp4LinearMethod,
            )
        with patch(
            "sglang.multimodal_gen.runtime.layers.quantization."
            "kitchen_int8._load_comfy_kitchen"
        ):
            self.assertIsInstance(
                config.get_quant_method(
                    LinearBase(input_size=256, output_size=32),
                    "blocks.0.attn.out_proj",
                ),
                KitchenInt8LinearMethod,
            )
        self.assertIsInstance(
            config.get_quant_method(
                LinearBase(input_size=64, output_size=32),
                "blocks.0.mlp.fc1",
            ),
            Fp8LinearMethod,
        )

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
