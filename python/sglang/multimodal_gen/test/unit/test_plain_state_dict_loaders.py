# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig
from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
    MOVADualTowerConfig,
)
from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import LTXVocoderConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    PipelineComponentLoader,
    PlainStateDictComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# real architectures with reduced widths; checkpoint names use the external layout
CASES = [
    (
        "dual_tower_bridge",
        "DualTowerConditionalBridge",
        MOVADualTowerConfig,
        {
            "visual_layers": 1,
            "audio_layers": 1,
            "visual_hidden_dim": 16,
            "audio_hidden_dim": 16,
            "head_dim": 8,
        },
    ),
    (
        "duration_head",
        "LTX2DurationHead",
        LTX2DurationHeadConfig,
        {
            "video_cross_attention_dim": 8,
            "audio_cross_attention_dim": 8,
            "pooler_hidden_dim": 8,
            "num_pooler_heads": 2,
            "mlp_hidden_dim": 8,
        },
    ),
    (
        "connectors",
        "LTX2TextConnectors",
        LTX2ConnectorConfig,
        {
            "caption_channels": 8,
            "text_proj_in_factor": 2,
            "per_modality_projections": True,
            "video_hidden_dim": 8,
            "audio_hidden_dim": 8,
            "video_connector_num_attention_heads": 2,
            "video_connector_attention_head_dim": 4,
            "video_connector_num_layers": 1,
            "video_connector_num_learnable_registers": 2,
            "audio_connector_num_attention_heads": 2,
            "audio_connector_attention_head_dim": 4,
            "audio_connector_num_layers": 1,
            "audio_connector_num_learnable_registers": 2,
        },
    ),
    (
        "diffusion_decoder",
        "LTX2VideoDiffusionDecoderModel",
        LTX25DiffusionDecoderConfig,
        {
            "latent_channels": 4,
            "decoder_head_dim": 8,
            "decoder_t_emb_dim": 8,
            "decoder_stage_channels": [8, 8, 8, 8, 8],
            "decoder_stage_depths": [1, 1, 1, 1, 1],
            "decoder_upsample_channel_reductions": [1, 1, 1, 1],
        },
    ),
    (
        "vocoder",
        "LTX2VocoderWithBWE",
        LTXVocoderConfig,
        {
            "hidden_channels": 32,
            "upsample_factors": [2],
            "upsample_kernel_sizes": [4],
            "resnet_kernel_sizes": [3],
            "resnet_dilations": [[1, 3, 5]],
            "bwe_hidden_channels": 32,
            "bwe_upsample_factors": [2],
            "bwe_upsample_kernel_sizes": [4],
            "bwe_resnet_kernel_sizes": [3],
            "bwe_resnet_dilations": [[1, 3, 5]],
            "input_sampling_rate": 16000,
            "output_sampling_rate": 32000,
        },
    ),
]


def _write_checkpoint(path, config, weights):
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    save_file(
        {name: tensor.contiguous() for name, tensor in weights.items()},
        path / "model.safetensors",
    )


@pytest.mark.parametrize("role,class_name,config_cls,raw_config", CASES)
@pytest.mark.parametrize("residency", ["component-offload", "resident"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_real_components_restore_weights_and_exact_policy(
    tmp_path, role, class_name, config_cls, raw_config, residency, precision
):
    if residency == "resident" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    config = config_cls()
    config.update_model_arch(raw_config)
    model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    with set_default_torch_dtype(dtype):
        reference = model_cls(config).eval()
    if role != "vocoder":
        reference = reference.to(dtype=dtype)
    # custom linear constructors allocate empty storage, not checkpoint values
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    weights = {}
    for name, tensor in reference.state_dict().items():
        assert torch.isfinite(tensor).all(), name
        name = name.replace("video_aggregate_embed.", "video_text_proj_in.").replace(
            "audio_aggregate_embed.", "audio_text_proj_in."
        )
        if role == "vocoder":
            name = name.replace(".conv_pre.", ".conv_in.").replace(
                ".conv_post.", ".conv_out."
            )
            name = name.replace(".act_post.", ".act_out.").replace(
                ".ups.", ".upsamplers."
            )
            name = name.replace(".resblocks.", ".resnets.").replace(
                ".downsample.lowpass.filter", ".downsample.filter"
            )
        weights[name] = tensor
    component = tmp_path / role
    _write_checkpoint(component, {"_class_name": class_name, **raw_config}, weights)
    # the exact key differs from the structural role, including for vocoders
    name = role + "_2"
    args = ServerArgs(
        model_path="x",
        component_precisions={name: precision},
        component_residency={name: residency},
        component_weights_paths={name: str(component / "model.safetensors")},
    )
    model, _ = PipelineComponentLoader.load_component(
        name, str(component), "diffusers", args, component_type=role
    )
    device = "cuda" if residency == "resident" else "cpu"
    assert not model.training
    assert args.model_paths[name] == str(component)
    for key, expected in reference.state_dict().items():
        actual = model.state_dict()[key]
        assert actual.device.type == device
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    # constructor-owned nonpersistent filters must survive common placement too
    for key, expected in reference.named_buffers():
        torch.testing.assert_close(
            model.get_buffer(key).cpu(), expected, rtol=0, atol=0
        )
    if role == "duration_head":
        reference = reference.to(device)
        inputs = torch.ones(1, 2, 8, device=device, dtype=dtype)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)


class _DecoderOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config["width"], 2)


class _WeightNormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.utils.parametrizations.weight_norm(
            nn.Linear(2, 4, bias=False)
        )


def test_plain_component_restores_folded_weight_norm(tmp_path):
    expected = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    _write_checkpoint(
        tmp_path, {"_class_name": "TestWeightNormModule"}, {"proj.weight": expected}
    )
    args = ServerArgs(
        model_path="x",
        component_precisions={"auxiliary": "fp32"},
        component_residency={"auxiliary": "component-offload"},
    )
    with patch.dict(ModelRegistry.registered_models):
        ModelRegistry.register_model("TestWeightNormModule", _WeightNormModule)
        model, _ = PlainStateDictComponentLoader().load(
            str(tmp_path), args, "auxiliary", "diffusers"
        )
    torch.testing.assert_close(model.proj.weight, expected, rtol=0, atol=0)
    assert set(model.state_dict()) == {"proj.weight"}


@dataclass
class _MappedArch(ArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^q.weight$": ("proj.weight", 0, 2),
            r"^k.weight$": ("proj.weight", 1, 2),
        }
    )


@dataclass
class _MappedConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=_MappedArch)


class _MappedModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(2, 4, bias=False)


@pytest.mark.parametrize("invalid", [None, "collision", "incomplete"])
def test_plain_components_use_shared_fused_weight_mapping(tmp_path, invalid):
    weights = {"q.weight": torch.ones(2, 2), "k.weight": torch.zeros(2, 2)}
    if invalid == "collision":
        weights["proj.weight"] = torch.ones(4, 2)
    elif invalid == "incomplete":
        weights.pop("k.weight")
    _write_checkpoint(tmp_path, {"_class_name": "TestMappedModule"}, weights)
    loader = PlainStateDictComponentLoader()
    loader.config_classes = {"auxiliary": _MappedConfig}
    args = ServerArgs(
        model_path="x", component_residency={"auxiliary": "component-offload"}
    )
    with patch.dict(ModelRegistry.registered_models):
        ModelRegistry.register_model("TestMappedModule", _MappedModule)
        if invalid:
            with pytest.raises(ComponentCheckpointUnsupportedError):
                loader.load(str(tmp_path), args, "auxiliary", "diffusers")
        else:
            model, _ = loader.load(str(tmp_path), args, "auxiliary", "diffusers")
            expected = torch.cat([weights["q.weight"], weights["k.weight"]]).bfloat16()
            torch.testing.assert_close(model.proj.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [None, "missing", "unexpected", "shape"])
def test_sound_tokenizer_only_ignores_encoder_weights(tmp_path, invalid):
    reference = _DecoderOnly({"width": 4})
    weights = dict(reference.state_dict(), **{"encoder.weight": torch.ones(2, 4)})
    if invalid == "missing":
        weights.pop("decoder.bias")
    elif invalid == "unexpected":
        weights["other.weight"] = torch.ones(2, 4)
    elif invalid == "shape":
        weights["decoder.weight"] = torch.ones(3, 4)
    _write_checkpoint(tmp_path, {"_class_name": "TestDecoderOnly", "width": 4}, weights)
    args = ServerArgs(
        model_path="x", component_residency={"sound_tokenizer": "component-offload"}
    )
    with patch.dict(ModelRegistry.registered_models):
        ModelRegistry.register_model("TestDecoderOnly", _DecoderOnly)
        if invalid:
            with pytest.raises(ComponentCheckpointUnsupportedError):
                PipelineComponentLoader.load_component(
                    "sound_tokenizer", str(tmp_path), "diffusers", args
                )
        else:
            model, _ = PipelineComponentLoader.load_component(
                "sound_tokenizer", str(tmp_path), "diffusers", args
            )
            torch.testing.assert_close(
                model.decoder.weight,
                reference.decoder.weight,
                rtol=0,
                atol=0,
            )
