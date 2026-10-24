import pytest
import torch
from diffusers import SanaTransformer2DModel as DiffusersSanaTransformer2DModel

from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
    initialize_model,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

# Efficient-Large-Model/<id>/transformer/config.json, minus `_diffusers_version`.
_CHECKPOINT_CONFIGS = {
    "Sana_600M_1024px_diffusers": {
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": False,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 1152,
        "cross_attention_head_dim": 72,
        "dropout": 0.0,
        "in_channels": 32,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 36,
        "num_cross_attention_heads": 16,
        "num_layers": 28,
        "out_channels": 32,
        "patch_size": 1,
        "sample_size": 32,
    },
    "Sana_1600M_1024px_diffusers": {
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": False,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 2240,
        "cross_attention_head_dim": 112,
        "dropout": 0.0,
        "in_channels": 32,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 70,
        "num_cross_attention_heads": 20,
        "num_layers": 20,
        "out_channels": 32,
        "patch_size": 1,
        "sample_size": 32,
    },
    "SANA1.5_1.6B_1024px_diffusers": {
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": False,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 2240,
        "cross_attention_head_dim": 112,
        "dropout": 0.0,
        "guidance_embeds": False,
        "in_channels": 32,
        "interpolation_scale": None,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 70,
        "num_cross_attention_heads": 20,
        "num_layers": 20,
        "out_channels": 32,
        "patch_size": 1,
        "qk_norm": "rms_norm_across_heads",
        "sample_size": 32,
    },
    "SANA1.5_4.8B_1024px_diffusers": {
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": False,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 2240,
        "cross_attention_head_dim": 112,
        "dropout": 0.0,
        "guidance_embeds": False,
        "in_channels": 32,
        "interpolation_scale": None,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 70,
        "num_cross_attention_heads": 20,
        "num_layers": 60,
        "out_channels": 32,
        "patch_size": 1,
        "qk_norm": "rms_norm_across_heads",
        "sample_size": 32,
    },
    "Sana_Sprint_0.6B_1024px_diffusers": {
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": False,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 1152,
        "cross_attention_head_dim": 72,
        "dropout": 0.0,
        "guidance_embeds": True,
        "guidance_embeds_scale": 0.1,
        "in_channels": 32,
        "interpolation_scale": None,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 36,
        "num_cross_attention_heads": 16,
        "num_layers": 28,
        "out_channels": 32,
        "patch_size": 1,
        "qk_norm": "rms_norm_across_heads",
        "sample_size": 32,
        "timestep_scale": 1.0,
    },
    "Sana_Sprint_1.6B_1024px_diffusers": {
        "_class_name": "SanaTransformer2DModel",
        "attention_bias": False,
        "attention_head_dim": 32,
        "caption_channels": 2304,
        "cross_attention_dim": 2240,
        "cross_attention_head_dim": 112,
        "dropout": 0.0,
        "guidance_embeds": True,
        "guidance_embeds_scale": 0.1,
        "in_channels": 32,
        "interpolation_scale": None,
        "mlp_ratio": 2.5,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 70,
        "num_cross_attention_heads": 20,
        "num_layers": 20,
        "out_channels": 32,
        "patch_size": 1,
        "qk_norm": "rms_norm_across_heads",
        "sample_size": 32,
    },
}


def _diffusers_state_dict(config: dict) -> dict[str, torch.Tensor]:
    with torch.device("meta"):
        model = DiffusersSanaTransformer2DModel.from_config(config)
    return model.state_dict()


def _native_model(config: dict) -> torch.nn.Module:
    """Build the native DiT the way TransformerLoader.load_customized does."""
    config = dict(config)
    dit_config = SanaConfig()
    dit_config.update_model_arch(config)
    model_cls, _ = ModelRegistry.resolve_model_cls(config.pop("_class_name"))
    return initialize_model(
        model_cls=model_cls,
        init_params={"config": dit_config, "hf_config": config},
        dtype=torch.bfloat16,
        device=torch.device("meta"),
    )


@pytest.mark.parametrize("checkpoint", sorted(_CHECKPOINT_CONFIGS))
def test_native_sana_dit_loads_every_diffusers_weight(
    checkpoint, single_process_model_parallel
):
    """Every checkpoint weight must land on a native parameter of the same shape:
    the loader only warns on unmatched keys and fills unreached params with ones/zeros."""
    config = _CHECKPOINT_CONFIGS[checkpoint]
    native = _native_model(config)
    native_sd = native.state_dict()
    native_names = set(native_sd)
    loaded, _ = hf_to_custom_state_dict(
        hf_param_sd=_diffusers_state_dict(config),
        param_names_mapping=get_param_names_mapping(
            mapping_dict=native.param_names_mapping, valid_target_names=native_names
        ),
        valid_target_names=native_names,
        strict=True,
    )

    skipped = sorted(set(loaded) - native_names)
    unloaded = sorted(native_names - set(loaded))
    assert not skipped, f"checkpoint keys with no native parameter: {skipped}"
    assert not unloaded, f"native parameters no checkpoint key loads: {unloaded}"
    misshapen = sorted(
        f"{name}: checkpoint {tuple(tensor.shape)}, native {tuple(native_sd[name].shape)}"
        for name, tensor in loaded.items()
        if tensor.shape != native_sd[name].shape
    )
    assert not misshapen, f"checkpoint tensors that do not fit: {misshapen}"
