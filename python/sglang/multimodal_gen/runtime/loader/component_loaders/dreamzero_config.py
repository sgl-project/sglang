# SPDX-License-Identifier: Apache-2.0
"""DreamZero checkpoint config helpers.

DreamZero checkpoints package the action head config inside the top-level
``config.json``.  Runtime stages read ``pipeline_config.*.arch_config``, while
the component loaders instantiate modules directly from the checkpoint config.
These helpers keep both views synchronized.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DIT_CONFIG_KEYS = {
    "model_type",
    "patch_size",
    "frame_seqlen",
    "text_len",
    "in_dim",
    "dim",
    "ffn_dim",
    "freq_dim",
    "text_dim",
    "out_dim",
    "num_heads",
    "num_layers",
    "max_chunk_size",
    "qk_norm",
    "cross_attn_norm",
    "eps",
    "num_frame_per_block",
    "num_action_per_block",
    "num_state_per_block",
    "concat_first_frame_latent",
}

_DREAMZERO_VAE_TARGET = "WanVideoVAE"
_DREAMZERO_VAE38_TARGET = "WanVideoVAE38"
_DREAMZERO_VAE38_LATENTS_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
_DREAMZERO_VAE38_LATENTS_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


def dreamzero_checkpoint_config_path(model_path: str | os.PathLike[str]) -> Path:
    return Path(model_path) / "config.json"


def has_dreamzero_checkpoint_config(model_path: str | os.PathLike[str]) -> bool:
    return dreamzero_checkpoint_config_path(model_path).is_file()


def load_dreamzero_checkpoint_config(
    model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    with dreamzero_checkpoint_config_path(model_path).open() as f:
        return json.load(f)


def _action_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("action_head_cfg", {}).get("config", {})


def _diffusion_config(config: dict[str, Any]) -> dict[str, Any]:
    action_cfg = _action_config(config)
    if action_cfg:
        return action_cfg.get("diffusion_model_cfg", {})
    if "model_type" in config and "dim" in config:
        return config
    return {}


def _vae_config(config: dict[str, Any]) -> dict[str, Any]:
    action_cfg = _action_config(config)
    if action_cfg:
        return action_cfg.get("vae_cfg", {})
    diffusion_cfg = _diffusion_config(config)
    if int(diffusion_cfg.get("in_dim", 0) or 0) == 48:
        return {
            "_target_": _DREAMZERO_VAE38_TARGET,
            "z_dim": 48,
            "dim": 160,
        }
    return {}


def dreamzero_dit_init_kwargs_from_config(
    config: dict[str, Any],
    *,
    use_tensor_parallel: bool = False,
) -> dict[str, Any]:
    diffusion_cfg = _diffusion_config(config)
    if not diffusion_cfg:
        raise KeyError(
            "DreamZero config must contain action_head_cfg.config.diffusion_model_cfg "
            "or a Wan diffusion config with model_type/dim"
        )
    action_cfg = _action_config(config)
    is_ti2v = (
        diffusion_cfg.get("model_type") == "ti2v"
        or int(diffusion_cfg.get("in_dim", 0) or 0) == 48
    )

    kwargs = {
        key: diffusion_cfg[key] for key in _DIT_CONFIG_KEYS if key in diffusion_cfg
    }
    kwargs.update(
        action_dim=action_cfg.get("action_dim", config.get("action_dim", 32)),
        max_state_dim=action_cfg.get("max_state_dim", 64),
        # action_head_cfg.config.hidden_size belongs to the surrounding policy head.
        # The DiT state/action MLP hidden width follows CausalWanModel's default
        # unless diffusion_model_cfg explicitly overrides it.
        hidden_size=diffusion_cfg.get("hidden_size", 1024),
        use_tensor_parallel=use_tensor_parallel,
    )
    if "patch_size" in kwargs:
        kwargs["patch_size"] = tuple(kwargs["patch_size"])
    else:
        # CausalWanModel's runtime default.  Several checkpoint configs omit it,
        # but stages need the same value materialized into arch_config.
        kwargs["patch_size"] = (1, 2, 2)
    kwargs.setdefault("frame_seqlen", 50 if is_ti2v else 880)
    kwargs.setdefault("max_chunk_size", 4)
    kwargs.setdefault("num_frame_per_block", 2)
    kwargs.setdefault("num_action_per_block", 24)
    kwargs.setdefault("num_state_per_block", 1)
    if "concat_first_frame_latent" not in kwargs:
        kwargs["concat_first_frame_latent"] = not is_ti2v
    return kwargs


def dreamzero_dit_init_kwargs_from_checkpoint_config(
    model_path: str | os.PathLike[str],
    *,
    use_tensor_parallel: bool = False,
) -> dict[str, Any]:
    return dreamzero_dit_init_kwargs_from_config(
        load_dreamzero_checkpoint_config(model_path),
        use_tensor_parallel=use_tensor_parallel,
    )


def dreamzero_vae_runtime_config_from_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    vae_cfg = _vae_config(config)
    raw_target = str(vae_cfg.get("_target_", _DREAMZERO_VAE_TARGET))
    target = raw_target.rsplit(".", 1)[-1]
    is_vae38 = target == _DREAMZERO_VAE38_TARGET
    runtime_config = {
        "runtime_target": target,
        "z_dim": int(vae_cfg.get("z_dim", 48 if is_vae38 else 16)),
        "dim": int(vae_cfg.get("dim", 160 if is_vae38 else 96)),
        "decoder_dim": 256 if is_vae38 else int(vae_cfg.get("dim", 96)),
        "in_channels": 12 if is_vae38 else 3,
        "out_channels": 12 if is_vae38 else 3,
        "patch_size": 2 if is_vae38 else None,
        "scale_factor_spatial": 16 if is_vae38 else 8,
        "is_residual": is_vae38,
    }
    if is_vae38:
        runtime_config["latents_mean"] = _DREAMZERO_VAE38_LATENTS_MEAN
        runtime_config["latents_std"] = _DREAMZERO_VAE38_LATENTS_STD
    return runtime_config


def dreamzero_vae_runtime_config_from_checkpoint_config(
    model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    return dreamzero_vae_runtime_config_from_config(
        load_dreamzero_checkpoint_config(model_path)
    )


def _set_existing_or_extra(target: Any, name: str, value: Any) -> None:
    setattr(target, name, value)


def materialize_arch_configs_from_checkpoint(
    model_path: str | os.PathLike[str],
    pipeline_config: Any,
) -> bool:
    """Synchronize DreamZero checkpoint arch fields into pipeline_config.

    Returns ``True`` when a DreamZero action-head config was found and applied.
    Non-DreamZero configs are ignored so lightweight registry tests can keep
    using temporary model paths.
    """

    if not has_dreamzero_checkpoint_config(model_path):
        return False
    config = load_dreamzero_checkpoint_config(model_path)
    if not _diffusion_config(config):
        return False

    dit_kwargs = dreamzero_dit_init_kwargs_from_config(config)
    dit_arch = pipeline_config.dit_config.arch_config
    for key, value in dit_kwargs.items():
        if key == "use_tensor_parallel":
            continue
        _set_existing_or_extra(dit_arch, key, value)
    if hasattr(dit_arch, "__post_init__"):
        dit_arch.__post_init__()

    vae_runtime = dreamzero_vae_runtime_config_from_config(config)
    vae_config = pipeline_config.vae_config
    vae_arch = vae_config.arch_config
    vae_config.runtime_target = vae_runtime["runtime_target"]
    _set_existing_or_extra(vae_arch, "runtime_target", vae_runtime["runtime_target"])
    _set_existing_or_extra(vae_arch, "z_dim", vae_runtime["z_dim"])
    _set_existing_or_extra(vae_arch, "dim", vae_runtime["dim"])
    _set_existing_or_extra(vae_arch, "base_dim", vae_runtime["dim"])
    _set_existing_or_extra(vae_arch, "decoder_base_dim", vae_runtime["decoder_dim"])
    _set_existing_or_extra(vae_arch, "in_channels", vae_runtime["in_channels"])
    _set_existing_or_extra(vae_arch, "out_channels", vae_runtime["out_channels"])
    _set_existing_or_extra(vae_arch, "patch_size", vae_runtime["patch_size"])
    _set_existing_or_extra(vae_arch, "is_residual", vae_runtime["is_residual"])
    _set_existing_or_extra(
        vae_arch,
        "scale_factor_spatial",
        vae_runtime["scale_factor_spatial"],
    )
    if "latents_mean" in vae_runtime:
        _set_existing_or_extra(vae_arch, "latents_mean", vae_runtime["latents_mean"])
        _set_existing_or_extra(vae_arch, "latents_std", vae_runtime["latents_std"])
    _set_existing_or_extra(
        vae_arch,
        "spatial_compression_ratio",
        int(
            getattr(
                vae_arch,
                "scale_factor_spatial",
                16 if vae_runtime["runtime_target"] == _DREAMZERO_VAE38_TARGET else 8,
            )
        ),
    )
    _set_existing_or_extra(
        vae_arch,
        "temporal_compression_ratio",
        int(getattr(vae_arch, "scale_factor_temporal", 4)),
    )
    latents_mean = getattr(vae_arch, "latents_mean", ())
    if hasattr(vae_arch, "__post_init__") and len(latents_mean) == vae_runtime["z_dim"]:
        vae_arch.__post_init__()
    action_cfg = _action_config(config)
    for key in ("action_horizon", "num_frames"):
        if key in action_cfg and hasattr(pipeline_config, key):
            setattr(pipeline_config, key, action_cfg[key])
    if "target_video_height" in action_cfg and hasattr(
        pipeline_config, "synthetic_height"
    ):
        pipeline_config.synthetic_height = int(action_cfg["target_video_height"])
    if "target_video_width" in action_cfg and hasattr(
        pipeline_config, "synthetic_width"
    ):
        pipeline_config.synthetic_width = int(action_cfg["target_video_width"])
    return True
