# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading shared by the Cosmos3 LiDAR encoder and decoder."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


COSMOS3_LIDAR_COMPONENT = "lidar_vae"


COSMOS3_LIDAR_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"


def _load_lidar_component(
    model_cls: type[nn.Module],
    model_path: str,
    config: Mapping[str, Any],
    device: torch.device | str,
    prefixes: tuple[str, ...],
) -> nn.Module:
    folder = os.path.join(model_path, COSMOS3_LIDAR_COMPONENT)
    with open(os.path.join(folder, "config.json")) as handle:
        _check_component_matches(json.load(handle), config)
    model = model_cls(config).float()
    # read only this half of the VAE, rather than materializing both state dicts
    with safe_open(
        os.path.join(folder, COSMOS3_LIDAR_WEIGHTS_FILE), framework="pt", device="cpu"
    ) as checkpoint:
        weights = {
            key: checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith(prefixes)
            or key in ("coords", "latent_mean", "latent_std")
        }
    model.load_state_dict(weights, strict=True)
    if (
        not torch.isfinite(model.latent_mean).all()
        or not torch.isfinite(model.latent_std).all()
        or bool((model.latent_std <= 0).any())
    ):
        raise ValueError(
            "Cosmos3 LiDAR latent statistics must be finite with positive std"
        )
    logger.info(
        "Loaded %s (%d tensors) from %s", model_cls.__name__, len(weights), folder
    )
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def _check_component_matches(
    component: Mapping[str, Any], deployment: Mapping[str, Any]
) -> None:
    """The transformer's ``multiview.lidar`` block must agree with ``lidar_vae/config.json``."""
    for key, expected in deployment.items():
        actual = component.get(key)
        if key == "network_config":
            for sub_key, sub_expected in expected.items():
                if actual is None or actual.get(sub_key) != sub_expected:
                    raise ValueError(
                        f"LiDAR encoder metadata disagrees with the transformer contract on "
                        f"network_config.{sub_key}: {actual and actual.get(sub_key)!r} != {sub_expected!r}."
                    )
        elif actual != expected:
            raise ValueError(
                f"LiDAR encoder metadata disagrees with the transformer contract on {key}: "
                f"{actual!r} != {expected!r}."
            )
