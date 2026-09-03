from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn as nn

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
from sglang.srt.models.token_probe.config import (
    SING_PROBE_ATTN_MODEL_TYPE,
    SING_PROBE_IDENTITY_MODEL_TYPE,
    SING_PROBE_MLP_MODEL_TYPE,
    ProbeConfig,
)
from sglang.srt.models.token_probe.heads import (
    IdentityProbeHead,
    SingProbeAttnModel,
    SingProbeMlpModel,
)

PROBE_MODELS: dict[str, type[nn.Module]] = {
    SING_PROBE_IDENTITY_MODEL_TYPE: IdentityProbeHead,
    SING_PROBE_MLP_MODEL_TYPE: SingProbeMlpModel,
    SING_PROBE_ATTN_MODEL_TYPE: SingProbeAttnModel,
}


def read_probe_config(ckpt_path: str) -> dict:
    with open(os.path.join(ckpt_path, "config.json"), encoding="utf-8") as config_file:
        return json.load(config_file)


def read_probe_labels(ckpt_path: str) -> tuple[str, ...]:
    return ProbeConfig.from_dict(read_probe_config(ckpt_path)).labels


def load_probe_head(
    ckpt_path: str,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a canonical SingProbe head and its safetensors weights."""
    config = ProbeConfig.from_dict(read_probe_config(ckpt_path))
    head_cls = PROBE_MODELS.get(config.model_type)
    if head_cls is None:
        raise ValueError(
            f"unknown token probe model_type {config.model_type!r}; "
            f"expected one of {sorted(PROBE_MODELS)}"
        )

    model = head_cls.from_config(config, dtype=dtype)
    if config.model_type != SING_PROBE_IDENTITY_MODEL_TYPE:
        files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors weights found under {ckpt_path}")
        DefaultModelLoader.load_weights_and_postprocess(
            model,
            safetensors_weights_iterator(files),
            torch.device(device),
        )
    return model.eval()
