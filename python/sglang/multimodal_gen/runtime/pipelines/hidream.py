"""HiDream-O1-Image unified VLM diffusion pipeline.

HiDream is a unified model: the Qwen3VL backbone handles text encoding,
vision encoding, and diffusion denoising in a single forward pass.
No separate text encoder, VAE, or scheduler modules are needed.
"""

from __future__ import annotations

import json
import math
import os
from glob import glob
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

from sglang.multimodal_gen.configs.models.dits.hidream import HiDreamDitConfig
from sglang.multimodal_gen.configs.pipeline_configs.hidream import (
    HiDreamPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.hidream import (
    _VARIANT_DEFAULTS,
    _resolve_variant,
    HiDreamSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.hidream import HiDreamForCausalLM
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream import (
    HiDreamGenerationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HiDreamPipeline(ComposedPipelineBase):
    """Pipeline for HiDream-O1-Image unified VLM diffusion model."""

    pipeline_name = "HiDreamPipeline"
    pipeline_config_cls = HiDreamPipelineConfig
    sampling_params_cls = HiDreamSamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "HiDreamPipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None and {"model", "processor"} <= set(loaded_modules):
            return loaded_modules

        model_path = self.model_path
        logger.info("Loading HiDream model from %s", model_path)

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            hf_config = json.load(f)

        dit_config = HiDreamDitConfig()
        model = HiDreamForCausalLM(dit_config, hf_config)

        # TP linears shard via param.weight_loader, not load_state_dict
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_files = sorted(set(index["weight_map"].values()))
        else:
            weight_files = sorted(glob(os.path.join(model_path, "*.safetensors")))

        params_dict = dict(model.named_parameters())
        loaded_keys = set()
        # SRT fused layers: (checkpoint suffix, target suffix, shard_id)
        stacked_mapping = {
            "self_attn.q_proj.": ("self_attn.qkv_proj.", "q"),
            "self_attn.k_proj.": ("self_attn.qkv_proj.", "k"),
            "self_attn.v_proj.": ("self_attn.qkv_proj.", "v"),
            "mlp.gate_proj.": ("mlp.gate_up_proj.", 0),
            "mlp.up_proj.": ("mlp.gate_up_proj.", 1),
        }
        for wf in weight_files:
            shard = load_file(os.path.join(model_path, wf))
            for key, value in shard.items():
                model_key = key[6:] if key.startswith("model.") else key
                # reused SRT VisionAttention names the fused layer qkv_proj
                if ".attn.qkv." in model_key:
                    model_key = model_key.replace(".attn.qkv.", ".attn.qkv_proj.")
                shard_id = None
                for src_suffix, (dst_suffix, sid) in stacked_mapping.items():
                    if src_suffix in model_key:
                        model_key = model_key.replace(src_suffix, dst_suffix)
                        shard_id = sid
                        break
                if model_key not in params_dict:
                    continue
                param = params_dict[model_key]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    if shard_id is not None:
                        weight_loader(param, value, shard_id)
                    else:
                        weight_loader(param, value)
                else:
                    param.data.copy_(value)
                loaded_keys.add(model_key)

        missing_keys = set(params_dict.keys()) - loaded_keys
        if missing_keys:
            missing_list = sorted(missing_keys)
            logger.warning("Missing keys when loading HiDream (%d): %s",
                           len(missing_list), missing_list[:10])
        logger.info("HiDream model loaded successfully (%d/%d params loaded)",
                     len(loaded_keys), len(params_dict))

        # Variant dtype policy (deliberate deployment mapping, mirrored in
        # _VARIANT_DEFAULTS)
        variant = _resolve_variant(model_path)
        w_dtype = getattr(torch, _VARIANT_DEFAULTS[variant]["torch_dtype"])
        model = model.to(w_dtype).eval()
        logger.info("HiDream %s weights cast to %s", variant, w_dtype)
        device = get_local_torch_device()
        if device.type != "cpu":
            model = model.to(device)
            logger.info("HiDream model moved to %s", device)

        return {"model": model, "processor": processor}

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        del server_args
        self.add_stage(InputValidationStage())
        self.add_stage(
            HiDreamGenerationStage(
                model=self.get_module("model"),
                processor=self.get_module("processor"),
            ),
            "hidream_generation_stage",
        )


EntryClass = HiDreamPipeline
