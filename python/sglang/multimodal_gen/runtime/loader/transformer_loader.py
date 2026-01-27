
import dataclasses
import glob
import importlib.util
import json
import os
import traceback
from abc import ABC
from collections.abc import Generator, Iterable
from copy import deepcopy
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import AutoModel
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed import init_device_mesh
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.multimodal_gen.configs.models import EncoderConfig, ModelConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loader import ComponentLoader
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    maybe_load_fsdp_model,
    shard_model,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype, _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_config,
    get_diffusers_component_config,
    get_hf_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


logger = init_logger(__name__)

class TransformerLoader(ComponentLoader):
    """Loader for transformer."""

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the transformer based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)
        hf_config = deepcopy(config)
        cls_name = config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        server_args.model_paths["transformer"] = component_model_path

        # Config from Diffusers supersedes sgl_diffusion's model config
        dit_config = server_args.pipeline_config.dit_config
        dit_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")

        # Check if we should use custom initialization weights
        custom_weights_path = getattr(
            server_args, "init_weights_from_safetensors", None
        )
        use_custom_weights = False

        if use_custom_weights:
            logger.info(
                "Using custom initialization weights from: %s", custom_weights_path
            )
            assert (
                custom_weights_path is not None
            ), "Custom initialization weights must be provided"
            if os.path.isdir(custom_weights_path):
                safetensors_list = _list_safetensors_files(custom_weights_path)
            else:
                assert custom_weights_path.endswith(
                    ".safetensors"
                ), "Custom initialization weights must be a safetensors file"
                safetensors_list = [custom_weights_path]

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        logger.info(
            "Loading %s from %s safetensors files, default_dtype: %s",
            cls_name,
            len(safetensors_list),
            default_dtype,
        )

        # Load the model using FSDP loader
        assert server_args.hsdp_shard_dim is not None
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={"config": dit_config, "hf_config": hf_config},
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            # TODO(will): make these configurable
            default_dtype=default_dtype,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        assert (
            next(model.parameters()).dtype == default_dtype
        ), "Model dtype does not match default dtype"

        model = model.eval()

        return model
