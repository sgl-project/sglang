
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
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype, skip_init_modules, \
    _list_safetensors_files
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



class VAELoader(ComponentLoader):
    """Loader for VAE."""

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, module_name: str
    ):
        """Load the VAE based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        server_args.model_paths[module_name] = component_model_path

        logger.debug("HF model config: %s", config)
        if module_name == "audio_vae":
            vae_config = server_args.pipeline_config.audio_vae_config
            vae_precision = server_args.pipeline_config.audio_vae_precision
        else:
            vae_config = server_args.pipeline_config.vae_config
            vae_precision = server_args.pipeline_config.vae_precision

        vae_config.update_model_arch(config)

        # NOTE: some post init logics are only available after updated with config
        vae_config.post_init()

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        # Check for auto_map first (custom VAE classes)
        auto_map = config.get("auto_map", {})
        auto_model_map = auto_map.get("AutoModel")
        if auto_model_map:
            module_path, cls_name = auto_model_map.rsplit(".", 1)
            custom_module_file = os.path.join(component_model_path, f"{module_path}.py")
            spec = importlib.util.spec_from_file_location("_custom", custom_module_file)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            vae_cls = getattr(custom_module, cls_name)
            vae_dtype = PRECISION_TO_TYPE[vae_precision]
            with set_default_torch_dtype(vae_dtype):
                vae = vae_cls.from_pretrained(
                    component_model_path,
                    revision=server_args.revision,
                    trust_remote_code=server_args.trust_remote_code,
                )
            vae = vae.to(device=target_device, dtype=vae_dtype)
            return vae.eval()

        # Load from ModelRegistry (standard VAE classes)
        with (
            set_default_torch_dtype(PRECISION_TO_TYPE[vae_precision]),
            skip_init_modules(),
        ):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {component_model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded, strict=False)
        return vae.eval()

