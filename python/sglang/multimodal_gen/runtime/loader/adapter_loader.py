
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
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype, _list_safetensors_files, \
    skip_init_modules
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

class AdapterLoader(ComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads a single safetensors state_dict.
    """

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        config = get_diffusers_component_config(model_path=component_model_path)

        cls_name = config.pop("_class_name", None)
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        config.pop("_diffusers_version", None)
        config.pop("_name_or_path", None)

        server_args.model_paths["connectors"] = component_model_path

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        target_device = get_local_torch_device()
        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        from types import SimpleNamespace

        with set_default_torch_dtype(default_dtype), skip_init_modules():
            connector_cfg = SimpleNamespace(**config)
            model = model_cls(connector_cfg).to(
                device=target_device, dtype=default_dtype
            )

        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")
        if len(safetensors_list) != 1:
            raise ValueError(
                f"Found {len(safetensors_list)} safetensors files in {component_model_path}, expected 1"
            )

        loaded = safetensors_load_file(safetensors_list[0])
        model.load_state_dict(loaded, strict=False)

        return model.eval()

