

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
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
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


class VisionLanguageEncoderLoader(ComponentLoader):
    """Loader for vision language encoder (typically Causal LM or Vision2Seq)."""

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str = "vision_language_encoder",
    ) -> Any:
        if transformers_or_diffusers == "vision_language_encoder":
            from transformers import GlmImageForConditionalGeneration

            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            model = GlmImageForConditionalGeneration.from_pretrained(
                component_model_path,
                config=config,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            ).to(get_local_torch_device())
            return model
        else:
            raise ValueError(
                f"Unsupported library for VisionLanguageEncoder: {transformers_or_diffusers}"
            )
