import contextlib
import glob
import json
import os
import shutil
import time
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from diffusers.loaders.lora_base import (
    _best_guess_weight_name,  # watch out for potetential removal from diffusers
)
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import get_lock
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs
from sglang.utils import is_in_ci

logger = init_logger(__name__)


def find_quant_modelslim_config(model_config):
    quant_config_file = Path(model_config["model_path"], "quant_model_description.json")
    quant_cfg = None
    if quant_config_file.is_file():
        with open(quant_config_file) as f:
            quant_cfg = json.load(f)
        # This field is required for flagless model loading but is not present in
        # modelslim model description, so we're adding it here manually.
        quant_cfg["quant_method"] = "modelslim"

    return quant_cfg


def get_quant_config(
    model_config,
    packed_modules_mapping: Dict[str, List[str]] = {},
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cfg = find_quant_modelslim_config(model_config)

    if quant_cfg is not None:
        quant_cls = get_quantization_config(quant_cfg["quant_method"])
        return quant_cls.from_config(quant_cfg)
    else:
        return None
