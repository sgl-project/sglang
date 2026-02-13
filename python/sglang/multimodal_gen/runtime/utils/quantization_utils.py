import json
from pathlib import Path
from typing import Dict, List

from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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
