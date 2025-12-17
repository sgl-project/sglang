import os

from torch import nn

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
import json
from types import SimpleNamespace
from typing import Any


class SimplePipeline(nn.Module):
    def __init__(self, model_path: str, server_args: ServerArgs):
        super().__init__()
        self.model_path = model_path
        self.server_args = server_args

        self.config = self._load_config()
        # In fact, I think getting the config here is not a good idea;
        # it should be passed in from the upper layer instead.

    def _load_config(self) -> dict[str, Any]:
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        # server_args.downloaded_model_path = model_path
        logger.info("Model path: %s", model_path)
        config = verify_model_config_and_directory(model_path)
        return config

    def load_submodules_config(self, submodule_name=None):
        if submodule_name is None:
            raise ValueError("submodule_name is required")
        submodule_path = os.path.join(self.model_path, submodule_name)
        config_path = os.path.join(submodule_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Convert dict to object with attribute access
        return self._dict_to_namespace(config)

    def _dict_to_namespace(self, d):
        """Recursively convert dict to SimpleNamespace for attribute access."""
        if isinstance(d, dict):
            return SimpleNamespace(
                **{k: self._dict_to_namespace(v) for k, v in d.items()}
            )
        elif isinstance(d, list):
            return [self._dict_to_namespace(item) for item in d]
        else:
            return d

    def forward(self, req, server_args):
        pass
