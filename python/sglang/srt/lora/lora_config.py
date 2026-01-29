# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import logging
import os
from typing import Dict, Optional, Set

from huggingface_hub import snapshot_download
from safetensors import safe_open

from sglang.srt.lora.utils import get_target_module_name

logger = logging.getLogger(__name__)


class LoRAConfig:
    def __init__(
        self,
        path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        added_tokens_config: Optional[Dict] = None,
    ) -> None:
        self.path = path

        if config_dict is not None:
            self.hf_config = config_dict
            self.added_tokens_config = added_tokens_config
        else:
            self.hf_config = self.get_lora_config()
            self.added_tokens_config = self.get_added_tokens_config()

        self.target_modules = self.hf_config["target_modules"]
        self.r = self.hf_config["r"]
        self.lora_alpha = self.hf_config["lora_alpha"]
        self.lora_added_tokens_size = (
            len(self.added_tokens_config) if self.added_tokens_config is not None else 0
        )

        # Compute effective target modules from actual weights in safetensors
        # This is more accurate than using the declared target_modules from config,
        # especially when exclude_modules was used during training.
        if path is not None:
            self.effective_target_modules = self._compute_effective_target_modules()
        else:
            # For from_dict case, fall back to declared target_modules
            self.effective_target_modules = None

    def _compute_effective_target_modules(self) -> Optional[Set[str]]:
        """
        Extract actual module names from safetensors weights.

        This scans the safetensors file to determine which modules actually have
        LoRA weights, rather than relying on the declared target_modules in the
        config (which may include modules that were excluded during training).

        Returns:
            Set of module names that actually have LoRA weights, or None if
            safetensors file not found.
        """
        weights_dir = self._get_weights_dir()
        if weights_dir is None:
            logger.debug(
                f"Path '{self.path}' is not a local directory, "
                "falling back to declared target_modules"
            )
            return None

        safetensors_path = self._find_safetensors_file(weights_dir)
        if safetensors_path is None:
            logger.warning(
                f"No safetensors file found in {weights_dir}, "
                "falling back to declared target_modules"
            )
            return None

        modules = set()
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                module_name = get_target_module_name(key, set(self.target_modules))
                modules.add(module_name)

        if not modules:
            logger.warning(
                "No LoRA modules found in safetensors, "
                "falling back to declared target_modules"
            )
            return None

        logger.debug(f"Computed effective target modules from safetensors: {modules}")
        return modules

    def _get_weights_dir(self) -> Optional[str]:
        """Get the directory containing adapter weights."""
        if os.path.isdir(self.path):
            return self.path
        # For HF repo IDs, fall back to declared target_modules
        # (safetensors would need to be downloaded separately)
        return None

    def _find_safetensors_file(self, weights_dir: str) -> Optional[str]:
        """Find the safetensors file in the weights directory."""
        path = os.path.join(weights_dir, "adapter_model.safetensors")
        if os.path.exists(path):
            return path
        return None

    def filter_added_tokens(self, base_vocab_size: int) -> None:
        """
        Filter added_tokens_config to only include truly added tokens.

        Tokens with ID < base_vocab_size are already part of the base model's
        vocabulary and should not be treated as added tokens. This commonly
        happens when added_tokens.json is copied from the base model's tokenizer.

        Args:
            base_vocab_size: The vocabulary size of the base model.
        """
        if not self.added_tokens_config:
            return

        original_count = len(self.added_tokens_config)
        self.added_tokens_config = {
            token: token_id
            for token, token_id in self.added_tokens_config.items()
            if token_id >= base_vocab_size
        }
        self.lora_added_tokens_size = len(self.added_tokens_config)

        filtered_count = original_count - self.lora_added_tokens_size
        if filtered_count > 0:
            logger.debug(
                f"Filtered {filtered_count} tokens from added_tokens_config "
                f"(ID < {base_vocab_size}). Remaining: {self.lora_added_tokens_size}"
            )

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
    ) -> "LoRAConfig":
        return cls(config_dict=config_dict, added_tokens_config=added_tokens_config)

    def get_lora_config(self, dummy=False):
        if dummy:
            raise NotImplementedError()
        else:
            if not os.path.isdir(self.path):
                weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
            else:
                weights_dir = self.path
            config_name = "adapter_config.json"
            with open(os.path.join(weights_dir, config_name), "r") as f:
                return json.load(f)

    def get_added_tokens_config(self):
        """Load added tokens from the LoRA adapter if the file exists."""
        # Determine the weights directory
        if not os.path.isdir(self.path):
            weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
        else:
            weights_dir = self.path

        # Construct the path to added_tokens.json
        added_tokens_path = os.path.join(weights_dir, "added_tokens.json")

        # Return None if the file doesn't exist (optional for standard LoRA adapters)
        if not os.path.exists(added_tokens_path):
            return None

        # Load and return the added tokens
        try:
            with open(added_tokens_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # Log warning but don't crash if JSON is malformed
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse added_tokens.json: {e}")
            return None
