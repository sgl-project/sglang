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
from typing import Dict, Optional

from huggingface_hub import hf_hub_download

from sglang.srt.utils.common import download_hf_file_if_exists

logger = logging.getLogger(__name__)


class LoRAConfig:
    def __init__(
        self,
        path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        added_tokens_config: Optional[Dict] = None,
        base_vocab_size: Optional[int] = None,
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
        self.use_dora = self.hf_config.get("use_dora", False)

        # Filter fake added tokens: tokens with ID < base_vocab_size are already
        # part of the base vocabulary and should not be treated as added tokens.
        # This commonly happens when added_tokens.json is copied from the base
        # model's tokenizer.
        if self.added_tokens_config and base_vocab_size is not None:
            self.added_tokens_config = {
                token: token_id
                for token, token_id in self.added_tokens_config.items()
                if token_id >= base_vocab_size
            }

        self.lora_added_tokens_size = (
            len(self.added_tokens_config) if self.added_tokens_config is not None else 0
        )

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
        base_vocab_size: Optional[int] = None,
    ) -> "LoRAConfig":
        return cls(
            config_dict=config_dict,
            added_tokens_config=added_tokens_config,
            base_vocab_size=base_vocab_size,
        )

    def get_lora_config(self, dummy=False):
        if dummy:
            raise NotImplementedError()
        else:
            config_name = "adapter_config.json"
            if not os.path.isdir(self.path):
                config_path = hf_hub_download(repo_id=self.path, filename=config_name)
            else:
                config_path = os.path.join(self.path, config_name)
            with open(config_path, "r") as f:
                return json.load(f)

    def get_added_tokens_config(self):
        """Load added tokens from the LoRA adapter if the file exists."""
        if not os.path.isdir(self.path):
            added_tokens_path = download_hf_file_if_exists(
                repo_id=self.path, filename="added_tokens.json"
            )
        else:
            added_tokens_path = os.path.join(self.path, "added_tokens.json")

        # Return None if the file doesn't exist (optional for standard LoRA adapters)
        if added_tokens_path is None or not os.path.exists(added_tokens_path):
            return None

        # Load and return the added tokens
        try:
            with open(added_tokens_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse added_tokens.json: {e}")
            return None
