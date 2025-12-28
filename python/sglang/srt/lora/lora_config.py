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
import os

from huggingface_hub import snapshot_download


class LoRAConfig:
    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        self.hf_config = self.get_lora_config()
        self.target_modules = self.hf_config["target_modules"]

        self.r = self.hf_config["r"]
        self.lora_alpha = self.hf_config["lora_alpha"]

        self.added_tokens_config = self.get_added_tokens_config()
        self.lora_added_tokens_size = (
            len(self.added_tokens_config) if self.added_tokens_config is not None else 0
        )

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
