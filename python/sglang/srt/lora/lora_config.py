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
        self.added_tokens = self.get_added_tokens()
        self.target_modules = self.hf_config["target_modules"]

        # TODO: Support lm_head modules
        if any(module in self.target_modules for module in ["lm_head"]):
            raise ValueError("Not supported yet")
        # Uncomment below for testing adapter: yard1/llama-2-7b-sql-lora-test (since it has lm_head which is not supported yet)
        # if "lm_head" in self.target_modules:
        #     self.target_modules.remove("lm_head")

        self.r = self.hf_config["r"]
        self.lora_alpha = self.hf_config["lora_alpha"]
        self.extra_vocab_size = (
            len(self.added_tokens) if self.added_tokens is not None else 0
        )

    def get_added_tokens(self, dummy=False):
        if dummy:
            raise NotImplementedError()
        else:
            if not os.path.isdir(self.path):
                weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
            else:
                weights_dir = self.path
            added_tokens_config_name = "added_tokens.json"
            if os.path.exists(os.path.join(weights_dir, added_tokens_config_name)):
                with open(
                    os.path.join(weights_dir, added_tokens_config_name), "r"
                ) as f:
                    return json.load(f)
            else:
                return None

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
