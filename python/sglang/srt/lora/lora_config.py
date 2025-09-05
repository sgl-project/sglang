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
        self.weights_dir = self.get_weights_dir()
        self.hf_config = self.get_lora_config()
        self.added_tokens = self.get_added_tokens()
        self.target_modules = self.hf_config["target_modules"]

        # TODO: Support lm_head modules
        if any(module in self.target_modules for module in ["lm_head"]):
            raise ValueError("Not supported yet")
        # TODO: remove below after we support lm_head
        # Uncomment below for testing adapter: yard1/llama-2-7b-sql-lora-test (since it has lm_head which is not supported yet)
        # if "lm_head" in self.target_modules:
        #     self.target_modules.remove("lm_head")

        self.r = self.hf_config["r"]
        self.lora_alpha = self.hf_config["lora_alpha"]
        self.extra_vocab_size = (
            len(self.added_tokens) if self.added_tokens is not None else 0
        )

    def get_weights_dir(self):
        if not os.path.isdir(self.path):
            return snapshot_download(self.path, allow_patterns=["*.json"])
        else:
            return self.path

    def get_added_tokens(self):
        added_tokens_config_name = "added_tokens.json"
        added_tokens_path = os.path.join(self.weights_dir, added_tokens_config_name)
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path, "r") as f:
                return json.load(f)
        else:
            return None

    def get_lora_config(self, dummy=False):
        if dummy:
            raise NotImplementedError()
        else:
            config_name = "adapter_config.json"
            with open(os.path.join(self.weights_dir, config_name), "r") as f:
                return json.load(f)
