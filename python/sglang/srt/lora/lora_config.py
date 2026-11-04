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
from typing import Dict, Iterable, Optional, Union

from huggingface_hub import snapshot_download

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
            logger.warning(f"Failed to parse added_tokens.json: {e}")
            return None


def get_normalized_target_modules(
    target_modules: Union[str, Iterable[str]],
) -> set[str]:
    """
    Mapping a list of target module name to names of the normalized LoRA weights.
    Handles both base module names (e.g., "gate_proj") and prefixed module names (e.g., "feed_forward.gate_proj").

    Also handles PEFT shorthand strings like "all-linear" or "all" by returning
    {"all"} as a sentinel value.  Callers that need a concrete module set
    should use :func:`auto_detect_lora_target_modules` to resolve the shorthand
    against the loaded base model.
    """
    # Handle PEFT shorthand strings — return {"all"} as sentinel.
    # Callers can resolve to concrete names via auto_detect_lora_target_modules().
    if isinstance(target_modules, str):
        if target_modules not in ["all", "all-linear"]:
            raise ValueError(
                "Only 'all' or 'all-linear' can be used as the string for target module"
            )
        return {"all"}

    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "out_proj": "out_proj",
        "embed_tokens": "embed_tokens",
        "vocab_emb": "embed_tokens",
        "embeddings": "embed_tokens",
        "word_embeddings": "embed_tokens",
        "lm_head": "lm_head",
        "output": "lm_head",
        "unembed_tokens": "lm_head",
        "q_a_proj": "fused_qkv_a_proj_with_mqa",
        "kv_a_proj_with_mqa": "fused_qkv_a_proj_with_mqa",
        # DSA indexer projections are qualified with their parent module name
        # because the bare leaf names collide with unrelated modules in other
        # models (e.g. DeepSeek-V4 attention `wq_b`, Pixtral vision `wk`).
        "wq_b": "indexer.wq_b",
        "wk": "indexer.wk",
        "weights_proj": "indexer.weights_proj",
    }

    result = set()
    for name in target_modules:
        base_name = name.split(".")[-1]
        normalized_name = params_mapping.get(base_name, base_name)
        result.add(normalized_name)
    return result
