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
"""Jina Embeddings v5 text models: a Qwen3 embedder plus one task LoRA adapter.

The checkpoint ships one LoRA adapter per task under ``adapters/<task>/``.
The task is fixed per server and its adapter is merged into the base weights
at load time. Choose it with
``--json-model-override-args '{"default_task": "text-matching"}'``.
"""

import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

# The adapter the reference model loads first.
_DEFAULT_TASK = "retrieval"


def _resolve_adapter_dir(*, name_or_path: str, task: str) -> str:
    if os.path.isdir(name_or_path):
        root = name_or_path
    else:
        root = snapshot_download(name_or_path, allow_patterns=[f"adapters/{task}/*"])
    return os.path.join(root, "adapters", task)


# Returns fp32 ``scale * B @ A`` keyed by the base weight name.
def _load_lora_deltas(adapter_dir: str) -> Dict[str, torch.Tensor]:
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    for flag in ("use_dora", "use_rslora", "fan_in_fan_out"):
        if adapter_config.get(flag):
            raise ValueError(f"Unsupported LoRA option {flag} in {adapter_dir}")
    scale = adapter_config["lora_alpha"] / adapter_config["r"]

    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    deltas = {}
    for name, lora_a in tensors.items():
        if not name.endswith(".lora_A.weight"):
            continue
        lora_b = tensors[name.replace(".lora_A.", ".lora_B.")]
        base_name = name.removeprefix("base_model.model.").replace(
            ".lora_A.weight", ".weight"
        )
        deltas[base_name] = (lora_b.float() @ lora_a.float()) * scale
    return deltas


class JinaEmbeddingsV5Model(Qwen3ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.task = getattr(config, "default_task", None) or _DEFAULT_TASK
        if self.task not in config.task_names:
            raise ValueError(
                f"Unknown default_task {self.task!r}; expected one of {config.task_names}"
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        adapter_dir = _resolve_adapter_dir(
            name_or_path=self.config._name_or_path, task=self.task
        )
        deltas = _load_lora_deltas(adapter_dir)

        def merged_weights():
            for name, weight in weights:
                delta = deltas.pop(name.removeprefix("model."), None)
                if delta is not None:
                    weight = (weight.float() + delta.to(weight.device)).to(weight.dtype)
                yield name, weight

        super().load_weights(merged_weights())
        if deltas:
            raise ValueError(
                f"{len(deltas)} LoRA weights in {adapter_dir} match no base weight, "
                f"e.g. {next(iter(deltas))}"
            )


EntryClass = JinaEmbeddingsV5Model
