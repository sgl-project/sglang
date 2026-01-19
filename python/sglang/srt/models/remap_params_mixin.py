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
"""Mixin for mapping checkpoint weight names to SGLang model parameters.

Supported: Llama3, Qwen2, Qwen3, Qwen3-MoE, GLM4, GLM4-MoE, DeepSeekV2/V3
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

StackedEntry = Tuple[
    str, str, Union[int, str]
]  # (sglang_weight, ckpt_weight, shard_id)
ExpertEntry = Tuple[
    str, str, int, Union[int, str]
]  # (sglang_weight, ckpt_weight, expert_id, shard_id)
# (sglang_weight, shard_id, num_shards, expert_id)
MappingResult = Tuple[str, Optional[Union[int, str]], int, Optional[int]]

# Standard scale remapping: (suffix, pattern, replacement)
_SCALE_REMAP = [
    (".k_scale", ".self_attn.k_proj.k_scale", ".self_attn.attn.k_scale"),
    (".v_scale", ".self_attn.v_proj.v_scale", ".self_attn.attn.v_scale"),
    (".k_scale", ".k_scale", ".attn.k_scale"),
    (".v_scale", ".v_scale", ".attn.v_scale"),
]

# Quark scale remapping
_QUARK_SCALE_REMAP = {
    ".q_proj.output_scale": ".attn.q_scale",
    ".k_proj.output_scale": ".attn.k_scale",
    ".v_proj.output_scale": ".attn.v_scale",
    "self_attn.prob_output_scale": ".attn.prob_scale",
}


class RemapParamsMixin:
    """
    Mixin for ckpt->sglang weight name mapping. Override hooks for model-specific logic.
    Supported Models: Llama3, Qwen2, Qwen3, Qwen3-MoE, GLM4, GLM4-MoE, DeepSeekV2
    """

    def mutate_weight_preload(self, name: str) -> str:
        """Pre-transform hook: mutate name unconditionally before mapping.
        Use for: shared expert fusion (MoE), prefix remapping.
        """
        return name

    def custom_scale_remap(self, name: str) -> str:
        """Model-specific scale remapping. Called only when 'scale' in name.
        Use for: Llama activation_scale->input_scale, DeepSeek k_proj->attn_mqa.k_scale.
        """
        return name

    def _apply_scale_remap(self, name: str) -> str:
        for suffix, pattern, replacement in _SCALE_REMAP:
            if name.endswith(suffix) and pattern in name:
                return name.replace(pattern, replacement)
        for quark_suffix, replacement in _QUARK_SCALE_REMAP.items():
            if name.endswith(quark_suffix):
                return name.replace(quark_suffix, replacement)
        return name

    def get_stacked_params_mapping(self) -> List[StackedEntry]:
        return list(getattr(self, "stacked_params_mapping", []) or [])

    def get_expert_params_mapping(self) -> List[ExpertEntry]:
        return list(getattr(self, "expert_params_mapping", []) or [])

    def get_packed_modules_mapping(
        self,
    ) -> Dict[str, List[Tuple[str, Union[int, str]]]]:
        packed: Dict[str, List[Tuple[str, Union[int, str]]]] = {}
        for sglang_param, ckpt, shard_id in self.get_stacked_params_mapping():
            packed.setdefault(sglang_param, []).append((ckpt, shard_id))
        return packed

    @lru_cache(maxsize=128)
    def _get_num_shards(self, sglang_param: str) -> int:
        count = sum(
            1 for p, _, _ in self.get_stacked_params_mapping() if p == sglang_param
        )
        return count if count > 0 else 1

    def get_num_shards_for_param(self, sglang_param: str) -> int:
        return self._get_num_shards(sglang_param)

    def map_weight_name(self, ckpt_weight_name: str) -> MappingResult:
        """Map ckpt name -> (sglang_name, shard_id, num_shards, expert_id)."""
        name = self.mutate_weight_preload(ckpt_weight_name)

        if "scale" in name:
            name = self.custom_scale_remap(name)
            name = self._apply_scale_remap(name)

        for sglang_param, ckpt, expert_id, shard_id in self.get_expert_params_mapping():
            if ckpt in name:
                mapped = name.replace(ckpt, sglang_param)
                return mapped, shard_id, self._get_num_shards(sglang_param), expert_id

        for sglang_param, ckpt, shard_id in self.get_stacked_params_mapping():
            if ckpt in name:
                mapped = name.replace(ckpt, sglang_param)
                return mapped, shard_id, self._get_num_shards(sglang_param), None

        return name, None, 1, None
