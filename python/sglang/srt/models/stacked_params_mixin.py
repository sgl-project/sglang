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
"""Mixin for mapping checkpoint weight names to SGLang model parameters."""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

# (sglang_param, ckpt_component, shard_id)
StackedEntry = Tuple[str, str, Union[int, str]]
# (sglang_param, ckpt_component, expert_id, shard_id)
ExpertEntry = Tuple[str, str, int, Union[int, str]]


class StackedParamsMixin:
    """Maps checkpoint weight names to SGLang model parameter names with shard info."""

    def get_stacked_params_mapping(self) -> List[StackedEntry]:
        """Return list of (sglang_param, ckpt_component, shard_id) tuples."""
        return list(getattr(self, "stacked_params_mapping", []) or [])

    def get_expert_params_mapping(self) -> List[ExpertEntry]:
        """Return list of (sglang_param, ckpt_component, expert_id, shard_id) tuples."""
        return list(getattr(self, "expert_params_mapping", []) or [])

    def get_packed_modules_mapping(
        self,
    ) -> Dict[str, List[Tuple[str, Union[int, str]]]]:
        """Return sglang_param -> [(ckpt_component, shard_id), ...] mapping."""
        packed: Dict[str, List[Tuple[str, Union[int, str]]]] = {}
        for sglang_param, ckpt_component, shard_id in self.get_stacked_params_mapping():
            packed.setdefault(sglang_param, []).append((ckpt_component, shard_id))
        return packed

    @lru_cache(maxsize=128)
    def _get_num_shards(self, sglang_param: str) -> int:
        """Return number of shards for a given sglang parameter name."""
        count = sum(
            1 for p, _, _ in self.get_stacked_params_mapping() if p == sglang_param
        )
        return count if count > 0 else 1

    def get_num_shards_for_param(self, sglang_param: str) -> int:
        """Return number of shards expected for a parameter."""
        return self._get_num_shards(sglang_param)

    def map_weight_name(
        self, ckpt_weight_name: str
    ) -> Tuple[str, Optional[Union[int, str]], int, Optional[int]]:
        """Map checkpoint weight name to (sglang_name, shard_id, num_shards, expert_id)."""
        # Check expert mappings first (MoE models)
        for (
            sglang_param,
            ckpt_component,
            expert_id,
            shard_id,
        ) in self.get_expert_params_mapping():
            if ckpt_component in ckpt_weight_name:
                mapped_name = ckpt_weight_name.replace(ckpt_component, sglang_param)
                return (
                    mapped_name,
                    shard_id,
                    self._get_num_shards(sglang_param),
                    expert_id,
                )

        # Check stacked params mapping
        for sglang_param, ckpt_component, shard_id in self.get_stacked_params_mapping():
            if ckpt_component in ckpt_weight_name:
                mapped_name = ckpt_weight_name.replace(ckpt_component, sglang_param)
                return mapped_name, shard_id, self._get_num_shards(sglang_param), None

        # No mapping found, returns exact mapping
        return ckpt_weight_name, None, 1, None
