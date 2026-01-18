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
"""Mixin providing a unified interface for stacked/fused parameter mappings."""

from typing import Dict, List, Optional, Tuple, Union

# (param_name, ckpt_component_name, shard_id)
StackedEntry = Tuple[str, str, Union[int, str]]
# (param_name_template, ckpt_component_name, expert_id, shard_id)
ExpertEntry = Tuple[str, str, int, Union[int, str]]


class StackedParamsMixin:
    """
    Provide a normalized, discoverable interface for mapping checkpoint weight names
    to model parameter names + shard/expert info.

    Models should either:
      - set self.stacked_params_mapping: list[StackedEntry]
      - set self.expert_params_mapping: list[ExpertEntry] (for MoE)
    Or provide these via properties with the same names.
    """

    def get_stacked_params_mapping(
        self, load_format: Optional[str] = None
    ) -> List[StackedEntry]:
        """Return the normalized stacked params mapping.

        Args:
            load_format: Optional load format hint (e.g., for bitsandbytes).

        Returns:
            List of (param_name, ckpt_component_name, shard_id) tuples.
        """
        return list(getattr(self, "stacked_params_mapping", []) or [])

    def get_expert_params_mapping(self) -> List[ExpertEntry]:
        """Return expert params mapping if present (MoE models).

        Returns:
            List of (param_name_template, ckpt_component_name, expert_id, shard_id) tuples.
        """
        return list(getattr(self, "expert_params_mapping", []) or [])

    def get_packed_modules_mapping(
        self,
    ) -> Dict[str, List[Tuple[str, Union[int, str]]]]:
        """
        Return grouped (inverse) mapping: parent_param -> list of (child_name, shard_id).
        This is derived from stacked_params_mapping.

        Returns:
            Dict mapping parameter names to their component parts.
        """
        packed: Dict[str, List[Tuple[str, Union[int, str]]]] = {}
        for param_name, ckpt_name, shard_id in self.get_stacked_params_mapping():
            packed.setdefault(param_name, []).append((ckpt_name, shard_id))
        return packed

    def map_weight_name(
        self, ckpt_weight_name: str, load_format: Optional[str] = None
    ) -> Tuple[str, Optional[Union[int, str]], int, Optional[int]]:
        """
        Map an input checkpoint weight name to (mapped_model_param_name, shard_id, num_shards, expert_id).

        Args:
            ckpt_weight_name: The weight name from the checkpoint.
            load_format: Optional load format hint.

        Returns:
            Tuple of:
              - mapped_model_param_name: str
              - shard_id: int|str|None
              - num_shards: int
              - expert_id: int|None (present if this is an expert mapping)
        """
        # 1) Expert mappings (MoE) — these are prioritized because they often
        #    include expert indices in the ckpt names.
        for (
            param_name_template,
            ckpt_component_name,
            expert_id,
            shard_id,
        ) in self.get_expert_params_mapping():
            if ckpt_component_name in ckpt_weight_name:
                # Replace the generic checkpoint component with the param_name_template.
                mapped = ckpt_weight_name.replace(
                    ckpt_component_name, param_name_template
                )
                # Count num_shards for this param_name_template in stacked_params_mapping if present,
                # else default to 1
                num_shards = sum(
                    1
                    for p, _, _ in self.get_stacked_params_mapping()
                    if p == param_name_template
                )
                num_shards = num_shards or 1
                return mapped, shard_id, num_shards, expert_id

        # 2) Regular stacked params mapping
        mapping = self.get_stacked_params_mapping(load_format=load_format)
        for param_name, ckpt_component_name, shard_id in mapping:
            if ckpt_component_name in ckpt_weight_name:
                mapped = ckpt_weight_name.replace(ckpt_component_name, param_name)
                # find how many shards map to this param_name
                num_shards = sum(1 for p, _, _ in mapping if p == param_name)
                num_shards = num_shards or 1
                return mapped, shard_id, num_shards, None

        # 3) Not found — default returning original name, no shard, single shard
        return ckpt_weight_name, None, 1, None

    def get_num_shards_for_param(
        self, mapped_param_name: str, load_format: Optional[str] = None
    ) -> int:
        """Return how many shards a mapped parameter expects (derived from stacked mapping).

        Args:
            mapped_param_name: The mapped parameter name.
            load_format: Optional load format hint.

        Returns:
            Number of shards for this parameter.
        """
        mapping = self.get_stacked_params_mapping(load_format=load_format)
        return sum(1 for p, _, _ in mapping if p == mapped_param_name) or 1
