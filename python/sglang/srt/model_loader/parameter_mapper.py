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
"""Parameter mapping from HuggingFace checkpoint names to SGLang model parameters.

This module provides utilities for translating weight names between HuggingFace
checkpoint format and SGLang's internal parameter naming, handling:

1. Stacked Parameter Fusion
   - gate_proj + up_proj → gate_up_proj (num_shards=2)
   - q_proj + k_proj + v_proj → qkv_proj (num_shards=3)
   - q_a_proj + kv_a_proj_with_mqa → fused_qkv_a_proj_with_mqa (DeepSeek MLA)

2. Expert Parameter Sharding (MoE models)
   - experts.{id}.gate_proj + experts.{id}.up_proj → experts.w13_weight (num_shards=2)
   - experts.{id}.down_proj → experts.w2_weight (num_shards=1)
   - Handles expert parallelism: num_local_experts = n_routed // ep_size + shared

3. Scale Remapping (Quantized models)
   - k_proj.k_scale → attn.k_scale
   - v_proj.v_scale → attn.v_scale
   - Quark-specific: output_scale → per-component scales

Supported Models:
    Dense: Llama, Qwen2, Qwen3, GLM4
    MoE:   DeepSeekV2/V3/R1, Qwen3-MoE, GLM4-MoE

Example:
    >>> mapper = ParameterMapper.from_model(model)
    >>> result = mapper.map("model.layers.0.mlp.gate_proj.weight")
    >>> result.sglang_name  # "model.layers.0.mlp.gate_up_proj.weight"
    >>> result.shard_id     # 0
    >>> result.num_shards   # 2
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

StackedParamsEntry = Tuple[str, str, Union[int, str]]
ExpertParamsEntry = Tuple[str, str, int, Union[int, str]]


@dataclass
class MappingResult:
    """Result of mapping a HuggingFace checkpoint weight name to SGLang parameter."""

    sglang_name: str
    shard_id: Optional[Union[int, str]]
    num_shards: int
    expert_id: Optional[int]
    num_local_experts: Optional[int]


# Standard FP8 scale remapping patterns
_SCALE_REMAP_PATTERNS: List[Tuple[str, str, str]] = [
    (".k_scale", ".self_attn.k_proj.k_scale", ".self_attn.attn.k_scale"),
    (".v_scale", ".self_attn.v_proj.v_scale", ".self_attn.attn.v_scale"),
    (".k_scale", ".k_scale", ".attn.k_scale"),
    (".v_scale", ".v_scale", ".attn.v_scale"),
]

# Quark quantization scale remapping
_QUARK_SCALE_REMAP: Dict[str, str] = {
    ".q_proj.output_scale": ".attn.q_scale",
    ".k_proj.output_scale": ".attn.k_scale",
    ".v_proj.output_scale": ".attn.v_scale",
    "self_attn.prob_output_scale": ".attn.prob_scale",
}


class ParameterMapper:
    """Maps HuggingFace checkpoint weight names to SGLang model parameters.

    This class pre-computes lookup tables at initialization for efficient
    repeated mapping. It handles:
    - Stacked/fused parameter mapping (gate_up_proj, qkv_proj, etc.)
    - Expert parameter mapping with shard information
    - Scale remapping for quantized models
    - Model-specific weight name mutations
    """

    def __init__(
        self,
        stacked_params_mapping: List[StackedParamsEntry],
        expert_params_mapping: List[ExpertParamsEntry],
        num_local_experts: int = 0,
        mutate_weight_preload: Optional[Callable[[str], str]] = None,
        custom_scale_remap: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the parameter mapper with model-specific configuration.

        Args:
            stacked_params_mapping: List of (sglang_name, hf_name, shard_id) tuples.
                Example: [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
            expert_params_mapping: List of (sglang_name, hf_name, expert_id, shard_id) tuples.
                Example: [("w13_weight", "experts.0.gate_proj.weight", 0, 0), ...]
            num_local_experts: Number of experts in the current model rank.
                For EP=1: num_local_experts = n_routed_experts + num_fused_shared_experts
                For EP>1: num_local_experts = n_routed_experts // ep_size + num_fused_shared_experts
            mutate_weight_preload: Optional function to transform weight names before mapping.
                Used for shared expert fusion in DeepSeek (shared_experts → experts.{n_routed}).
            custom_scale_remap: Optional function for model-specific scale remapping.
                Used for DeepSeek k_proj/v_proj → attn_mqa scale mapping.
        """
        self.num_local_experts = num_local_experts
        self._mutate_weight_preload = mutate_weight_preload
        self._custom_scale_remap = custom_scale_remap

        self._stacked_lookup, self._stacked_num_shards = self._build_stacked_lookup(
            stacked_params_mapping
        )
        self._expert_lookup, self._expert_num_shards = self._build_expert_lookup(
            expert_params_mapping
        )

    @staticmethod
    def _build_stacked_lookup(
        mapping: List[StackedParamsEntry],
    ) -> Tuple[Dict[str, Tuple[str, Union[int, str]]], Dict[str, int]]:
        """Build lookup table and num_shards from stacked params mapping."""
        lookup: Dict[str, Tuple[str, Union[int, str]]] = {}
        shard_counts: Dict[str, int] = {}

        for sglang_name, hf_name, shard_id in mapping:
            lookup[hf_name] = (sglang_name, shard_id)
            shard_counts[sglang_name] = shard_counts.get(sglang_name, 0) + 1

        return lookup, shard_counts

    @staticmethod
    def _build_expert_lookup(
        mapping: List[ExpertParamsEntry],
    ) -> Tuple[Dict[str, Tuple[str, int, Union[int, str]]], Dict[str, int]]:
        """Build lookup table and num_shards from expert params mapping."""
        lookup: Dict[str, Tuple[str, int, Union[int, str]]] = {}
        shard_counts: Dict[str, int] = {}

        for sglang_name, hf_name, expert_id, shard_id in mapping:
            lookup[hf_name] = (sglang_name, expert_id, shard_id)

        for sglang_name, _, _, shard_id in mapping:
            key = sglang_name
            if key not in shard_counts:
                unique_shards = set(
                    s_id for s_name, _, _, s_id in mapping if s_name == sglang_name
                )
                shard_counts[key] = len(unique_shards)

        return lookup, shard_counts

    def _apply_scale_remap(self, name: str) -> str:
        """Apply standard and Quark scale remapping patterns."""
        for suffix, pattern, replacement in _SCALE_REMAP_PATTERNS:
            if name.endswith(suffix) and pattern in name:
                return name.replace(pattern, replacement)

        for quark_suffix, replacement in _QUARK_SCALE_REMAP.items():
            if name.endswith(quark_suffix):
                return name.replace(quark_suffix, replacement)

        return name

    def map(self, hf_weight_name: str) -> MappingResult:
        """Map a HuggingFace checkpoint weight name to SGLang parameter info.

        Args:
            hf_weight_name: The weight name from HuggingFace checkpoint.

        Returns:
            MappingResult with mapped name and sharding information.
        """
        name = hf_weight_name

        if self._mutate_weight_preload is not None:
            name = self._mutate_weight_preload(name)

        if "scale" in name:
            if self._custom_scale_remap is not None:
                name = self._custom_scale_remap(name)
            name = self._apply_scale_remap(name)

        for hf_pattern, (
            sglang_name,
            expert_id,
            shard_id,
        ) in self._expert_lookup.items():
            if hf_pattern in name:
                mapped_name = name.replace(hf_pattern, sglang_name)
                return MappingResult(
                    sglang_name=mapped_name,
                    shard_id=shard_id,
                    num_shards=self._expert_num_shards.get(sglang_name, 1),
                    expert_id=expert_id,
                    num_local_experts=self.num_local_experts,
                )

        for hf_pattern, (sglang_name, shard_id) in self._stacked_lookup.items():
            if hf_pattern in name:
                mapped_name = name.replace(hf_pattern, sglang_name)
                return MappingResult(
                    sglang_name=mapped_name,
                    shard_id=shard_id,
                    num_shards=self._stacked_num_shards.get(sglang_name, 1),
                    expert_id=None,
                    num_local_experts=None,
                )

        return MappingResult(
            sglang_name=name,
            shard_id=None,
            num_shards=1,
            expert_id=None,
            num_local_experts=None,
        )

    @classmethod
    def from_model(cls, model) -> "ParameterMapper":
        """Create a ParameterMapper from a model instance."""
        stacked_mapping = list(getattr(model, "stacked_params_mapping", []) or [])
        expert_mapping = list(getattr(model, "expert_params_mapping", []) or [])

        num_local_experts = 0
        if hasattr(model, "num_local_experts"):
            num_local_experts = model.num_local_experts
        elif expert_mapping:
            expert_ids = set(entry[2] for entry in expert_mapping)
            num_local_experts = len(expert_ids)

        mutate_fn = None
        if hasattr(model, "mutate_weight_preload"):
            mutate_fn = model.mutate_weight_preload

        scale_fn = None
        if hasattr(model, "custom_scale_remap"):
            scale_fn = model.custom_scale_remap

        return cls(
            stacked_params_mapping=stacked_mapping,
            expert_params_mapping=expert_mapping,
            num_local_experts=num_local_experts,
            mutate_weight_preload=mutate_fn,
            custom_scale_remap=scale_fn,
        )
