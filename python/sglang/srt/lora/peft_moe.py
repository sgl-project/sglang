# Copyright 2023-2026 SGLang Team
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

"""Convert PEFT ParamWrapper factors to SGLang's per-expert LoRA format."""

import re
from collections.abc import Mapping

import torch

# ParamWrapper nests another wrapper under base_layer when multiple parameters
# on one experts module are targeted. The nesting order is not a reliable way
# to identify the parameter. Named modules and existing 3D formats do not match.
_PARAM_WRAPPER = re.compile(
    r"(?P<experts>.+\.experts)(?P<wrappers>(?:\.base_layer)*)"
    r"\.lora_(?P<factor>[AB])(?P<adapter>\.[^.]+)?\.weight"
)


def _config_value(config, name):
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def normalize_peft_moe_weights(
    weights: dict[str, torch.Tensor], base_hf_config, adapter_config: dict
) -> None:
    """Unpack unnamed 2D expert factors, before gate/up stacking and scaling.

    Supports fused gate_up_proj and down_proj of gated MoE layers. Both factor
    shapes must uniquely identify the projection and its orientation. PEFT
    0.18 treats 3D parameters as [experts, in, out], which also produces valid
    adapters for HF weights stored as [experts, out, in], but with transposed
    factors. Keep both layouts and preserve their effective weight update.

    Validate all pairs before replacing any input entries. Existing named
    expert factors, ordinary linears and shared experts stay on the old path.
    """
    groups = {}
    for name in weights:
        match = _PARAM_WRAPPER.fullmatch(name)
        if match is None:
            continue
        key = (match["experts"], match["wrappers"], match["adapter"] or "")
        groups.setdefault(key, {})[match["factor"]] = name
    if not groups:
        return

    rank = adapter_config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("PEFT MoE ParamWrapper requires a positive integer r")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise ValueError(
            "PEFT MoE ParamWrapper rank_pattern/alpha_pattern are not supported"
        )
    targets = adapter_config.get("target_parameters")
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ValueError("PEFT MoE ParamWrapper requires a list of target_parameters")

    config = base_hf_config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    elif _config_value(config, "text_config") is not None:
        config = _config_value(config, "text_config")
    experts = _config_value(config, "num_experts")
    if experts is None:
        experts = _config_value(config, "num_local_experts")
    hidden = _config_value(config, "hidden_size")
    intermediate = _config_value(config, "moe_intermediate_size")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (experts, hidden, intermediate)
    ):
        raise ValueError(
            "PEFT MoE ParamWrapper requires num_experts, hidden_size and "
            "moe_intermediate_size in the base model text config"
        )

    # (projection, input features, output features), in serving orientation.
    projections = (
        ("gate_up_proj", hidden, 2 * intermediate),
        ("down_proj", intermediate, hidden),
    )
    replacements = {}
    consumed = []
    for (prefix, _, adapter), factors in groups.items():
        if set(factors) != {"A", "B"}:
            raise ValueError(
                f"Incomplete PEFT MoE ParamWrapper A/B pair: {list(factors.values())}"
            )
        a, b = weights[factors["A"]], weights[factors["B"]]
        candidates = []
        for projection, in_features, out_features in projections:
            parameter = f"{prefix}.{projection}"
            if not any(
                parameter == target or parameter.endswith(f".{target}")
                for target in targets
            ):
                continue
            if a.shape == (experts * rank, in_features) and b.shape == (
                out_features,
                rank * experts,
            ):
                candidates.append((projection, False))
            if a.shape == (experts * rank, out_features) and b.shape == (
                in_features,
                rank * experts,
            ):
                candidates.append((projection, True))
        if len(candidates) != 1:
            raise ValueError(
                "Cannot uniquely resolve PEFT MoE ParamWrapper projection/layout "
                f"for {factors['A']} {tuple(a.shape)} and "
                f"{factors['B']} {tuple(b.shape)}; candidates={candidates}. "
                "Check target_parameters and base model dimensions."
            )

        projection, transposed = candidates[0]
        unpacked_a = a.reshape(experts, rank, a.shape[1])
        # PEFT packs B's expert index inside the rank index (r, E), not (E, r).
        unpacked_b = b.reshape(b.shape[0], rank, experts).permute(2, 0, 1)
        if transposed:
            normalized_a = unpacked_b.transpose(1, 2)
            normalized_b = unpacked_a.transpose(1, 2)
        else:
            normalized_a, normalized_b = unpacked_a, unpacked_b
        for factor, tensor in (("A", normalized_a), ("B", normalized_b)):
            name = f"{prefix}.{projection}.lora_{factor}{adapter}.weight"
            if name in weights or name in replacements:
                raise ValueError(f"PEFT MoE ParamWrapper destination collision: {name}")
            replacements[name] = tensor.contiguous()
        consumed.extend(factors.values())

    for name in consumed:
        del weights[name]
    weights.update(replacements)
