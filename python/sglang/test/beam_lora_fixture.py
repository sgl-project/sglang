"""Seeded PEFT LoRA adapters for the Beam test fixture, and their merged control.

``make_beam_lora_adapter`` writes random LoRA factors, under PEFT key names, for
the checkpoint that ``make_beam_fixture`` writes. ``merge_beam_lora`` folds the
factors into the base weights, so ``eager_beam_logits`` on the merged weights is
the reference a LoRA engine must reproduce. The factors are not trained: they
bound the LoRA serving path on the fixture, not adapter quality.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import msgspec
import torch
from safetensors.torch import save_file

PEFT_PREFIX = "base_model.model."

# Each group adapts one serving path, so a failure names the path. The attention
# gate keeps its qualified target name even when MLP gates are included.
LORA_TARGET_GROUPS: Dict[str, Tuple[str, ...]] = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "attention_gate": ("self_attn.gate_proj",),
    "dense_mlp": ("gate_proj", "up_proj", "down_proj"),
    "experts": ("gate_proj", "up_proj", "down_proj"),
    "embed_tokens": ("embed_tokens",),
    "lm_head": ("lm_head",),
}

EXPERT_LAYOUTS = ("per_expert", "shared_outer")

# Delta size relative to the adapted weight's standard deviation. The attention
# gate reaches the output through a sigmoid, whose slope is at most 1/4, so it
# gets a larger delta to move the output as much as the linear paths do.
DEFAULT_STRENGTH = 0.5
GROUP_STRENGTH = {"attention_gate": 1.0}

_FACTOR_KEY = re.compile(
    r"^" + re.escape(PEFT_PREFIX) + r"(.+)\.lora_(?:embedding_)?([AB])(?:\.weight)?$"
)
_SHARED_OUTER_MODULE = re.compile(
    r"^(.+\.mlp\.experts)\.(gate_proj|up_proj|down_proj)$"
)


class BeamLoRAAdapter(msgspec.Struct, frozen=True, kw_only=True):
    path: str
    # PEFT key -> BF16 factor, exactly as saved.
    weights: Dict[str, torch.Tensor]
    scaling: float


def _group_modules(config, group: str) -> List[str]:
    layers = range(config.num_hidden_layers)
    dense = [i for i in layers if config.mlp_layer_types[i] == "dense"]
    sparse = [i for i in layers if config.mlp_layer_types[i] == "sparse"]
    if group == "attention":
        return [
            f"model.layers.{i}.self_attn.{name}"
            for i in layers
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
    if group == "attention_gate":
        return [f"model.layers.{i}.self_attn.gate_proj" for i in layers]
    if group == "dense_mlp":
        return [
            f"model.layers.{i}.mlp.{name}"
            for i in dense
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
    if group == "experts":
        return [
            f"model.layers.{i}.mlp.experts.{expert}.{name}"
            for i in sparse
            for expert in range(config.num_experts)
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
    if group == "embed_tokens":
        return ["model.embed_tokens"]
    if group == "lm_head":
        return ["lm_head"]
    raise ValueError(f"unknown LoRA target group {group!r}")


def _target_modules(groups: Iterable[str]) -> List[str]:
    targets = {name for group in groups for name in LORA_TARGET_GROUPS[group]}
    return sorted(targets)


def make_beam_lora_adapter(
    path,
    config,
    state: Dict[str, torch.Tensor],
    groups: Iterable[str],
    *,
    rank: int = 8,
    alpha: int = 16,
    seed: int = 0,
    strength: Optional[float] = None,
    expert_layout: str = "per_expert",
) -> BeamLoRAAdapter:
    """Write a PEFT adapter directory for the fixture and return its factors.

    Each delta ``scaling * B @ A`` has ``strength`` times the standard deviation
    of the base weight it adapts (by default, the group's entry in
    ``GROUP_STRENGTH``), so a dropped or misplaced factor moves the output far
    beyond the fixture's KL bound. ``shared_outer`` stores the expert factors as
    3D tensors whose gate/up A and down B are shared across experts.
    """
    groups = list(groups)
    if expert_layout not in EXPERT_LAYOUTS:
        raise ValueError(f"expert_layout must be one of {EXPERT_LAYOUTS}")
    scaling = alpha / rank
    generator = torch.Generator().manual_seed(seed)

    def group_strength(group):
        if strength is not None:
            return strength
        return GROUP_STRENGTH.get(group, DEFAULT_STRENGTH)

    def factors(a_shape, b_shape, base_weight, group):
        b_std = group_strength(group) * base_weight.float().std().item() / scaling
        lora_a = torch.randn(a_shape, generator=generator) * rank**-0.5
        lora_b = torch.randn(b_shape, generator=generator) * b_std
        return lora_a.bfloat16(), lora_b.bfloat16()

    weights = {}
    shared_outer_experts = set()
    for group in groups:
        for module in _group_modules(config=config, group=group):
            if group == "experts" and expert_layout == "shared_outer":
                layer_prefix, _, name = module.rsplit(".", 2)
                shared_outer_experts.add((layer_prefix, name))
                continue
            base_weight = state[f"{module}.weight"]
            out_features, in_features = base_weight.shape
            if group == "embed_tokens":
                # PEFT embeddings: A [r, vocab] is gathered, B [hidden, r] expands.
                lora_a, lora_b = factors(
                    a_shape=(rank, out_features),
                    b_shape=(in_features, rank),
                    base_weight=base_weight,
                    group=group,
                )
                weights[f"{PEFT_PREFIX}{module}.lora_embedding_A"] = lora_a
                weights[f"{PEFT_PREFIX}{module}.lora_embedding_B"] = lora_b
            else:
                lora_a, lora_b = factors(
                    a_shape=(rank, in_features),
                    b_shape=(out_features, rank),
                    base_weight=base_weight,
                    group=group,
                )
                weights[f"{PEFT_PREFIX}{module}.lora_A.weight"] = lora_a
                weights[f"{PEFT_PREFIX}{module}.lora_B.weight"] = lora_b

    for layer_prefix, name in sorted(shared_outer_experts):
        base_weight = state[f"{layer_prefix}.0.{name}.weight"]
        out_features, in_features = base_weight.shape
        experts = config.num_experts
        # gate/up share A across experts; down shares B.
        a_experts, b_experts = (experts, 1) if name == "down_proj" else (1, experts)
        lora_a, lora_b = factors(
            a_shape=(a_experts, rank, in_features),
            b_shape=(b_experts, out_features, rank),
            base_weight=base_weight,
            group="experts",
        )
        weights[f"{PEFT_PREFIX}{layer_prefix}.{name}.lora_A.weight"] = lora_a
        weights[f"{PEFT_PREFIX}{layer_prefix}.{name}.lora_B.weight"] = lora_b

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "target_modules": _target_modules(groups),
        "modules_to_save": None,
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    return BeamLoRAAdapter(path=str(path), weights=weights, scaling=scaling)


def merge_beam_lora(
    state: Dict[str, torch.Tensor], adapter: BeamLoRAAdapter
) -> Dict[str, torch.Tensor]:
    """Return base weights plus every adapter delta, added in FP32 and rounded once."""
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, factor in adapter.weights.items():
        match = _FACTOR_KEY.match(key)
        if match is None:
            raise ValueError(f"not a PEFT LoRA factor: {key}")
        module, side = match.groups()
        pairs.setdefault(module, {})[side] = factor

    deltas: Dict[str, torch.Tensor] = {}
    for module, factor in pairs.items():
        lora_a, lora_b = factor["A"].float(), factor["B"].float()
        shared_outer = _SHARED_OUTER_MODULE.match(module)
        if shared_outer is not None and lora_a.dim() == 3:
            layer_prefix, name = shared_outer.groups()
            for expert in range(max(lora_a.shape[0], lora_b.shape[0])):
                expert_a = lora_a[expert if lora_a.shape[0] > 1 else 0]
                expert_b = lora_b[expert if lora_b.shape[0] > 1 else 0]
                deltas[f"{layer_prefix}.{expert}.{name}.weight"] = expert_b @ expert_a
        elif module.endswith("embed_tokens"):
            deltas[f"{module}.weight"] = (lora_b @ lora_a).T
        else:
            deltas[f"{module}.weight"] = lora_b @ lora_a

    merged = dict(state)
    for key, delta in deltas.items():
        base = state[key]
        merged[key] = (base.float() + adapter.scaling * delta.to(base.device)).to(
            base.dtype
        )
    return merged
