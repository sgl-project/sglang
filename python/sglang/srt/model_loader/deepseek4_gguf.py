# SPDX-License-Identifier: Apache-2.0
"""Exact GGUF tensor-name mapping for DeepSeek-V4 checkpoints.

DeepSeek-V4 GGUF files use the ``deepseek4`` architecture label, while the
current gguf Python package only exposes the closely related DeepSeek-V2 name
map.  The shared entries are sufficient for most tensors; V4-only attention
compressor and mHC tensors are handled explicitly below.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

_ROUTED_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<role>gate|up|down)_exps\.weight$"
)
_TENSOR_SUFFIXES = frozenset(("weight", "bias", "scale", "tid2eid"))


def routed_expert_tensor(name: str) -> tuple[int, str] | None:
    """Return ``(layer, role)`` for an aggregated routed-expert tensor."""

    match = _ROUTED_EXPERT_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group("layer")), match.group("role")


def _split_suffix(name: str) -> tuple[str, str]:
    base, separator, suffix = name.rpartition(".")
    if separator and suffix in _TENSOR_SUFFIXES:
        return base, suffix
    return name, ""


def _v4_checkpoint_name(name: str) -> str | None:
    base, suffix = _split_suffix(name)
    suffix_part = f".{suffix}" if suffix else ""

    top_level = {
        "token_embd": "embed",
        "output": "head",
        "output_norm": "norm",
    }
    if base in top_level:
        return top_level[base] + suffix_part

    match = re.fullmatch(r"output_hc_(base|fn|scale)", base)
    if match:
        # These are direct nn.Parameters.  llama.cpp adds the .weight alias
        # when reading converted four-expert files, but SGLang does not.
        return f"hc_head_{match.group(1)}"

    match = re.fullmatch(r"blk\.(\d+)\.(.+)", base)
    if match:
        layer, tensor = match.groups()

        direct_parameter = {
            "attn_sinks": "attn.attn_sink",
            "ffn_gate_tid2eid": "ffn.gate.tid2eid",
            "hc_attn_base": "hc_attn_base",
            "hc_attn_fn": "hc_attn_fn",
            "hc_attn_scale": "hc_attn_scale",
            "hc_ffn_base": "hc_ffn_base",
            "hc_ffn_fn": "hc_ffn_fn",
            "hc_ffn_scale": "hc_ffn_scale",
        }
        if tensor in direct_parameter:
            return f"layers.{layer}.{direct_parameter[tensor]}"

        linear_or_norm = {
            "attn_kv": "attn.wkv",
            "attn_kv_a_norm": "attn.kv_norm",
            "attn_norm": "attn_norm",
            "attn_output_a": "attn.wo_a",
            "attn_output_b": "attn.wo_b",
            "attn_q_a": "attn.wq_a",
            "attn_q_a_norm": "attn.q_norm",
            "attn_q_b": "attn.wq_b",
            "ffn_down_exps": "ffn.experts.w2",
            "ffn_down_shexp": "ffn.shared_experts.w2",
            "ffn_gate_exps": "ffn.experts.w1",
            "ffn_gate_inp": "ffn.gate",
            "ffn_gate_shexp": "ffn.shared_experts.w1",
            "ffn_norm": "ffn_norm",
            "ffn_up_exps": "ffn.experts.w3",
            "ffn_up_shexp": "ffn.shared_experts.w3",
            "indexer.attn_q_b": "attn.indexer.wq_b",
            "indexer.proj": "attn.indexer.weights_proj",
        }
        if tensor in linear_or_norm:
            return f"layers.{layer}.{linear_or_norm[tensor]}{suffix_part}"

        match = re.fullmatch(r"(attn|indexer)_compressor_(ape|gate|kv|norm)", tensor)
        if match:
            owner, component = match.groups()
            owner_part = "attn" if owner == "attn" else "attn.indexer"
            if component == "ape":
                # Compressor.ape is a direct nn.Parameter.
                return f"layers.{layer}.{owner_part}.compressor.ape"
            component = {"gate": "wgate", "kv": "wkv"}.get(component, component)
            return f"layers.{layer}.{owner_part}.compressor.{component}{suffix_part}"

        if tensor == "exp_probs_b":
            return f"layers.{layer}.ffn.gate{suffix_part}"

    return None


def _candidate_score(alias: str) -> tuple[int, int, str]:
    # DeepSeek's native checkpoint aliases use layers.N.attn/ffn.  Selecting
    # them keeps the downstream DeepSeek-V4 remapper authoritative.
    if alias.startswith("layers.") and (".attn." in alias or ".ffn." in alias):
        priority = 0
    elif alias.startswith("layers."):
        priority = 1
    elif alias.startswith("model.layers."):
        priority = 2
    else:
        priority = 3
    return priority, len(alias), alias


def build_deepseek4_checkpoint_name_map(
    gguf_module: Any,
    tensor_names: Iterable[str],
    num_layers: int,
) -> dict[str, str]:
    """Map every source GGUF tensor to a DeepSeek checkpoint tensor name.

    The function fails closed if a source tensor has no deterministic mapping
    or if two source tensors would load the same checkpoint tensor.
    """

    try:
        arch = gguf_module.MODEL_ARCH.DEEPSEEK2
    except AttributeError as exc:
        raise RuntimeError(
            "gguf package does not provide the DeepSeek name map"
        ) from exc

    name_map = gguf_module.get_tensor_name_map(arch, num_layers)
    aliases_by_gguf_base: dict[str, list[str]] = defaultdict(list)
    for alias, mapping in name_map.mapping.items():
        aliases_by_gguf_base[mapping[1]].append(alias)

    result: dict[str, str] = {}
    reverse: dict[str, str] = {}
    missing: list[str] = []
    for tensor_name in tensor_names:
        checkpoint_name = _v4_checkpoint_name(tensor_name)
        if checkpoint_name is None:
            base, suffix = _split_suffix(tensor_name)
            candidates = aliases_by_gguf_base.get(base, ())
            if candidates:
                alias = min(candidates, key=_candidate_score)
                checkpoint_name = alias
                if suffix and not alias.endswith(f".{suffix}"):
                    checkpoint_name += f".{suffix}"

        if checkpoint_name is None:
            missing.append(tensor_name)
            continue
        if checkpoint_name in reverse:
            other = reverse[checkpoint_name]
            raise RuntimeError(
                "DeepSeek-V4 GGUF mapping collision: "
                f"{other!r} and {tensor_name!r} -> {checkpoint_name!r}"
            )
        result[tensor_name] = checkpoint_name
        reverse[checkpoint_name] = tensor_name

    if missing:
        preview = ", ".join(repr(name) for name in missing[:8])
        raise RuntimeError(
            f"No DeepSeek-V4 checkpoint mapping for {len(missing)} GGUF tensors: "
            f"{preview}"
        )
    return result
