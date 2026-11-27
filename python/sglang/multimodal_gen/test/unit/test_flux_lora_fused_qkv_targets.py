# SPDX-License-Identifier: Apache-2.0
"""FLUX LoRA Q/K/V weights land on split layers in bf16 and on fused QKV otherwise."""

from sglang.multimodal_gen.configs.models.dits.flux import FluxArchConfig
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
    resolve_lora_target_name,
)

ATTN = "transformer_blocks.0.attn"


def _targets(layers: list[str]) -> set[str]:
    return {f"{ATTN}.{m}.{s}" for m in layers for s in ("lora_A", "lora_B", "alpha")}


def _resolve(name: str, layers: list[str]):
    targets = _targets(layers)
    fn = get_param_names_mapping(
        FluxArchConfig().param_names_mapping, valid_target_names=targets
    )
    return resolve_lora_target_name(fn, name, targets)


SPLIT = ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"]
FUSED = ["to_qkv", "to_added_qkv"]


def test_split_qkv_dit_keeps_split_lora_targets():
    for layer in SPLIT:
        name = f"{ATTN}.{layer}.lora_A"
        assert _resolve(name, SPLIT) == (name, None, None)
    # Diffusers files carry a `transformer.` prefix that the mapping strips.
    assert _resolve(f"transformer.{ATTN}.to_k.lora_B", SPLIT) == (
        f"{ATTN}.to_k.lora_B",
        None,
        None,
    )


def test_fused_qkv_dit_stacks_split_lora_into_fused_layers():
    assert _resolve(f"{ATTN}.to_v.lora_A", FUSED) == (f"{ATTN}.to_qkv.lora_A", 2, 3)
    assert _resolve(f"{ATTN}.add_k_proj.lora_B", FUSED) == (
        f"{ATTN}.to_added_qkv.lora_B",
        1,
        3,
    )


def test_base_weights_are_not_redirected():
    fn = get_param_names_mapping(FluxArchConfig().param_names_mapping)
    assert fn(f"{ATTN}.to_q.weight") == (f"{ATTN}.to_q.weight", None, None)
