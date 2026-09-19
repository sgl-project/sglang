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
# ==============================================================================
"""Per-architecture GGUF -> HF tensor name maps.

``GGUFModelLoader`` normally derives this map from ``gguf.get_tensor_name_map``,
which only covers architectures upstream gguf-py knows, and from a meta-device
``AutoModelForCausalLM.from_config`` to enumerate the HF parameter names. Neither
works for an architecture that lives outside transformers, so those are supplied
here instead.

A builder returns the complete ``{gguf_tensor_name: hf_param_name}`` map. Any
GGUF tensor left out of the map is skipped by ``gguf_quant_weights_iterator``,
which is how dummy tensors are dropped.
"""

import glob
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict

from transformers import PretrainedConfig

# Sandwich naming: ffn_norm is the pre-FFN norm.
_MUSE_GLIMMER_LAYER_TENSORS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attn_norm",
    "ffn_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_ffn_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_gate": "self_attn.output_gate_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

_MUSE_GLIMMER_GLOBAL_TENSORS = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",
}

# attn_q_norm/attn_k_norm omitted: Muse Glimmer's QK-norm is non-parametric.


def build_muse_glimmer_name_map(config: PretrainedConfig, *_: Any) -> Dict[str, str]:
    name_map = {
        f"{gguf}.weight": f"{hf}.weight"
        for gguf, hf in _MUSE_GLIMMER_GLOBAL_TENSORS.items()
    }
    for layer in range(config.num_hidden_layers):
        for gguf, hf in _MUSE_GLIMMER_LAYER_TENSORS.items():
            name_map[f"blk.{layer}.{gguf}.weight"] = f"model.layers.{layer}.{hf}.weight"
    return name_map


def _sidecar_weight_names(gguf_path: str) -> list[str]:
    """Read the HF parameter namespace used by a GGUF sidecar checkpoint."""
    sidecar_dir = Path(gguf_path).parent

    index_path = sidecar_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            return list(json.load(f)["weight_map"])

    from safetensors import safe_open

    shard_paths = glob.glob(str(sidecar_dir / "*.safetensors"))
    if not shard_paths:
        raise RuntimeError(
            "GGUF Qwen3.5/Gemma4 loading requires a standard Hugging Face "
            "sidecar next to the GGUF file. Place config.json and either "
            f"model.safetensors.index.json or safetensors shards in {sidecar_dir}."
        )
    names = []
    for path in shard_paths:
        with safe_open(path, framework="pt") as shard:
            names.extend(shard.keys())
    return names


def _text_tower_weight_names(gguf_path: str) -> list[str]:
    """Keep the text tower, preserving text-only checkpoint names too."""
    names = _sidecar_weight_names(gguf_path)
    multimodal_names = [
        name for name in names if name.startswith("model.language_model.")
    ]
    if multimodal_names:
        return multimodal_names + [
            name for name in names if name.startswith("lm_head.")
        ]
    return [
        name
        for name in names
        if name.startswith("model.") or name.startswith("lm_head.")
    ]


def _text_level_name(name: str) -> str:
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "model." + name[len(prefix) :]
    return name


def build_qwen35_name_map(
    config: PretrainedConfig, gguf: Any, arch: Any, gguf_path: str
) -> Dict[str, str]:
    """Build a Qwen3.5 GGUF map without constructing a Transformers meta model."""
    text_config = getattr(config, "text_config", config)
    name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)
    result = {}
    for hf_name in _text_tower_weight_names(gguf_path):
        text_name = _text_level_name(hf_name)
        # qwen3_5_text.Qwen3_5ForCausalLM strips one leading ``model.`` before
        # handing weights to the shared body, whose native namespace starts at
        # ``layers.*`` / ``embed_tokens.*``.
        runtime_name = (
            text_name if hf_name.startswith("model.language_model.") else hf_name
        )
        gdn_match = re.match(
            r"model\.layers\.(\d+)\.linear_attn\.(A_log|dt_bias)$", text_name
        )
        if gdn_match:
            layer, component = gdn_match.groups()
            gguf_name = (
                f"blk.{layer}.ssm_a"
                if component == "A_log"
                else f"blk.{layer}.ssm_dt.bias"
            )
        else:
            base, _, suffix = text_name.rpartition(".")
            mapped_base = name_map.get_name(base)
            if mapped_base is None:
                continue
            gguf_name = f"{mapped_base}.{suffix}"
        result[gguf_name] = runtime_name
    return result


def build_gemma4_name_map(
    config: PretrainedConfig, gguf: Any, arch: Any, gguf_path: str
) -> Dict[str, str]:
    """Build the text-only Gemma4 GGUF map from an HF sidecar namespace."""
    text_config = getattr(config, "text_config", config)
    name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)
    result = {}
    for hf_name in _text_tower_weight_names(gguf_path):
        if hf_name.endswith((".weight_scale", ".weight_scale_inv", ".input_scale")):
            continue
        if hf_name.endswith(".qweight"):
            hf_name = hf_name[: -len(".qweight")] + ".weight"
        text_name = _text_level_name(hf_name)
        router_match = re.match(
            r"model\.layers\.(\d+)\.router\.(scale|per_expert_scale)$", text_name
        )
        if router_match:
            layer, component = router_match.groups()
            gguf_name = (
                f"blk.{layer}.ffn_gate_inp.scale"
                if component == "scale"
                else f"blk.{layer}.ffn_down_exps.scale"
            )
        else:
            base, _, suffix = text_name.rpartition(".")
            mapped_base = name_map.get_name(base)
            if mapped_base is not None:
                gguf_name = f"{mapped_base}.{suffix}"
            else:
                mapped_base = name_map.get_name(text_name)
                if mapped_base is None:
                    continue
                gguf_name = f"{mapped_base}.weight"
        result[gguf_name] = hf_name
    return result


# Keyed by HF ``config.model_type`` (loader.py looks it up with that), which is
# not the GGUF ``general.architecture`` that GGUF_NATIVE_CONFIG_BUILDERS uses:
# llama.cpp spells the arch "muse-glimmer" while the HF config says "muse_glimmer".
GGUF_HF_NAME_MAP_BUILDERS: Dict[str, Callable[..., Dict[str, str]]] = {
    "muse_glimmer": build_muse_glimmer_name_map,
    "qwen3_5": build_qwen35_name_map,
    "qwen3_5_text": build_qwen35_name_map,
    "qwen3_5_moe": build_qwen35_name_map,
    "qwen3_5_moe_text": build_qwen35_name_map,
    "gemma4": build_gemma4_name_map,
    "gemma4_text": build_gemma4_name_map,
}
