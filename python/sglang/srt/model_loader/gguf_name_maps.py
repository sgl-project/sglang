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

from typing import Callable, Dict

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


def build_muse_glimmer_name_map(config: PretrainedConfig) -> Dict[str, str]:
    name_map = {
        f"{gguf}.weight": f"{hf}.weight"
        for gguf, hf in _MUSE_GLIMMER_GLOBAL_TENSORS.items()
    }
    for layer in range(config.num_hidden_layers):
        for gguf, hf in _MUSE_GLIMMER_LAYER_TENSORS.items():
            name_map[f"blk.{layer}.{gguf}.weight"] = f"model.layers.{layer}.{hf}.weight"
    return name_map


_QWEN3_NEXT_GLOBAL_TENSORS = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",
}

# Present on every Qwen3-Next layer (GDN and full-attention alike).
# Routed-expert packed tensors (``ffn_{gate,up,down}_exps``) are deliberately
# absent: ``gguf_quant_weights_iterator`` splits them per expert before the
# name map is consulted.
_QWEN3_NEXT_COMMON_LAYER_TENSORS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
}

_QWEN3_NEXT_FULL_ATTN_LAYER_TENSORS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
}

# Gated-DeltaNet layers: llama.cpp stores the linear-attention block under
# ``ssm_*`` names. ``ssm_in`` is the fused qkvz in-projection, ``ssm_ba`` the
# fused b/a in-projection, ``ssm_conv1d`` the causal conv over qkv.
_QWEN3_NEXT_GDN_LAYER_TENSORS = {
    "ssm_in": "linear_attn.in_proj_qkvz",
    "ssm_ba": "linear_attn.in_proj_ba",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_norm": "linear_attn.norm",
    "ssm_out": "linear_attn.out_proj",
}

# GDN tensors whose gguf/hf names do not both end in ``.weight``: the decay
# matrix ``ssm_a`` has no suffix on either side, and ``ssm_dt`` is a bias.
_QWEN3_NEXT_GDN_RAW_TENSORS = {
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
}


def _is_qwen3_next_full_attn_layer(layer: int, config: PretrainedConfig) -> bool:
    interval = getattr(config, "full_attention_interval", 4) or 4
    return layer % interval == interval - 1


def build_qwen3_next_name_map(config: PretrainedConfig) -> Dict[str, str]:
    """GGUF -> HF tensor map for Qwen3-Next (hybrid GDN + full-attention MoE).

    The generic ``gguf.get_tensor_name_map`` lookup used by
    ``GGUFModelLoader`` cannot cover this architecture: it is queried with the
    suffix-stripped base name, which misses the two non-``.weight`` GDN
    tensors (``blk.N.ssm_a`` -> ``linear_attn.A_log``,
    ``blk.N.ssm_dt.bias`` -> ``linear_attn.dt_bias``). This builder returns the
    complete map for every layer, keyed by HF ``config.model_type ==
    "qwen3_next"``.
    """
    name_map = {
        f"{gguf}.weight": f"{hf}.weight"
        for gguf, hf in _QWEN3_NEXT_GLOBAL_TENSORS.items()
    }
    for layer in range(config.num_hidden_layers):
        for gguf, hf in _QWEN3_NEXT_COMMON_LAYER_TENSORS.items():
            name_map[f"blk.{layer}.{gguf}.weight"] = f"model.layers.{layer}.{hf}.weight"
        if _is_qwen3_next_full_attn_layer(layer, config):
            tensors = _QWEN3_NEXT_FULL_ATTN_LAYER_TENSORS
            for gguf, hf in tensors.items():
                name_map[f"blk.{layer}.{gguf}.weight"] = (
                    f"model.layers.{layer}.{hf}.weight"
                )
        else:
            for gguf, hf in _QWEN3_NEXT_GDN_LAYER_TENSORS.items():
                name_map[f"blk.{layer}.{gguf}.weight"] = (
                    f"model.layers.{layer}.{hf}.weight"
                )
            for gguf, hf in _QWEN3_NEXT_GDN_RAW_TENSORS.items():
                name_map[f"blk.{layer}.{gguf}"] = f"model.layers.{layer}.{hf}"
    return name_map


# Keyed by HF ``config.model_type`` (loader.py looks it up with that), which is
# not the GGUF ``general.architecture`` that GGUF_NATIVE_CONFIG_BUILDERS uses:
# llama.cpp spells the arch "muse-glimmer" while the HF config says "muse_glimmer".
GGUF_HF_NAME_MAP_BUILDERS: Dict[str, Callable[[PretrainedConfig], Dict[str, str]]] = {
    "muse_glimmer": build_muse_glimmer_name_map,
    "qwen3_next": build_qwen3_next_name_map,
}
