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

from typing import TYPE_CHECKING, Callable, Dict, Sequence

from transformers import PretrainedConfig

if TYPE_CHECKING:
    import torch

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


# Keyed by HF ``config.model_type`` (loader.py looks it up with that), which is
# not the GGUF ``general.architecture`` that GGUF_NATIVE_CONFIG_BUILDERS uses:
# llama.cpp spells the arch "muse-glimmer" while the HF config says "muse_glimmer".
GGUF_HF_NAME_MAP_BUILDERS: Dict[str, Callable[[PretrainedConfig], Dict[str, str]]] = {
    "muse_glimmer": build_muse_glimmer_name_map,
}


# =============================================================================
# Qwen3.5 / Qwen3-Next (Gated DeltaNet) GGUF support
# =============================================================================
#
# Three things stop a llama.cpp-produced Qwen3.5 GGUF from loading correctly and
# none of them can be expressed as a name map alone:
#
#  1. gguf-py spells the architecture "qwen35" while transformers spells the
#     model_type "qwen3_5" / "qwen3_5_text", and gguf-py's name map cannot
#     resolve ``linear_attn.A_log`` / ``linear_attn.dt_bias`` because
#     ``_get_gguf_weights_map`` splits an HF name on its LAST dot and asks the
#     map about the remainder ("....linear_attn").  Fixed by the builder below.
#
#  2. llama.cpp orders the Gated-DeltaNet VALUE heads differently from HF.  The
#     GDN kernels pair value head ``i`` with key head ``i // (n_v / n_k)``
#     (k-group-major) in SGLang and in transformers, but llama.cpp broadcasts
#     the key/query heads with a plain ``ggml_repeat`` (see
#     ``llama_model_qwen35::graph::build_layer_attn_linear``), which pairs value
#     head ``p`` with key head ``p % n_k``.  llama.cpp's converter therefore
#     stores the value-head-indexed tensors transposed:
#         GGUF value head p  ==  HF value head (p % n_k) * (n_v / n_k) + p // n_k
#     Loading those rows verbatim silently mis-pairs every value head with a
#     key head, which does not crash and does not look wrong in a checksum --
#     it just produces incoherent text.
#
#  3. llama.cpp stores ``blk.N.ssm_a`` already exponentiated and negated,
#     ``-exp(A_log)``, because its graph multiplies the gate by it directly
#     ("-A_log.exp() * softplus" in build_layer_attn_linear).  SGLang's
#     ``fused_gdn_gating`` kernel computes ``-exp(A_log) * softplus`` itself,
#     so the tensor must be turned back into ``A_log = log(-ssm_a)``.
#
# Plus one shape convention: the depthwise causal conv weight is [conv_dim, 1,
# kernel] in HF and [conv_dim, kernel] in GGUF (llama.cpp drops the singleton).
#
# 2 and 3 and the conv rank are weight *values*, not names, so they are applied
# by a weight transform (GGUF_WEIGHT_TRANSFORMS) that GGUFModelLoader runs over
# the weight iterator.

_QWEN3_5_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_next", "qwen3_next_text")

# Qwen3.5 uses a zero-centred RMSNorm everywhere except the Gated-DeltaNet's own
# gated norm: transformers' Qwen3_5RMSNorm.forward is
#     output = _norm(x) * (1.0 + self.weight)
# (modeling_qwen3_5.py:736) and SGLang matches it by building these five norms
# with GemmaRMSNorm (qwen3_5.py:939, 1179-1185, 1652).  llama.cpp has only a
# plain RMS norm, so its converter folds the +1 into the stored weight.  The
# GDN norm (Qwen3_5RMSNormGated / RMSNormGated, plain `self.weight * x`) is
# stored raw and must NOT be touched -- it is the exception that makes this
# defect easy to miss.
_QWEN3_5_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_QWEN3_5_GEMMA_NORM_EXACT = ("model.norm.weight",)

# gguf tensor suffix -> HF parameter suffix, for the two parameters gguf-py's
# name map cannot resolve.
_QWEN3_5_GDN_EXTRA = {
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
}


def build_qwen3_5_name_map(config: PretrainedConfig) -> Dict[str, str]:
    """{gguf_tensor_name: hf_param_name} for qwen3_5 / qwen3_5_text."""
    import gguf
    import torch
    from transformers import AutoModelForCausalLM

    text_config = getattr(config, "text_config", None) or config
    num_layers = text_config.num_hidden_layers

    # transformers' Qwen3_5DecoderLayer reads config.layer_types[layer_idx].  A
    # Qwen3_5TextConfig rebuilt from GGUF metadata (or narrowed from a VL
    # config) can arrive without it; the schedule is fully determined by
    # full_attention_interval.
    if getattr(text_config, "layer_types", None) is None:
        interval = getattr(text_config, "full_attention_interval", 4)
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]

    arch = None
    for key, value in gguf.MODEL_ARCH_NAMES.items():
        if value == "qwen35":
            arch = key
            break
    if arch is None:
        raise RuntimeError(
            "gguf-py does not know the 'qwen35' architecture; a newer gguf "
            f"package is required (have {getattr(gguf, '__version__', 'unknown')})"
        )
    name_map = gguf.get_tensor_name_map(arch, num_layers)

    with torch.device("meta"):
        dummy_model = AutoModelForCausalLM.from_config(text_config)
    state_dict = dummy_model.state_dict()

    gguf_to_hf_name_map: Dict[str, str] = {}
    unresolved = []
    for hf_name in state_dict:
        stem, _, suffix = hf_name.rpartition(".")
        gguf_name = name_map.get_name(stem)
        if gguf_name is None:
            unresolved.append(hf_name)
            continue
        gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name

    for layer in range(num_layers):
        for gguf_suffix, hf_suffix in _QWEN3_5_GDN_EXTRA.items():
            hf_name = f"model.layers.{layer}.{hf_suffix}"
            if hf_name in state_dict:
                gguf_to_hf_name_map[f"blk.{layer}.{gguf_suffix}"] = hf_name
                if hf_name in unresolved:
                    unresolved.remove(hf_name)

    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} Qwen3.5 parameters have no GGUF tensor name, "
            f"e.g. {unresolved[:4]}"
        )
    return gguf_to_hf_name_map


def _permute_head_blocks(
    tensor: "torch.Tensor", index: Sequence[int], block: int, offset: int = 0
):
    """Reorder ``len(index)`` groups of ``block`` consecutive rows.

    Row ``offset + i * block + j`` of the result is row
    ``offset + index[i] * block + j`` of the input.  Rows before ``offset`` are
    left alone.  For a GGUF-quantised tensor the rows are packed bytes and each
    row is self-contained, so this is an exact byte move for every quant type.
    """
    import torch

    body = tensor[offset:]
    n = len(index)
    if body.shape[0] != n * block:
        raise ValueError(
            f"cannot split {body.shape[0]} rows into {n} groups of {block}"
        )
    rest = tuple(body.shape[1:])
    body = body.reshape(n, block, *rest)[list(index)].reshape(n * block, *rest)
    if offset == 0:
        return body
    return torch.cat([tensor[:offset], body], dim=0)


class Qwen3_5GGUFWeightTransform:
    """Turn llama.cpp's Gated-DeltaNet weight conventions into HF's.

    Called with the HF parameter name, the tensor as it comes out of
    ``gguf_quant_weights_iterator`` (packed uint8 rows for a quantised tensor,
    real values for F32) and the GGML type id the loader announced for it.
    """

    def __init__(self, config: PretrainedConfig):
        text_config = getattr(config, "text_config", None) or config
        self.num_k_heads = int(text_config.linear_num_key_heads)
        self.num_v_heads = int(text_config.linear_num_value_heads)
        self.head_k_dim = int(text_config.linear_key_head_dim)
        self.head_v_dim = int(text_config.linear_value_head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        if self.num_v_heads % self.num_k_heads:
            raise ValueError(
                f"linear_num_value_heads {self.num_v_heads} is not a multiple "
                f"of linear_num_key_heads {self.num_k_heads}"
            )
        group = self.num_v_heads // self.num_k_heads
        # HF value head h is stored by llama.cpp at GGUF value head
        # (h % group) * num_k_heads + h // group  -- the inverse of
        # perm[p] = (p % num_k_heads) * group + p // num_k_heads.
        self.gguf_index_of_hf_head = [
            (h % group) * self.num_k_heads + h // group for h in range(self.num_v_heads)
        ]
        self.counts = {}

    def _count(self, what: str):
        self.counts[what] = self.counts.get(what, 0) + 1

    def _permute_columns(self, name: str, tensor, weight_type):
        """Permute value-head groups along the INPUT dim of out_proj."""
        import torch

        index = self.gguf_index_of_hf_head
        if tensor.dtype != torch.uint8:
            # unquantised: columns are real values
            if tensor.shape[1] != self.value_dim:
                raise ValueError(f"{name}: expected {self.value_dim} columns")
            return (
                _permute_head_blocks(
                    tensor.transpose(0, 1).contiguous(), index, self.head_v_dim
                )
                .transpose(0, 1)
                .contiguous()
            )

        import gguf

        quant = gguf.GGMLQuantizationType(int(weight_type))
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant]
        if self.head_v_dim % block_size:
            raise NotImplementedError(
                f"{name}: value head dim {self.head_v_dim} is not a multiple of "
                f"the {quant.name} block size {block_size}, so llama.cpp's "
                "value-head order cannot be undone without requantising. "
                "Re-quantise this tensor with a type whose block size divides "
                f"{self.head_v_dim} (e.g. Q8_0)."
            )
        group_bytes = (self.head_v_dim // block_size) * type_size
        if tensor.shape[1] != self.num_v_heads * group_bytes:
            raise ValueError(
                f"{name}: {tensor.shape[1]} bytes per row, expected "
                f"{self.num_v_heads * group_bytes} for {quant.name}"
            )
        out = tensor.reshape(tensor.shape[0], self.num_v_heads, group_bytes)
        return out[:, list(index), :].reshape(tensor.shape[0], -1).contiguous()

    def __call__(self, name: str, tensor, weight_type):
        import torch

        index = self.gguf_index_of_hf_head

        if name in _QWEN3_5_GEMMA_NORM_EXACT or (
            name.endswith(_QWEN3_5_GEMMA_NORM_SUFFIXES)
            and ".linear_attn.norm." not in name
        ):
            if tensor.dtype == torch.uint8:
                raise NotImplementedError(
                    f"{name}: a zero-centred RMSNorm weight must be loaded "
                    "unquantised so the folded +1 can be removed"
                )
            self._count("gemma_norm")
            return tensor - 1.0

        if ".linear_attn." not in name:
            return tensor

        if name.endswith(".linear_attn.A_log"):
            # llama.cpp stores -exp(A_log); SGLang's gating kernel exponentiates
            # A_log itself.
            tensor = _permute_head_blocks(tensor, index, 1)
            if not bool((tensor < 0).all()):
                raise ValueError(
                    f"{name}: expected every ssm_a entry to be negative "
                    "(-exp(A_log)); this GGUF does not follow llama.cpp's "
                    "qwen35 convention"
                )
            self._count("A_log")
            return torch.log(-tensor.to(torch.float32)).to(tensor.dtype)

        if name.endswith(".linear_attn.dt_bias"):
            self._count("dt_bias")
            return _permute_head_blocks(tensor, index, 1)

        if ".linear_attn.conv1d." in name:
            # rows are [key_dim (q), key_dim (k), value_dim (v)] channels
            tensor = _permute_head_blocks(
                tensor, index, self.head_v_dim, offset=2 * self.key_dim
            )
            if tensor.dim() == 2:
                # HF keeps the depthwise singleton channel dim that llama.cpp drops
                tensor = tensor.unsqueeze(1)
            self._count("conv1d")
            return tensor

        if ".linear_attn.in_proj_qkv." in name:
            self._count("in_proj_qkv")
            return _permute_head_blocks(
                tensor, index, self.head_v_dim, offset=2 * self.key_dim
            )

        if ".linear_attn.in_proj_z." in name:
            self._count("in_proj_z")
            return _permute_head_blocks(tensor, index, self.head_v_dim)

        if ".linear_attn.in_proj_b." in name or ".linear_attn.in_proj_a." in name:
            self._count("in_proj_ba")
            return _permute_head_blocks(tensor, index, 1)

        if ".linear_attn.out_proj." in name:
            self._count("out_proj")
            return self._permute_columns(name, tensor, weight_type)

        return tensor


def build_qwen3_5_weight_transform(config: PretrainedConfig):
    return Qwen3_5GGUFWeightTransform(config)


GGUF_HF_NAME_MAP_BUILDERS.update(
    {model_type: build_qwen3_5_name_map for model_type in _QWEN3_5_MODEL_TYPES}
)

# Keyed like GGUF_HF_NAME_MAP_BUILDERS, by HF ``config.model_type``.  A builder
# returns ``f(hf_param_name, tensor, ggml_type_id) -> tensor``; ``ggml_type_id``
# is None for a tensor the GGUF stores unquantised.
GGUF_WEIGHT_TRANSFORMS: Dict[str, Callable[[PretrainedConfig], Callable]] = {
    model_type: build_qwen3_5_weight_transform for model_type in _QWEN3_5_MODEL_TYPES
}


def get_gguf_weight_transform(config: PretrainedConfig):
    builder = GGUF_WEIGHT_TRANSFORMS.get(config.model_type)
    return None if builder is None else builder(config)


def apply_gguf_weight_transform(weights_iterator, transform):
    """Run ``transform`` over a GGUF weight iterator.

    ``gguf_quant_weights_iterator`` yields every ``*.qweight_type`` first and
    the tensors afterwards, so the GGML type of each quantised tensor is known
    by the time its data arrives.
    """
    weight_types = {}
    for name, tensor in weights_iterator:
        if name.endswith(".qweight_type"):
            weight_types[name[: -len(".qweight_type")]] = int(tensor.reshape(-1)[0])
            yield name, tensor
            continue
        stem, _, suffix = name.rpartition(".")
        yield name, transform(name, tensor, weight_types.get(stem))
