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
"""Read DFlash LiLiCorr GGUF exports using the existing NVFP4A16 runtime.

NVFP4 blocks are repacked at load time without dequantization. Dense tensors
keep their exported values. GGUF's materialized lattice bias is converted to
an equivalent slot-relative bias for the existing LiLiCorr head.
"""

import re

from transformers import Qwen3Config

from sglang.srt.utils.hf_transformers.gguf_reader import get_gguf_reader

_BACKBONE_NAMES = {
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_sinks.weight": "self_attn.attention_sink_bias",
    "attn_k_norm": "self_attn.k_norm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k": "self_attn.k_proj",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_conv_base": "attention_conv.base_kernel",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "attn_conv_proj": "attention_conv.kernel_projection",
    "ffn_conv_proj": "mlp_conv.kernel_projection",
}
_HEAD_NAMES = {
    "attn_output": "attn.out_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_up": "mlp.up_proj",
    "attn_norm": "attn_norm",
    "ffn_norm": "mlp_norm",
}
_FEATURE_NAMES = {
    "ffn_down": "feature_mlp.down_proj",
    "ffn_up": "feature_mlp.up_proj",
    "ffn_norm": "feature_norm",
}


def lilicorr_tensor_name(name: str, num_layers: int, num_head_layers: int) -> str:
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if match:
        layer, tail = int(match[1]), match[2]
        if layer < num_layers:
            names, prefix = _BACKBONE_NAMES, f"layers.{layer}."
        elif layer < num_layers + num_head_layers:
            names, prefix = _HEAD_NAMES, f"lilicorr.layers.{layer - num_layers}."
        elif layer == num_layers + num_head_layers:
            names, prefix = _FEATURE_NAMES, "lilicorr."
        else:
            raise ValueError(f"Unexpected LiLiCorr GGUF layer: {name}")
        for source, target in names.items():
            if tail == source or tail.startswith(source + "."):
                return prefix + target + tail[len(source) :]
        raise ValueError(f"Unknown LiLiCorr GGUF tensor: {name}")
    if name == "enc.output_norm.weight":
        return "hidden_norm.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if name in ("lilicorr.rank_embedding.weight", "lilicorr.slot_embedding.weight"):
        return name.removesuffix(".weight")
    if name not in ("fc.weight", "fc.scale") and not name.startswith("lilicorr."):
        raise ValueError(f"Unknown LiLiCorr GGUF tensor: {name}")
    return name.replace(".attn.in_proj.weight", ".attn.in_proj_weight").replace(
        ".attn.in_proj.bias", ".attn.in_proj_bias"
    )


def build_lilicorr_gguf_config(path: str) -> Qwen3Config:
    from gguf import GGMLQuantizationType

    reader = get_gguf_reader(path)

    def field(key):
        if key not in reader.fields:
            raise ValueError(f"LiLiCorr GGUF is missing metadata {key!r}")
        return reader.fields[key].contents()

    if field("general.architecture") != "dflash":
        raise ValueError("Expected a DFlash GGUF checkpoint")
    activation_precision = reader.fields.get("general.tensor_extra.prec_a4")
    if activation_precision is not None and any(activation_precision.contents()):
        raise ValueError("LiLiCorr GGUF loading requires 16-bit activations")
    layers = int(field("dflash.block_count"))
    head_layers = int(field("lilicorr.num_layers"))
    hidden = int(field("lilicorr.hidden_size"))
    block = int(field("dflash.block_size"))
    targets = [int(i) - 1 for i in field("dflash.target_layers")]
    # GGUF counts the embedding output; SGLang captures decoder layer inputs.
    if not targets or min(targets) < 0:
        raise ValueError("LiLiCorr GGUF target layers must be positive")
    dflash = {
        "mask_token_id": int(field("tokenizer.ggml.mask_token_id")),
        "target_layer_ids": targets,
        "causal": bool(field("dflash.attention.causal")),
        "attention_sink_bias": any(".attn_sinks." in t.name for t in reader.tensors),
        "projector_type": "lilicorr",
        "lilicorr_enabled": True,
        "lilicorr_mlp_ratio": int(field("lilicorr.mlp_hidden_size")) / hidden,
    }
    for key in ("conv_kernel_size", "conv_group_size"):
        dflash[key] = int(field("dflash." + key))
    for key in (
        "hidden_size",
        "candidate_topk",
        "num_heads",
        "num_layers",
        "factor_dim",
        "vector_eps",
        "logit_scale",
    ):
        dflash["lilicorr_" + key] = field("lilicorr." + key)
    excluded = []
    for tensor in reader.tensors:
        name = lilicorr_tensor_name(tensor.name, layers, head_layers)
        if tensor.tensor_type not in (
            GGMLQuantizationType.NVFP4,
            GGMLQuantizationType.F32,
            GGMLQuantizationType.BF16,
        ):
            raise ValueError(
                f"Unsupported LiLiCorr GGUF tensor type: {tensor.tensor_type}"
            )
        if (
            tensor.name.endswith(".weight")
            and tensor.tensor_type != GGMLQuantizationType.NVFP4
        ):
            # Keep dense weights out of the NVFP4 path.
            excluded.append(name.removesuffix(".weight"))
    window = int(field("dflash.attention.sliding_window"))
    pattern = field("dflash.attention.sliding_window_pattern")
    if len(pattern) != layers:
        raise ValueError("LiLiCorr GGUF sliding-window pattern must cover each layer")
    return Qwen3Config(
        architectures=["LiLiCorrDraftModel"],
        hidden_size=int(field("dflash.embedding_length")),
        intermediate_size=int(field("dflash.feed_forward_length")),
        num_hidden_layers=layers,
        num_attention_heads=int(field("dflash.attention.head_count")),
        num_key_value_heads=int(field("dflash.attention.head_count_kv")),
        head_dim=int(field("dflash.attention.key_length")),
        max_position_embeddings=int(field("dflash.context_length")),
        rms_norm_eps=float(field("dflash.attention.layer_norm_rms_epsilon")),
        rope_theta=float(field("dflash.rope.freq_base")),
        vocab_size=len(field("tokenizer.ggml.tokens")),
        layer_types=[
            "sliding_attention" if sliding else "full_attention" for sliding in pattern
        ],
        sliding_window=window,
        use_sliding_window=any(pattern),
        block_size=block,
        dtype="bfloat16",
        dflash_config=dflash,
        quantization_config={
            "quant_method": "modelopt_fp4",
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
            "ignore": excluded,
        },
    )


def repack_nvfp4(data, shape):
    import numpy as np
    import torch

    if len(shape) != 2 or int(shape[0]) % 64:
        raise ValueError(
            f"NVFP4 GGUF requires a matrix with K divisible by 64, got {shape}"
        )
    k, rows = map(int, shape)
    blocks = np.asarray(data).reshape(rows, k // 64, 36)
    scales = blocks[:, :, :4].copy().reshape(rows, k // 16)
    if np.any(scales > 127):
        raise ValueError("NVFP4 GGUF scales must use unsigned E4M3")
    scales[scales == 127] = 0
    packed = blocks[:, :, 4:].reshape(rows, k // 16, 8)
    # GGUF packs the two halves of each group; ModelOpt packs adjacent values.
    codes = np.concatenate([packed & 15, packed >> 4], axis=-1)
    weight = (codes[..., ::2] | (codes[..., 1::2] << 4)).reshape(rows, k // 2)
    return torch.from_numpy(weight.copy()), torch.from_numpy(scales).view(
        torch.float8_e4m3fn
    )


def slot_bias_from_gguf(bias, block_size: int, topk: int):
    import torch

    slots = block_size - 1
    if tuple(bias.shape[1:]) != (slots * topk, slots * topk):
        raise ValueError(f"Unexpected LiLiCorr GGUF attention bias shape: {bias.shape}")
    relative = bias.new_zeros((bias.shape[0], 2 * block_size - 1))
    for offset in range(-(slots - 1), slots):
        relative[:, offset + block_size - 1] = bias[
            :, max(offset, 0) * topk, max(-offset, 0) * topk
        ]
    ids = torch.arange(slots).repeat_interleave(topk)
    rebuilt = relative[:, ids[:, None] - ids[None, :] + block_size - 1]
    if not torch.equal(rebuilt, bias):
        raise ValueError("LiLiCorr GGUF attention bias is not slot-relative")
    # The export already combines the relative and same-slot terms.
    return relative, bias.new_zeros(bias.shape[0])


def lilicorr_gguf_weights(path: str, config):
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType

    reader = get_gguf_reader(path)
    head = config.dflash_config
    layers, head_layers = config.num_hidden_layers, head["lilicorr_num_layers"]
    tensors = {t.name: t for t in reader.tensors}
    for tensor in reader.tensors:
        name = lilicorr_tensor_name(tensor.name, layers, head_layers)
        if name.endswith(".scale"):
            continue
        if tensor.tensor_type == GGMLQuantizationType.NVFP4:
            weight, scale = repack_nvfp4(tensor.data, tensor.shape)
            scale_tensor = tensors.get(tensor.name.removesuffix(".weight") + ".scale")
            if (
                scale_tensor is None
                or scale_tensor.tensor_type != GGMLQuantizationType.F32
            ):
                raise ValueError(
                    f"NVFP4 GGUF tensor {tensor.name} requires an F32 global scale"
                )
            global_scale = torch.from_numpy(
                np.array(scale_tensor.data, copy=True)
            ).reshape(())
            if not torch.isfinite(global_scale) or global_scale < 0:
                raise ValueError(f"Invalid NVFP4 GGUF global scale for {tensor.name}")
            yield name, weight
            yield name.removesuffix(".weight") + ".weight_scale", scale
            yield name.removesuffix(".weight") + ".weight_scale_2", global_scale
            continue
        value = torch.from_numpy(np.array(tensor.data, copy=True))
        if tensor.tensor_type == GGMLQuantizationType.BF16:
            value = value.view(torch.bfloat16)
        elif tensor.tensor_type != GGMLQuantizationType.F32:
            raise ValueError(
                f"Unsupported LiLiCorr GGUF tensor type: {tensor.tensor_type}"
            )
        if name == "lilicorr.attention_bias.bias":
            relative, same = slot_bias_from_gguf(
                value, config.block_size, head["lilicorr_candidate_topk"]
            )
            yield "lilicorr.relative_slot_bias", relative
            yield "lilicorr.same_slot_bias", same
        elif name == "lilicorr.slot_embedding":
            yield name, value.reshape(1, 1, config.block_size - 1, 1, -1)
        elif name == "lilicorr.rank_embedding":
            yield name, value.reshape(1, 1, 1, head["lilicorr_candidate_topk"], -1)
        else:
            yield name, value
