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
"""Read plain DFlash GGUF drafts with F32/F16/BF16 weights.

These drafts share the target embedding and output head.
The public nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash export uses this layout.
LiLiCorr and packed draft weights use different runtimes and are not accepted.
Target layer indices include the embedding output in GGUF, so they are shifted
by one for SGLang's decoder-layer captures.
Detection and config loading share compact metadata; readers are not cached.
"""

from transformers import Qwen3Config

_LAYER_NAMES = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def dflash_gguf_name_map(num_layers):
    names = {
        "fc.weight": "fc.weight",
        "enc.output_norm.weight": "hidden_norm.weight",
        "output_norm.weight": "norm.weight",
    }
    for layer in range(num_layers):
        for source, target in _LAYER_NAMES.items():
            names[f"blk.{layer}.{source}.weight"] = f"layers.{layer}.{target}.weight"
    return names


def _extract_dflash_gguf_metadata(reader):
    tokens = reader.fields.get("tokenizer.ggml.tokens")
    return {
        "fields": {
            key: field.contents()
            for key, field in reader.fields.items()
            if key.startswith(("dflash.", "lilicorr."))
            or key in ("general.architecture", "tokenizer.ggml.mask_token_id")
        },
        "tensors": {
            tensor.name: (tuple(int(size) for size in tensor.shape), tensor.tensor_type)
            for tensor in reader.tensors
        },
        "vocab_size": None if tokens is None else len(tokens.data),
        "token_types": None if tokens is None else tuple(tokens.types),
    }


def is_plain_dflash_gguf(path):
    from gguf import GGMLQuantizationType

    from sglang.srt.utils.hf_transformers.gguf_native import _read_gguf_metadata

    _, metadata = _read_gguf_metadata(path)
    return (
        metadata is not None
        and "dflash.target_layers" in metadata["fields"]
        and not any(key.startswith("lilicorr.") for key in metadata["fields"])
        and bool(metadata["tensors"])
        and all(
            tensor_type
            in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16,
            )
            for _, tensor_type in metadata["tensors"].values()
        )
    )


def _validate_plain_dflash_gguf(metadata):
    from gguf import GGMLQuantizationType

    if metadata is None or any(
        key.startswith("lilicorr.") for key in metadata["fields"]
    ):
        raise ValueError("Expected a plain DFlash GGUF without LiLiCorr metadata")
    meta = metadata["fields"]
    if meta.get("general.architecture") != "dflash":
        raise ValueError("Expected a plain DFlash GGUF without LiLiCorr metadata")
    names = dflash_gguf_name_map(int(meta["dflash.block_count"]))
    actual = metadata["tensors"].keys()
    if actual != names.keys():
        raise ValueError(
            f"Unsupported DFlash GGUF tensors: missing={sorted(names.keys() - actual)}, "
            f"unexpected={sorted(actual - names.keys())}"
        )
    for name, (_, tensor_type) in metadata["tensors"].items():
        if tensor_type not in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16,
        ):
            raise ValueError(f"DFlash GGUF requires F32, F16, or BF16 weights: {name}")
    return names


def build_plain_dflash_gguf_config(path):
    return Qwen3Config(**_read_plain_dflash_gguf_config(path))


def _read_plain_dflash_gguf_config(path) -> dict:
    from gguf import GGUFValueType

    from sglang.srt.utils.hf_transformers.gguf_native import _read_gguf_metadata

    _, metadata = _read_gguf_metadata(path)
    _validate_plain_dflash_gguf(metadata)
    meta = metadata["fields"]
    if metadata["token_types"] is None:
        raise KeyError("tokenizer.ggml.tokens")
    if metadata["token_types"] != (GGUFValueType.ARRAY, GGUFValueType.STRING):
        raise ValueError(
            "DFlash GGUF tokenizer.ggml.tokens must be an array of strings"
        )

    def field(name):
        key = "dflash." + name
        if key not in meta:
            raise ValueError(f"DFlash GGUF is missing metadata {key!r}")
        return meta[key]

    layers = int(field("block_count"))
    # GGUF counts the embedding output; SGLang counts decoder layers.
    targets = [int(i) - 1 for i in field("target_layers")]
    if not targets or min(targets) < 0 or len(set(targets)) != len(targets):
        raise ValueError("DFlash GGUF target layers must be distinct positive indices")
    pattern = field("attention.sliding_window_pattern")
    if len(pattern) != layers:
        raise ValueError("DFlash GGUF sliding-window pattern must cover every layer")
    head_dim = int(field("attention.key_length"))
    if int(field("attention.value_length")) != head_dim:
        raise ValueError("DFlash GGUF requires equal key and value dimensions")
    hidden = int(field("embedding_length"))
    fc_shape, _ = metadata["tensors"]["fc.weight"]
    if fc_shape != (hidden * len(targets), hidden):
        raise ValueError("DFlash GGUF fc shape does not match its target layers")
    return dict(
        architectures=["DFlashDraftModel"],
        hidden_size=hidden,
        intermediate_size=int(field("feed_forward_length")),
        num_hidden_layers=layers,
        num_attention_heads=int(field("attention.head_count")),
        num_key_value_heads=int(field("attention.head_count_kv")),
        head_dim=head_dim,
        rms_norm_eps=float(field("attention.layer_norm_rms_epsilon")),
        max_position_embeddings=int(field("context_length")),
        rope_theta=float(field("rope.freq_base")),
        vocab_size=metadata["vocab_size"],
        is_causal=bool(field("attention.causal")),
        use_sliding_window=any(pattern),
        sliding_window=int(field("attention.sliding_window")),
        layer_types=["sliding_attention" if p else "full_attention" for p in pattern],
        block_size=int(field("block_size")),
        dtype="bfloat16",
        dflash_config={
            "target_layer_ids": targets,
            "mask_token_id": int(meta["tokenizer.ggml.mask_token_id"]),
        },
    )


def plain_dflash_gguf_weights(path):
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(path, mode="r")
    names = _validate_plain_dflash_gguf(_extract_dflash_gguf_metadata(reader))
    for tensor in reader.tensors:
        data = np.array(tensor.data, copy=True)
        if tensor.tensor_type == GGMLQuantizationType.BF16:
            # GGUF stores BF16 as bytes; preserve the bits rather than cast them.
            weight = torch.from_numpy(data.view(np.uint16)).view(torch.bfloat16)
        else:
            weight = torch.from_numpy(data)
        yield names[tensor.name], weight.reshape(tuple(reversed(tensor.shape)))
