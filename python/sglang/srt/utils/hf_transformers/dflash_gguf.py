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
"""

from transformers import Qwen3Config

from sglang.srt.utils.hf_transformers.gguf_reader import get_gguf_reader

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


def is_plain_dflash_gguf(path):
    from gguf import GGMLQuantizationType

    reader = get_gguf_reader(path)
    arch = reader.fields.get("general.architecture")
    return (
        arch is not None
        and arch.contents() == "dflash"
        and "dflash.target_layers" in reader.fields
        and not any(key.startswith("lilicorr.") for key in reader.fields)
        and bool(reader.tensors)
        and all(
            tensor.tensor_type
            in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16,
            )
            for tensor in reader.tensors
        )
    )


def _read_plain_dflash_gguf(path):
    from gguf import GGMLQuantizationType

    reader = get_gguf_reader(path)
    meta = {
        key: field.contents()
        for key, field in reader.fields.items()
        if key.startswith(("dflash.", "lilicorr."))
        or key in ("general.architecture", "tokenizer.ggml.mask_token_id")
    }
    if meta.get("general.architecture") != "dflash" or any(
        key.startswith("lilicorr.") for key in meta
    ):
        raise ValueError("Expected a plain DFlash GGUF without LiLiCorr metadata")
    names = dflash_gguf_name_map(int(meta["dflash.block_count"]))
    actual = {tensor.name for tensor in reader.tensors}
    if actual != names.keys():
        raise ValueError(
            f"Unsupported DFlash GGUF tensors: missing={sorted(names.keys() - actual)}, "
            f"unexpected={sorted(actual - names.keys())}"
        )
    for tensor in reader.tensors:
        if tensor.tensor_type not in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16,
        ):
            raise ValueError(
                f"DFlash GGUF requires F32, F16, or BF16 weights: {tensor.name}"
            )
    return reader, meta, names


def build_plain_dflash_gguf_config(path):
    reader, meta, _ = _read_plain_dflash_gguf(path)

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
    fc = next(t for t in reader.tensors if t.name == "fc.weight")
    if tuple(fc.shape) != (hidden * len(targets), hidden):
        raise ValueError("DFlash GGUF fc shape does not match its target layers")
    return Qwen3Config(
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
        vocab_size=len(reader.fields["tokenizer.ggml.tokens"].contents()),
        use_sliding_window=any(pattern),
        sliding_window=int(field("attention.sliding_window")),
        layer_types=["sliding_attention" if p else "full_attention" for p in pattern],
        block_size=int(field("block_size")),
        dtype="bfloat16",
        dflash_config={
            "target_layer_ids": targets,
            "mask_token_id": int(meta["tokenizer.ggml.mask_token_id"]),
            "causal": bool(field("attention.causal")),
        },
    )


def plain_dflash_gguf_weights(path):
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType

    reader, _, names = _read_plain_dflash_gguf(path)
    for tensor in reader.tensors:
        data = np.array(tensor.data, copy=True)
        if tensor.tensor_type == GGMLQuantizationType.BF16:
            # GGUF stores BF16 as bytes; preserve the bits rather than cast them.
            weight = torch.from_numpy(data.view(np.uint16)).view(torch.bfloat16)
        else:
            weight = torch.from_numpy(data)
        yield names[tensor.name], weight.reshape(tuple(reversed(tensor.shape)))
