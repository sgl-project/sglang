# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature-owned startup mappings for canonical model parameter names."""

from __future__ import annotations

import re

import torch

from sglang.srt.weight_sync.gpu_delta.bindings import (
    ConsumerSnapshot,
    DerivedImage,
    ParameterBindings,
    TensorBinding,
    _dtype_name,
    _full_slices,
    dense_target,
)


def _indexer_norm_binding(name, meta, target):
    """Match the CUDA indexer's ordinary BF16-checkpoint -> FP32 load.

    A numeric cast is not a byte permutation. These small parameters receive
    complete canonical target values, then use the normal loader's conversion
    while retaining the live FP32 parameter and graph address.
    """
    if meta["dtype"] != "BF16" or tuple(meta["shape"]) != tuple(target.shape):
        raise ValueError(f"unsupported canonical indexer norm: {name}")
    if target.dtype != torch.float32:
        raise ValueError(f"unsupported live indexer norm dtype: {name}")

    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        _full_slices(meta["shape"]),
        None,
        (target,),
        encoding="raw_bytes",
    )


class DeepSeekMlaMapping:
    """DeepSeek MLA/DSA family, including GLM's shared runtime implementation."""

    def __init__(self, model):
        self.model = model
        self.parameters = ParameterBindings(model, model.stacked_params_mapping)
        self.mla_layers = {}
        self.derived = []
        self.consumers = []

    def bind(self, name, meta):
        layer_id = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if layer_id and int(layer_id[1]) >= self.model.config.num_hidden_layers:
            self.parameters.excluded[name] = "static bundled draft"
            return None
        if name.endswith("rotary_emb.inv_freq"):
            self.parameters.excluded[name] = (
                "loader ignores computed rotary frequencies"
            )
            return None
        target_name = self.model.mutate_weight_preload(name)
        target = self.parameters.params.get(target_name)
        if target is None and ".experts." not in target_name:
            for param_stem, source_stem, shard in self.model.stacked_params_mapping:
                if (
                    param_stem != "fused_qkv_a_proj_with_mqa"
                    or source_stem not in target_name
                ):
                    continue
                candidate = target_name.replace(source_stem, param_stem)
                target = self.parameters.params.get(candidate)
                if target is None:
                    continue
                if meta["dtype"] != _dtype_name(target.dtype):
                    raise ValueError(
                        f"numerical fused mapping requires an adapter: {name}"
                    )
                sizes = [
                    self.model.config.q_lora_rank,
                    self.model.config.kv_lora_rank + self.model.config.qk_rope_head_dim,
                ]
                if not meta["shape"]:
                    raise ValueError(
                        "fused scalar projection metadata is not supported"
                    )
                target = target.narrow(0, sum(sizes[:shard]), sizes[shard])
                target_name = candidate
                break
        if target is None and ".indexer." in target_name:
            for source, at_end in (("wk", False), ("weights_proj", True)):
                marker = f".indexer.{source}.weight"
                if target_name.endswith(marker):
                    candidate = (
                        target_name[: -len(marker)] + ".indexer.wk_weights_proj.weight"
                    )
                    target = self.parameters.params.get(candidate)
                    if target is not None and meta["dtype"] == "BF16":
                        rows = meta["shape"][0]
                        target = target[-rows:] if at_end else target[:rows]
                        target_name = candidate
                    break
        if (
            target is not None
            and target.dtype == torch.float32
            and meta["dtype"] == "BF16"
            and re.fullmatch(
                r"model\.layers\.\d+\.self_attn\.indexer\.k_norm\.(weight|bias)",
                name,
            )
        ):
            module = self.parameters.modules[target_name.rsplit(".", 1)[0]]
            target, _ = dense_target(
                name, meta, module, target, _full_slices(meta["shape"])
            )
            return _indexer_norm_binding(name, meta, target)
        binding = self.parameters.bind(name, meta, target_name, target)
        if ".kv_b_proj.weight" in name:
            prefix = target_name.rsplit(".kv_b_proj.weight", 1)[0]
            self.mla_layers[prefix] = self.parameters.modules[prefix]
        return binding

    def finish(self):
        for prefix, attn in self.mla_layers.items():
            self._add_derived(prefix, attn)
            self.consumers.append(
                ConsumerSnapshot(lambda attn=attn: (attn.w_kc, attn.w_vc))
            )

    def _add_derived(self, prefix, attn):
        if attn.kv_b_proj.weight.dtype != torch.bfloat16:
            raise ValueError(
                "MLA delta cache refresh currently requires BF16 canonical weights"
            )

        key, value = attn.kv_b_proj.weight.unflatten(
            0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
        ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)

        self.derived.extend(
            (
                DerivedImage(f"{prefix}.w_kc", attn.w_kc, key),
                DerivedImage(
                    f"{prefix}.w_vc",
                    attn.w_vc,
                    value.transpose(1, 2),
                ),
            )
        )

    @staticmethod
    def aliases(a, b):
        for left, right in (("q_a_proj", "kv_a_proj_with_mqa"), ("wk", "weights_proj")):
            if a.replace(left, right) == b or b.replace(left, right) == a:
                return True
        return False


_MODEL_MAPPINGS = {
    name: DeepSeekMlaMapping
    for name in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
    )
}


def model_mapping(model):
    architecture = model.config.architectures[0]
    if architecture not in _MODEL_MAPPINGS:
        raise ValueError(f"unsupported GPU delta model mapping: {architecture}")
    return _MODEL_MAPPINGS[architecture](model)
