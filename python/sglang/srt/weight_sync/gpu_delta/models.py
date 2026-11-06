# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature-owned startup mappings for canonical model parameter names."""

from __future__ import annotations

import re
from dataclasses import replace

import torch

from sglang.srt.weight_sync.gpu_delta.bindings import (
    ConsumerSnapshot,
    DerivedImage,
    ParameterBindings,
    TensorBinding,
    _byte_view,
    _full_slices,
    dense_target,
)
from sglang.srt.weight_sync.gpu_delta.flashinfer import DestinationSegment


def _with_replicated_projection(binding, target, rows_per_head, key_rows=None):
    """Apply the same canonical mask to a DCP group's gathered MLA consumer.

    The decode payload remains complete. A logical head axis lets one affine
    descriptor select the K rows of every KV head, without per-head segments,
    temporary weights, or an all-gather while paused.
    """
    heads = target.shape[0] if key_rows is not None else target.shape[0] // rows_per_head
    span = heads * rows_per_head
    start = binding.slices[0][0] // span * span
    coverage = [[start, start + span], [0, binding.shape[1]]]
    if key_rows is None:
        source_shape = None
        slices = tuple(tuple(axis) for axis in coverage)
        expected = (span, binding.shape[1])
    else:
        source_shape = (
            binding.shape[0] // rows_per_head,
            rows_per_head,
            binding.shape[1],
        )
        slices = (
            (start // rows_per_head, start // rows_per_head + heads),
            (0, key_rows),
            (0, binding.shape[1]),
        )
        expected = (heads, key_rows, binding.shape[1])
    if (
        target.dtype != binding.torch_dtype
        or tuple(target.shape) != expected
        or start + span > binding.shape[0]
        or binding.shape[0] % rows_per_head
    ):
        raise ValueError("unsupported DCP replicated projection geometry")
    return replace(
        binding,
        slices=coverage,
        storage=(*binding.storage, target),
        destinations=(),
        segments=(
            DestinationSegment(
                tuple(tuple(axis) for axis in binding.slices), binding.destinations
            ),
            DestinationSegment(slices, (_byte_view(target),), source_shape),
        ),
    )


def _indexer_norm_binding(name, meta, target):
    """Match the CUDA indexer's ordinary BF16-checkpoint -> FP32 load.

    A numeric cast is not a byte permutation. These small parameters receive
    complete canonical target values, then use the normal loader's conversion
    while retaining the live FP32 parameter and graph address.
    """
    # The mapping selects this helper only for BF16 -> FP32 indexer norms.
    if tuple(meta["shape"]) != tuple(target.shape):
        raise ValueError(f"unsupported canonical indexer norm: {name}")

    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        _full_slices(meta["shape"]),
        (target,),
        encoding="raw_bytes",
    )


class DeepSeekMlaMapping:
    """DeepSeek MLA/DSA family, including GLM's shared runtime implementation."""

    # Canonical aliases used by DeepseekV2WeightLoaderMixin. The ordinary loader
    # owns its mappings locally; admission resolves only existing live targets.
    stacked_params = (
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("fused_qkv_a_proj_with_mqa", "q_a_proj", 0),
        ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
    )

    def __init__(self, model):
        self.model = model
        self.parameters = ParameterBindings(model, self.stacked_params)
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
        target_name = name
        if self.model.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
            target_name = name.replace(
                "mlp.shared_experts",
                f"mlp.experts.{self.model.config.n_routed_experts}",
            )
        target = self.parameters.params.get(target_name)
        if target is None and ".experts." not in target_name:
            for param_stem, source_stem, shard in self.stacked_params:
                if (
                    param_stem != "fused_qkv_a_proj_with_mqa"
                    or source_stem not in target_name
                ):
                    continue
                candidate = target_name.replace(source_stem, param_stem)
                target = self.parameters.params.get(candidate)
                if target is None:
                    continue
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
        if name.endswith((".q_b_proj.weight", ".q_proj.weight")):
            prefix = target_name.rsplit(".", 2)[0]
            attn = self.parameters.modules[prefix]
            if attn.q_b_proj_qrep is not None:
                raise ValueError("GPU delta requires unquantized DCP Q replication")
            if attn.q_b_proj_qrep_weight is not None:
                binding = _with_replicated_projection(
                    binding, attn.q_b_proj_qrep_weight, attn.qk_head_dim
                )
        if ".kv_b_proj.weight" in name:
            prefix = target_name.rsplit(".kv_b_proj.weight", 1)[0]
            attn = self.parameters.modules[prefix]
            self.mla_layers[prefix] = attn
            if attn.w_kc_qrep is not None:
                binding = _with_replicated_projection(
                    binding,
                    attn.w_kc_qrep,
                    attn.qk_nope_head_dim + attn.v_head_dim,
                    attn.qk_nope_head_dim,
                )
        return binding

    def finish(self):
        for prefix, attn in self.mla_layers.items():
            self._add_derived(prefix, attn)
            self.consumers.append(
                ConsumerSnapshot(
                    lambda attn=attn: (
                        attn.w_kc,
                        attn.w_vc,
                        attn.w_kc_qrep,
                        attn.q_b_proj_qrep_weight,
                        attn.q_b_proj_qrep,
                    )
                )
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
