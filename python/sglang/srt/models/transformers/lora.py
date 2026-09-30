# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from functools import lru_cache

import torch

_LAYER_PATH = re.compile(r"(?:^|\.)(?:layers|layer|h|block)\.(\d+)\.(.+)")


class TensorOutputLoRA:
    _hf_returns_tensor = True

    def apply_lora(self, base_output, *args, **kwargs):
        scale = (
            getattr(self.base_layer, "embed_scale", None)
            if self._hf_lora_embedding
            else None
        )
        if scale is not None:
            delta = super().apply_lora(torch.zeros_like(base_output), *args, **kwargs)
            if isinstance(scale, torch.Tensor):
                scale = scale.to(delta.dtype)
            return base_output + delta * scale
        return super().apply_lora(base_output, *args, **kwargs)

    def forward(self, inputs, *args, **kwargs):
        shape = inputs.shape
        if self._hf_lora_embedding:
            if inputs.ndim == 2:
                if shape[0] != 1:
                    raise ValueError("Transformers LoRA expects one packed token batch")
                inputs = inputs.reshape(-1)
            elif inputs.ndim != 1:
                raise ValueError("Embedding LoRA expects packed token IDs")
        elif inputs.ndim == 3:
            if shape[0] != 1:
                raise ValueError("Transformers LoRA expects one packed token batch")
            inputs = inputs.reshape(-1, shape[-1])
        elif inputs.ndim != 2:
            raise ValueError("Linear LoRA expects a token matrix or packed 3D tensor")
        result = super().forward(inputs, *args, **kwargs)
        if isinstance(result, tuple):
            result, bias = result
            if bias is not None:
                result = result + bias
        if self._hf_lora_embedding:
            return result.reshape(*shape, result.shape[-1])
        return result.reshape(*shape[:-1], result.shape[-1])


@lru_cache(maxsize=None)
def _tensor_lora_class(base_class):
    return type(
        f"Transformers{base_class.__name__}", (TensorOutputLoRA, base_class), {}
    )


def adapt_transformers_lora(module, *, embedding=False):
    if not isinstance(module, TensorOutputLoRA):
        module.__class__ = _tensor_lora_class(type(module))
    module._hf_lora_embedding = embedding
    return module


def _layer_path(name):
    match = _LAYER_PATH.search(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _base_layer(module):
    return getattr(module, "base_layer", module)


def _output_parts(module):
    if hasattr(module, "total_num_heads"):
        return (
            module.total_num_heads * module.head_size,
            module.total_num_kv_heads * module.head_size,
            module.total_num_kv_heads * module.v_head_size,
        )
    return tuple(getattr(module, "output_sizes", [module.output_size]))


class TransformersLoRAMixin:
    def get_lora_layer_id(self, name):
        result = _layer_path(name)
        return result[0] if result is not None else None

    def _get_lora_modules(self):
        result = {}
        for name, wrapped in self.named_modules():
            if ".base_layer" in name:
                continue
            module = _base_layer(wrapped)
            if not getattr(module, "_hf_returns_tensor", False) or not hasattr(
                module, "input_size"
            ):
                continue
            key = _layer_path(name)
            if key is None:
                continue
            if key in result:
                raise ValueError(f"Ambiguous Transformers LoRA module: {key}")
            result[key] = module
        return result

    def _get_lora_module(self, target, layer_index):
        modules = self._get_lora_modules()
        module = modules.get((layer_index, target))
        if module is None:
            module = next(
                (module for (_, name), module in modules.items() if name == target),
                None,
            )
        if module is None:
            raise ValueError(
                f"No Transformers LoRA module {target!r} at layer {layer_index}"
            )
        return module

    def _get_lora_aliases(self):
        aliases = {}
        for source, (destination, shard) in self._stacked_mapping.items():
            key, target = _layer_path(source), _layer_path(destination)
            if key is not None and target is not None:
                aliases[key] = (target, {"q": 0, "k": 1, "v": 2}.get(shard, shard))
        return aliases

    def normalize_lora_weight_name(self, name):
        return re.sub(
            r"\.(?:word_embeddings|embed_in|wte)\.(lora_[AB])\.",
            r".embed_tokens.\1.",
            name,
        )

    def get_lora_target_modules(self, targets):
        modules = self._get_lora_modules()
        if isinstance(targets, str):
            if targets not in {"all", "all-linear"}:
                raise ValueError("LoRA target modules must be suffixes or all-linear")
            result = {name for _, name in modules}
            if targets == "all":
                if self.model.get_input_embeddings() is not None:
                    result.add("embed_tokens")
                if getattr(self, "lm_head", None) is not None:
                    result.add("lm_head")
            return result
        aliases = self._get_lora_aliases()
        names = {name for _, name in modules}
        result = set()
        for requested in targets:
            if requested in {
                "embed_tokens",
                "word_embeddings",
                "embed_in",
                "wte",
                "lm_head",
            }:
                requested = "lm_head" if requested == "lm_head" else "embed_tokens"
                self.get_hidden_dim(requested, 0)
                result.add(requested)
                continue
            matched = {
                name
                for name in names
                if name == requested or name.endswith(f".{requested}")
            }
            matched.update(
                target[1]
                for (_, source), (target, _) in aliases.items()
                if source == requested or source.endswith(f".{requested}")
            )
            if not matched:
                raise ValueError(
                    f"LoRA target {requested!r} does not match a lowered Transformers layer"
                )
            result.update(matched)
        return result

    def get_hidden_dim(self, module_name, layer_idx):
        if module_name in {"embed_tokens", "lm_head"}:
            embedding = (
                self.model.get_input_embeddings()
                if module_name == "embed_tokens"
                else self.lm_head
            )
            embedding = _base_layer(embedding)
            if embedding is None:
                raise ValueError(f"This Transformers model has no {module_name}")
            shape = (embedding.org_vocab_size, embedding.embedding_dim)
            return shape if module_name == "embed_tokens" else shape[::-1]
        module = self._get_lora_module(module_name, layer_idx)
        return module.input_size, sum(_output_parts(module))

    def get_stacked_multiply(self, module_name):
        leaf = module_name.rsplit(".", 1)[-1]
        return len(self.packed_modules_mapping.get(leaf, (leaf,)))

    def get_lora_buffer_shape(self, kind, target, layer_index, max_rank, slots):
        module = self._get_lora_module(target, layer_index)
        if kind == "A":
            width = getattr(module, "input_size_per_partition", module.input_size)
            return slots, max_rank * self.get_stacked_multiply(target), width
        width = sum(getattr(module, "output_partition_sizes", (module.output_size,)))
        return slots, width, max_rank

    def normalize_lora_weights(self, weights, layer_index):
        aliases = self._get_lora_aliases()
        modules = self._get_lora_modules()
        groups = {}
        for name, tensor in weights.items():
            parsed = _layer_path(name)
            if parsed is None:
                raise ValueError(f"Cannot locate the layer for LoRA tensor {name!r}")
            index, suffix = parsed
            match = re.fullmatch(r"(.+)\.(lora_[AB])(?:\.[^.]+)?\.weight", suffix)
            if index != layer_index or match is None or tensor.ndim != 2:
                raise ValueError(f"Unsupported LoRA tensor {name!r}")
            source, kind = match.groups()
            key = (index, source)
            target, shard = aliases.get(key, (key, None))
            if target not in modules:
                raise ValueError(
                    f"LoRA tensor {name!r} has no matching Transformers module"
                )
            tensors = groups.setdefault(target, {})
            slot = (shard, kind)
            if slot in tensors:
                raise ValueError(f"Duplicate LoRA tensor for {target}: {slot}")
            tensors[slot] = tensor
        normalized = {}
        for (index, target), tensors in groups.items():
            module = modules[index, target]
            parts = _output_parts(module)
            count = self.get_stacked_multiply(target)
            if ((None, "lora_A") in tensors) != ((None, "lora_B") in tensors):
                raise ValueError(f"Both LoRA factors are required for {target}")
            ranks = {
                tensor.shape[0] if kind == "lora_A" else tensor.shape[1]
                for (_, kind), tensor in tensors.items()
            }
            if len(ranks) != 1:
                raise ValueError(f"Inconsistent LoRA ranks for {target}")
            rank = ranks.pop()
            sample = next(iter(tensors.values()))
            for kind in ("lora_A", "lora_B"):
                if (None, kind) in tensors:
                    if any(shard is not None for shard, _ in tensors):
                        raise ValueError(
                            f"Cannot mix fused and split LoRA tensors for {target}"
                        )
                    tensor = tensors[None, kind]
                    shape = (
                        (rank, module.input_size)
                        if kind == "lora_A"
                        else (sum(parts), rank)
                    )
                    if tensor.shape != shape:
                        raise ValueError(
                            f"Invalid {kind} shape for {target}: {tuple(tensor.shape)}, expected {shape}"
                        )
                    if kind == "lora_A":
                        tensor = tensor.repeat(count, 1)
                else:
                    chunks = []
                    for shard in range(count):
                        a, b = (
                            tensors.get((shard, "lora_A")),
                            tensors.get((shard, "lora_B")),
                        )
                        if (a is None) != (b is None):
                            raise ValueError(
                                f"Both LoRA factors are required for {target} slice {shard}"
                            )
                        shape = (
                            (rank, module.input_size)
                            if kind == "lora_A"
                            else (parts[shard], rank)
                        )
                        tensor = tensors.get((shard, kind))
                        if tensor is None:
                            tensor = sample.new_zeros(shape)
                        if tensor.shape != shape:
                            raise ValueError(
                                f"Invalid {kind} shape for {target} slice {shard}"
                            )
                        chunks.append(tensor)
                    tensor = torch.cat(chunks, dim=0)
                normalized[f"model.layers.{index}.{target}.{kind}.weight"] = tensor
        weights.clear()
        weights.update(normalized)
