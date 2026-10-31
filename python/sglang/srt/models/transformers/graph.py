# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from weakref import ref

import torch
from torch import nn


def _resolve_layer_stack(owner):
    count = owner.text_config.num_hidden_layers
    candidates = []
    for module in owner.model.modules():
        if (
            not isinstance(module, (nn.ModuleList, nn.ModuleDict))
            or len(module) != count
        ):
            continue
        layers = (
            list(module.values()) if isinstance(module, nn.ModuleDict) else list(module)
        )
        if all(
            any(
                getattr(child, "_sglang_attention_key", None) == str(index)
                for child in layers[index].modules()
            )
            for index in range(owner.start_layer, owner.end_layer)
        ):
            candidates.append(layers)
    if len(candidates) != 1:
        raise ValueError(
            "Cannot identify a unique language layer stack for prefill graphs"
        )
    return candidates[0]


class TransformersGraphBody(nn.Module):
    torch_compile_dynamic_arg_dims = {
        "input_ids": 0,
        "positions": -1,
        "input_embeds": 0,
        "pp_proxy_tensors": 0,
    }

    def __init__(self, owner):
        super().__init__()
        self._owner = ref(owner)
        object.__setattr__(self, "_layers", _resolve_layer_stack(owner))
        self.start_layer = owner.start_layer
        self.end_layer = owner.end_layer

    @property
    def layers(self):
        return self._layers

    @property
    def attention_instances(self):
        return self._owner().attention_instances

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        owner = self._owner()
        if owner is None:
            raise RuntimeError("The Transformers graph owner is no longer alive")
        if not owner.pp_group.is_first_rank:
            if pp_proxy_tensors is not None:
                input_embeds = pp_proxy_tensors["hidden_states"]
            if input_embeds is None:
                raise ValueError(
                    "Pipeline graph capture requires incoming hidden states"
                )
            input_ids = None
        capture = getattr(owner, "_aux_capture", None)
        if capture is not None:
            capture.reset()
        hidden = owner._run_hf_backbone_eager(
            input_ids=input_ids,
            input_embeds=input_embeds,
            positions=positions,
            forward_batch=forward_batch,
        )
        return (hidden, capture.collect()) if capture is not None else hidden


def _clear_missing_token_types(buffer, forward_batch, context):
    if getattr(forward_batch, "token_type_ids", None) is None:
        buffer.zero_()


class TransformersGraphMixin:
    @property
    def is_mrope_enabled(self):
        resolver = getattr(self, "_uses_mrope_positions", None)
        return bool(resolver is not None and resolver())

    def get_prefill_graph_model(self):
        if self.config is not self.text_config and not getattr(
            self, "_mm_cache_enabled", False
        ):
            raise ValueError(
                "Multimodal prefill graphs require cached feature composition"
            )
        adapter = getattr(self, "_prefill_graph_adapter", None)
        if adapter is None:
            adapter = TransformersGraphBody(self)
            self._prefill_graph_adapter = adapter
        return adapter

    def register_prefill_graph_inputs(self, registry):
        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            GraphSlot,
            PaddingPolicy,
        )

        if (
            not registry.has_slot("token_type_ids")
            and "token_type_ids" in inspect.signature(self.model.forward).parameters
        ):
            registry.register_slot(
                GraphSlot(
                    "token_type_ids",
                    lambda _bs, tokens: (tokens,),
                    torch.int64,
                    axis="tokens",
                    padding_policy=PaddingPolicy.ZERO,
                    post_fill=_clear_missing_token_types,
                )
            )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _run_hf_backbone(
        self, input_ids, input_embeds, positions, forward_batch, **kwargs
    ):
        adapter = getattr(self, "_prefill_graph_adapter", None)
        if adapter is None or kwargs:
            return self._run_hf_backbone_eager(
                input_ids=input_ids,
                input_embeds=input_embeds,
                positions=positions,
                forward_batch=forward_batch,
                **kwargs,
            )
        from .execution_context import transformers_execution_context

        with transformers_execution_context(forward_batch):
            output = adapter.forward(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        if isinstance(output, tuple):
            hidden, auxiliary = output
            self._aux_capture.values.update(zip(self._aux_capture.layer_ids, auxiliary))
            return hidden
        return output
