# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared-weight SenseNova Loop with independent attention state per visit."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.layernorm.mhc import hc_expand
from sglang.srt.layers.layer_boundary import (
    MHCState,
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.models import qwen3_5
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import add_prefix, make_pp_layers


class SenseNovaMHCRouter(nn.Module):
    """Canonical FP32 router with log-space Sinkhorn and BF16 residual math."""

    def __init__(self, *, hidden_size, settings):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        output_size = 2 * self.streams + (
            0 if self.fixed_identity else self.streams * self.streams
        )
        self.weight = nn.Parameter(
            torch.empty(output_size, self.streams * hidden_size, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.float32))

    def forward(self, hidden_states, out_norm_weight=None, out_norm_eps=None):
        residual = hidden_states.unflatten(-1, (self.streams, -1))
        flat = hidden_states.float()
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = F.linear(normalized, self.weight, self.bias)
        pre = logits[..., : self.streams].sigmoid()
        post = 2 * logits[..., self.streams : 2 * self.streams].sigmoid()
        if self.fixed_identity:
            mix = torch.eye(self.streams, device=flat.device, dtype=torch.float32)
            mix = mix.expand(*flat.shape[:-1], self.streams, self.streams)
        else:
            mix = logits[..., 2 * self.streams :].unflatten(
                -1, (self.streams, self.streams)
            )
            for _ in range(self.iterations):
                mix = mix - torch.logsumexp(mix, dim=-1, keepdim=True)
                mix = mix - torch.logsumexp(mix, dim=-2, keepdim=True)
            mix = mix.exp()
        branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(
            -2
        )
        if self.pre_scale != 1.0:
            branch = branch * self.pre_scale
        return branch, mix, post, False

    @staticmethod
    def merge(branch, residual, mix, post):
        streams = post.shape[-1]
        residual = residual.unflatten(-1, (streams, -1))
        mixed = torch.matmul(mix.to(residual.dtype), residual)
        update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        return (mixed + update).flatten(-2)


def _stage_facts(config, layer_id, *, residual=None):
    physical = config.execution_plan[layer_id]
    settings = config.mhc
    if settings is None or not settings["start"] <= physical < settings["end"]:
        return qwen3_5._qwen3_5_stage_facts(config, layer_id)
    residual = MHCState.facts() if residual is None else residual
    return (
        declare_attn(read=residual.attn_readout, update=residual.attn_update),
        declare_ffn(read=residual.ffn_readout, update=residual.ffn_update),
    )


class SenseNovaLoopModel(qwen3_5.Qwen3_5ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_parallel().pp_group
        if self.pp_group.world_size != 1:
            raise ValueError(
                "SenseNova Loop currently requires pipeline parallel size 1"
            )
        if quant_config is not None:
            raise ValueError(
                "SenseNova Loop currently supports unquantized weights only"
            )
        self.embed_tokens = self._build_embed_tokens(config)
        physical_layers = {}
        alt_stream = get_stream("alt")

        def get_layer(idx, prefix):
            physical = config.execution_plan[idx]
            linear = config.layers_block_type[idx] == "linear_attention"
            settings = config.mhc
            is_mhc = (
                settings is not None and settings["start"] <= physical < settings["end"]
            )
            if physical not in physical_layers:
                layer_class = (
                    qwen3_5.Qwen3_5LinearDecoderLayer
                    if linear
                    else qwen3_5.Qwen3_5AttentionDecoderLayer
                )
                layer = layer_class(
                    config=config,
                    layer_id=idx,
                    prefix=add_prefix("linear_attn" if linear else "self_attn", prefix),
                    alt_stream=alt_stream,
                    build_stages=False,
                )
                if is_mhc:
                    layer.mhc_attn = SenseNovaMHCRouter(
                        hidden_size=config.hidden_size, settings=settings
                    )
                    layer.mhc_ffn = SenseNovaMHCRouter(
                        hidden_size=config.hidden_size, settings=settings
                    )
                physical_layers[physical] = layer
            else:
                layer = self._new_visit(
                    physical_layers[physical], idx=idx, linear=linear
                )

            residual = None
            layer.starts_mhc = is_mhc and physical == settings["start"]
            if is_mhc:
                residual = MHCState(
                    hc_mult=config.mhc["streams"],
                    hc_attn_pre=layer.mhc_attn,
                    hc_ffn_pre=layer.mhc_ffn,
                    hc_post=SenseNovaMHCRouter.merge,
                    is_last_layer=physical == config.mhc["end"] - 1,
                ).residual_ops()
            attn, ffn = _stage_facts(config, idx, residual=residual)
            layer.attn_boundary, layer.ffn_boundary = append_stages(
                (attn, layer.input_layernorm),
                (ffn, layer.post_attention_layernorm),
            )
            return layer

        self.layers, self._start_layer, self._end_layer = make_pp_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
            stage_facts=lambda index: _stage_facts(config, index),
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.flashinfer_mnnvl_cutedsl_fusion = None
        self.layers_to_capture = []
        self._weight_layer_indices = {
            physical: config.execution_plan.index(physical)
            for physical in range(config.physical_num_hidden_layers)
        }

    @staticmethod
    def _new_visit(physical_layer, *, idx, linear):
        # Visits share all projections, norms and router parameters. Only cache
        # identities and the subsequently bound residual state belong to a visit.
        layer = copy.copy(physical_layer)
        layer._modules = layer._modules.copy()
        if linear:
            layer.linear_attn = copy.copy(layer.linear_attn)
            layer.linear_attn._modules = layer.linear_attn._modules.copy()
            layer.linear_attn.layer_id = idx
            layer.linear_attn.attn = copy.copy(layer.linear_attn.attn)
            layer.linear_attn.attn.layer_id = idx
        else:
            layer.attn = copy.copy(layer.attn)
            layer.attn.layer_id = idx
        layer.layer_id = idx
        return layer

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        residual_batch.start(forward_batch)
        for layer in self.layers:
            if layer.starts_mhc:
                hidden = residual_batch.fold(hidden, forward_batch)
                hidden = residual_batch.set_written(
                    hc_expand(hidden, layer.mhc_attn.streams), forward_batch
                )
            hidden = layer(
                positions=positions, hidden_states=hidden, forward_batch=forward_batch
            )
        return residual_batch.final_norm(
            hidden, forward_batch, self.norm, skip_empty=True
        )


class SenseNovaLoopForConditionalGeneration(qwen3_5.Qwen3_5ForConditionalGeneration):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=SenseNovaLoopModel,
        )
        if not self.language_model_only:
            self.visual.use_fp32_pos_embed_interpolation = True

    def set_dflash_layers_to_capture(self, layer_ids):
        raise NotImplementedError(
            "SenseNova Loop speculative decoding is not supported"
        )

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        raise NotImplementedError(
            "SenseNova Loop speculative decoding is not supported"
        )

    def load_weights(self, weights):
        def execution_weights():
            prefix = "model.language_model.layers."
            for name, weight in weights:
                if name == "model.language_model.loop_gate.logits":
                    continue
                if name.startswith(prefix):
                    physical, suffix = name[len(prefix) :].split(".", 1)
                    index = self.model._weight_layer_indices[int(physical)]
                    name = f"{prefix}{index}.{suffix}"
                yield name, weight

        return super().load_weights(execution_weights())


EntryClass = SenseNovaLoopForConditionalGeneration
