# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL's language model as a diffusion text encoder, on srt's model class.

The layers, kernels and weight layout are srt's ``Qwen3LLMModel``, so an
optimization there serves both the LLM server and the diffusion encoders.
This module only adapts the call: a padded ``[batch, seq]`` request runs as
one cache-free prefill (``prefill_only``), attention goes through the diffusion
attention backends, and the outputs keep the Hugging Face text model's shape.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Iterable, Optional

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.model_executor.prefill_only import prefill_only
from sglang.srt.models.qwen3_vl import Qwen3LLMModel
from sglang.srt.runtime_context import get_parallel

# An encoder runs its whole layer stack on every rank of its TP group.
_SINGLE_PIPELINE_STAGE = SimpleNamespace(
    is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
)

# checkpoint projection -> (fused srt parameter, shard id)
_STACKED_PARAMS = (
    (".q_proj.", ".qkv_proj.", "q"),
    (".k_proj.", ".qkv_proj.", "k"),
    (".v_proj.", ".qkv_proj.", "v"),
    (".gate_proj.", ".gate_up_proj.", 0),
    (".up_proj.", ".gate_up_proj.", 1),
)


class _ResidualReadout(nn.Module):
    """A final norm that returns the residual stream unnormalized."""

    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is not None:
            hidden_states = hidden_states + residual
        if post_residual_addition is not None:
            hidden_states = hidden_states + post_residual_addition
        return hidden_states if residual is None else (hidden_states, hidden_states)


class Qwen3VLTextModel(Qwen3LLMModel):
    """srt's Qwen3-VL language model, called like ``transformers``' text model."""

    # used only as `self.embed_tokens(input_ids)`; no tied output head here
    host_resident_table_names = ["embed_tokens"]

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
        deepstack_visual_indexes: Iterable[int] = (),
    ):
        srt_config = copy.copy(config)
        srt_config.vision_config = SimpleNamespace(
            deepstack_visual_indexes=list(deepstack_visual_indexes)
        )
        with get_parallel().override(pp_group=_SINGLE_PIPELINE_STAGE):
            super().__init__(
                config=srt_config, quant_config=quant_config, prefix=prefix
            )
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        attn_tp_size = get_parallel().attn_tp_size
        self.attn = LocalAttention(
            num_heads=config.num_attention_heads // attn_tp_size,
            head_size=head_dim,
            num_kv_heads=max(1, config.num_key_value_heads // attn_tp_size),
            softmax_scale=head_dim**-0.5,
            causal=True,
            supported_attention_backends=(
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    def drop_final_norm(self) -> None:
        """Return the last layer's residual stream instead of normalizing it."""
        self.norm = _ResidualReadout()

    def _attention(self, q, k, v, seq_lens, scale):
        # A padded batch: every sequence has the same length.
        batch = len(seq_lens)
        out = self.attn(
            q.view(batch, -1, *q.shape[1:]),
            k.view(batch, -1, *k.shape[1:]),
            v.view(batch, -1, *v.shape[1:]),
        )
        return out.reshape(q.shape[0], q.shape[1], -1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **_,
    ):
        """Causal over each padded row, like the reference: ``attention_mask``
        is not applied to the scores."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len, hidden = inputs_embeds.shape
        device = inputs_embeds.device

        # the hard-coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).expand(3, batch, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        if position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        positions = position_ids.reshape(3, -1).to(device)

        deepstack = None
        if deepstack_visual_embeds:
            mask = visual_pos_masks.reshape(-1).to(device)
            deepstack = inputs_embeds.new_zeros(
                batch * seq_len, hidden * len(deepstack_visual_embeds)
            )
            for i, embeds in enumerate(deepstack_visual_embeds):
                deepstack[mask, i * hidden : (i + 1) * hidden] = embeds.to(
                    device, inputs_embeds.dtype
                )

        # A layer's output is what the next layer's input norm reads; the last
        # layer's is the stream entering the final norm.
        captured_last = []
        hook = None
        if output_hidden_states:
            self.layers_to_capture = list(range(1, len(self.layers)))
            hook = self.norm.register_forward_hook(
                lambda _m, args, out: captured_last.append(
                    out[1].clone() if isinstance(out, tuple) else args[0].clone()
                )
            )
        try:
            with prefill_only(
                [seq_len] * batch, positions, attention=self._attention
            ) as forward_batch:
                out = super().forward(
                    None,
                    positions,
                    forward_batch,
                    input_embeds=inputs_embeds.reshape(batch * seq_len, hidden),
                    input_deepstack_embeds=deepstack,
                )
        finally:
            self.layers_to_capture = []
            if hook is not None:
                hook.remove()

        captured = []
        if isinstance(out, tuple):
            out, captured = out
        last_hidden_state = out.view(batch, seq_len, hidden)
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple(
                h.view(batch, seq_len, hidden) for h in [*captured, *captured_last]
            )
        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )

    def load_weight(self, name: str, loaded_weight: torch.Tensor) -> str:
        """Load one checkpoint tensor, named relative to this module."""
        return load_srt_text_weight(dict(self.named_parameters()), name, loaded_weight)


def load_srt_text_weight(params: dict, name: str, loaded_weight: torch.Tensor) -> str:
    """Load a checkpoint tensor of the language model into srt's fused
    projections; return the name of the parameter it filled."""
    for checkpoint_part, fused_part, shard_id in _STACKED_PARAMS:
        if checkpoint_part in name:
            name = name.replace(checkpoint_part, fused_part)
            param = params[name]
            param.weight_loader(param, loaded_weight.to(param.dtype), shard_id)
            return name
    param = params[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight.to(param.dtype))
    return name
