# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL's language model as a diffusion text encoder, on srt's model class.

The layers, kernels and weight layout are srt's ``Qwen3LLMModel``, so an
optimization there serves both the LLM server and the diffusion encoders.
This module only adapts the call: a padded ``[batch, seq]`` request runs as
one cache-free prefill (``prefill_only``), attention goes through the diffusion
attention backends, and the outputs keep the Hugging Face text model's shape.

srt fuses q/k/v and gate/up into one projection each. Checkpoint quantization
formats are per projection (their markers, packing and scales name ``q_proj``,
``gate_proj``, ...), so a quantized checkpoint runs each projection on its own
diffusion layer inside the fused srt layer (``_ProjectionQuantConfig``); an
unquantized one keeps srt's fused layers.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Iterable, Optional

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from sglang.multimodal_gen.runtime.distributed import tensor_model_parallel_all_gather
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
    WeightOnlyFP8ColumnParallelLinear,
    WeightOnlyFP8Linear,
    WeightOnlyFP8RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.common import add_prefix
from sglang.srt.layers.linear import LinearBase as SrtLinearBase
from sglang.srt.layers.linear import QKVParallelLinear as SrtQKVParallelLinear
from sglang.srt.layers.linear import RowParallelLinear as SrtRowParallelLinear
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig as SrtQuantizationConfig,
)
from sglang.srt.model_executor.prefill_only import eager_graphs, prefill_only
from sglang.srt.models.qwen3_vl import Qwen3LLMModel
from sglang.srt.runtime_context import get_parallel

# An encoder runs its whole layer stack on every rank of its TP group.
_SINGLE_PIPELINE_STAGE = SimpleNamespace(
    is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
)

# The TP "group" of a replicated encoder: one rank, so no collective runs.
_SINGLE_RANK = SimpleNamespace(world_size=1, rank_in_group=0)

# checkpoint projection -> (fused srt parameter, shard id)
_STACKED_PARAMS = (
    (".q_proj.", ".qkv_proj.", "q"),
    (".k_proj.", ".qkv_proj.", "k"),
    (".v_proj.", ".qkv_proj.", "v"),
    (".gate_proj.", ".gate_up_proj.", 0),
    (".up_proj.", ".gate_up_proj.", 1),
)
# fused srt projection -> the checkpoint projections it holds, in output order
_FUSED_PROJECTIONS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _tp_rank_size():
    parallel = get_parallel()
    return parallel.attn_tp_rank, parallel.attn_tp_size


class _ProjectionQuantConfig(SrtQuantizationConfig):
    """Quantize each checkpoint projection on its own diffusion layer.

    ``quant_config`` is a diffusion quantization config (serialized, online,
    Comfy, GGUF, Quanto, bitsandbytes, ...); ``weight_only_fp8`` selects the
    weight-only FP8 layers; ``replicate`` keeps every projection whole on each
    rank, for formats whose quant state cannot be sliced (bitsandbytes with
    nested states).
    """

    def __init__(self, quant_config=None, *, weight_only_fp8=False, replicate=False):
        super().__init__()
        self.quant_config = quant_config
        self.weight_only_fp8 = weight_only_fp8
        self.replicate = replicate

    def get_name(self) -> str:
        return "diffusion_projections"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError("built from a diffusion quantization config")

    def get_scaled_act_names(self) -> list[str]:
        return []

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, SrtLinearBase):
            return _ProjectionsMethod(self, prefix)
        return None

    def _column(self, input_size: int, output_size: int, prefix: str, shard: bool):
        if self.weight_only_fp8:
            if shard:
                return WeightOnlyFP8ColumnParallelLinear(
                    input_size,
                    output_size,
                    bias=False,
                    gather_output=False,
                    enable_fused_w8a8=False,
                )
            return WeightOnlyFP8Linear(
                input_size, output_size, bias=False, enable_fused_w8a8=False
            )
        layer_cls = ColumnParallelLinear if shard else ReplicatedLinear
        kwargs = dict(gather_output=False) if shard else {}
        return layer_cls(
            input_size,
            output_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=prefix,
            **kwargs,
        )

    def _row(self, input_size: int, output_size: int, prefix: str, shard: bool):
        if self.weight_only_fp8:
            if shard:
                return WeightOnlyFP8RowParallelLinear(
                    input_size,
                    output_size,
                    bias=False,
                    input_is_parallel=True,
                    reduce_results=False,
                    enable_fused_w8a8=False,
                )
            return WeightOnlyFP8Linear(
                input_size, output_size, bias=False, enable_fused_w8a8=False
            )
        if shard:
            # srt's layer boundary all-reduces the partial sums
            return RowParallelLinear(
                input_size,
                output_size,
                bias=False,
                input_is_parallel=True,
                reduce_results=False,
                quant_config=self.quant_config,
                prefix=prefix,
            )
        return ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=prefix,
        )

    def row_shards(self, prefix: str, input_size_per_partition: int) -> bool:
        if self.replicate:
            return False
        if self.weight_only_fp8 or self.quant_config is None:
            return True
        return self.quant_config.supports_input_partition(
            prefix, input_size_per_partition
        )


def _run(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


class _ProjectionsMethod(LinearMethodBase):
    """An srt projection computed by one diffusion layer per checkpoint
    projection. Column outputs keep srt's per-rank layout; a row projection
    returns this rank's partial sum for the srt layer boundary to reduce."""

    def __init__(self, config: _ProjectionQuantConfig, prefix: str):
        self.config = config
        self.prefix = prefix

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        parent, _, fused = self.prefix.rpartition(".")
        tp_rank, tp_size = _tp_rank_size()
        layer.projection_names = []
        layer.projection_slices = []
        if isinstance(layer, SrtRowParallelLinear):
            name = fused
            shard = tp_size == 1 or self.config.row_shards(
                add_prefix(name, parent), input_size_per_partition
            )
            layer.add_module(
                name,
                self.config._row(
                    input_size, output_size, add_prefix(name, parent), shard
                ),
            )
            layer.projection_names.append(name)
            # whole input on every rank: gather it, and let rank 0 alone carry
            # the product so that the boundary's all-reduce adds zeros elsewhere
            layer.projection_gathers_input = not shard and tp_size > 1
            return

        names = _FUSED_PROJECTIONS[fused]
        if isinstance(layer, SrtQKVParallelLinear):
            q_full = layer.total_num_heads * layer.head_size
            kv_full = layer.total_num_kv_heads * layer.head_size
            full_sizes = (q_full, kv_full, kv_full)
            replicas = getattr(layer, "num_kv_head_replicas", 1)
            owners = (tp_rank, tp_rank // replicas, tp_rank // replicas)
        else:
            full_sizes = tuple(layer.output_sizes)
            owners = (tp_rank,) * len(names)
        for name, full, part, owner in zip(
            names, full_sizes, output_partition_sizes, owners
        ):
            # A projection splits like srt's when its rows divide evenly over
            # the ranks; otherwise (replicated K/V heads, unsliceable quant
            # state) it runs whole and keeps this rank's rows.
            shard = not self.config.replicate and full == part * tp_size
            layer.add_module(
                name,
                self.config._column(input_size, full, add_prefix(name, parent), shard),
            )
            layer.projection_names.append(name)
            layer.projection_slices.append(
                None if shard or full == part else (owner * part, part)
            )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not layer.projection_slices:
            if layer.projection_gathers_input:
                x = tensor_model_parallel_all_gather(x.contiguous())
            out = _run(getattr(layer, layer.projection_names[0]), x)
            if layer.projection_gathers_input and _tp_rank_size()[0] != 0:
                out = torch.zeros_like(out)
        else:
            outs = []
            for name, rows in zip(layer.projection_names, layer.projection_slices):
                out = _run(getattr(layer, name), x)
                if rows is not None:
                    out = out.narrow(-1, *rows)
                outs.append(out)
            out = torch.cat(outs, dim=-1)
        return out if bias is None else out + bias

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # the diffusion layers inside are processed as modules of their own
        return


def causal_text_attention(config, attn_tp_size: int) -> LocalAttention:
    """The diffusion attention backends, for this rank's heads of every layer."""
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return LocalAttention(
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
        *,
        weight_only_fp8: bool = False,
        replicate_projections: bool = False,
    ):
        srt_config = copy.copy(config)
        srt_config.vision_config = SimpleNamespace(
            deepstack_visual_indexes=list(deepstack_visual_indexes)
        )
        projection_quant = None
        if quant_config is not None or weight_only_fp8 or replicate_projections:
            projection_quant = _ProjectionQuantConfig(
                quant_config,
                weight_only_fp8=weight_only_fp8,
                replicate=replicate_projections,
            )
        # Replicated projections replicate the whole stack: every rank runs all
        # heads, as one TP rank, and no layer exchanges anything.
        scope = (
            use_tensor_parallel_group(_SINGLE_RANK)
            if replicate_projections
            else nullcontext()
        )
        with (
            scope,
            eager_graphs(),
            get_parallel().override(pp_group=_SINGLE_PIPELINE_STAGE),
        ):
            super().__init__(
                config=srt_config, quant_config=projection_quant, prefix=prefix
            )
            embedding_prefix = add_prefix("embed_tokens", prefix)
            if quant_config is not None and quant_config.quantizes_embedding(
                embedding_prefix
            ):
                # a quantized table keeps the checkpoint's own format
                self.embed_tokens = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                    params_dtype=torch.get_default_dtype(),
                    quant_config=quant_config,
                    prefix=embedding_prefix,
                )
            self.attn = causal_text_attention(config, get_parallel().attn_tp_size)
        self.replicated = replicate_projections

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
            scope = (
                use_tensor_parallel_group(_SINGLE_RANK)
                if self.replicated
                else nullcontext()
            )
            with (
                scope,
                prefill_only(
                    [seq_len] * batch, positions, attention=self._attention
                ) as forward_batch,
            ):
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


class _ResidualReadout(nn.Module):
    """A final norm that returns the residual stream unnormalized."""

    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is not None:
            hidden_states = hidden_states + residual
        if post_residual_addition is not None:
            hidden_states = hidden_states + post_residual_addition
        return hidden_states if residual is None else (hidden_states, hidden_states)


def srt_text_param_name(params: dict, name: str) -> tuple[str, object]:
    """The parameter a language-model checkpoint tensor fills, and its shard id.

    A fused srt projection holds q/k/v or gate/up as shards of one parameter;
    a quantized one holds a diffusion layer per checkpoint projection.
    """
    for checkpoint_part, fused_part, shard_id in _STACKED_PARAMS:
        if checkpoint_part in name:
            fused = name.replace(checkpoint_part, fused_part)
            if fused in params:
                return fused, shard_id
            inner = name.replace(checkpoint_part, fused_part + checkpoint_part[1:])
            return inner, None
    if name in params:
        return name, None
    # a quantized row projection: `o_proj.weight` -> `o_proj.o_proj.weight`
    module, _, leaf = name.rpartition(".")
    projection = module.rpartition(".")[2]
    return f"{module}.{projection}.{leaf}", None


def load_srt_text_weight(
    params: dict, name: str, loaded_weight: torch.Tensor, *, keep_mapping: bool = False
) -> str:
    """Load a checkpoint tensor of the language model into srt's fused
    projections (or the diffusion layers inside a quantized one); return the
    name of the parameter it filled."""
    target, shard_id = srt_text_param_name(params, name)
    param = params[target]
    # The host-constrained single-rank H3 path keeps one dense parameter per
    # checkpoint projection so every streamed layer can retain its file views.
    if (
        keep_mapping
        and shard_id is None
        and param.device.type == loaded_weight.device.type == "cpu"
        and param.shape == loaded_weight.shape
        and param.dtype == loaded_weight.dtype
    ):
        param.data = loaded_weight
        return target
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    loaded_weight = loaded_weight.to(param.dtype)
    if shard_id is None:
        weight_loader(param, loaded_weight)
    else:
        weight_loader(param, loaded_weight, shard_id)
    return target
