# Copyright 2023-2024 SGLang Team
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
"""Inference-only HRM-Text (Hierarchical Reasoning Model -- Text) model.

Reference HuggingFace implementation:
    transformers/models/hrm_text/modeling_hrm_text.py  (transformers >= 5.9.0)

The model runs a hierarchical recurrent forward over two transformer stacks
(``H`` slow, ``L`` fast) inside nested loops. Each recurrence step gets its own
KV cache slot via a unique ``RadixAttention(layer_id=...)``; the global layer
index for a given ``(step, layer_in_stack)`` is
``step * num_layers_per_stack + layer_in_stack`` -- the same ``cycle_offset``
formula used by the HF reference. The total KV slot count is
``num_layers_per_stack * H_cycles * (L_cycles + 1)`` (==
``config.num_hidden_layers`` after the HF config ``__post_init__`` inflation),
which ``ModelConfig`` exposes as ``num_attention_layers``.

PrefixLM attention (prompt bidirectional during prefill, completion causal at
decode) is expressed with ``attn_type=AttentionType.DECODER_BIDIRECTIONAL``.
That mode is only honored by the Triton attention backend and only when
``--disable-cuda-graph`` and ``--chunked-prefill-size=-1`` are set;
``ModelRunner.model_specific_adjustment`` forces those settings (plus
``--disable-radix-cache``) for this model.

The on-disk attention weight ``attn.gqkv_proj.weight`` is a single tensor with
rows concatenated as ``[gate | q | k | v]`` along dim 0; ``mlp.gate_up_proj``
is ``[gate | up]``. Both are loaded directly by ``MergedColumnParallelLinear``'s
fused-on-disk auto-split path (``loaded_shard_id=None``).
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _num_layers_per_stack(config: PretrainedConfig) -> int:
    """Layers in one (H or L) stack.

    With a native transformers >= 5.9.0 config, ``__post_init__`` rewrites
    ``num_hidden_layers`` to ``num_layers_per_stack * H_cycles * (L_cycles + 1)``
    and stores the per-stack count in ``num_layers_per_stack``. Fall back to
    deriving it if the attribute is absent.
    """
    nlps = getattr(config, "num_layers_per_stack", None)
    if nlps is not None:
        return int(nlps)
    return config.num_hidden_layers // (config.H_cycles * (config.L_cycles + 1))


def _steps_used(config: PretrainedConfig, stack_kind: str) -> list[int]:
    """Recurrence steps at which a given stack runs.

    L runs at ``{h*(L+1)+l : 0 <= h < H, 0 <= l < L}``; H runs at the trailing
    ``{h*(L+1)+L : 0 <= h < H}``. The two sets are disjoint, so each
    ``(step, layer_in_stack)`` maps to a unique global KV layer index.
    """
    H_cycles = config.H_cycles
    L_cycles = config.L_cycles
    if stack_kind == "L":
        return [
            h * (L_cycles + 1) + low_idx
            for h in range(H_cycles)
            for low_idx in range(L_cycles)
        ]
    return [h * (L_cycles + 1) + L_cycles for h in range(H_cycles)]


class HrmTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"HrmTextMLP only supports hidden_act='silu', got {hidden_act!r}"
            )
        # Fused [gate | up] matching the on-disk `mlp.gate_up_proj.weight`.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HrmTextAttention(nn.Module):
    """One self-attention block; projection weights shared across recurrence
    steps. The per-step KV cache slots are realized by the (weightless)
    ``RadixAttention`` instances in ``self.attn``, keyed by recurrence step.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx_in_stack: int,
        stack_kind: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0, (
            f"num_attention_heads={self.total_num_heads} must be divisible "
            f"by tp_size={tp_size}"
        )
        # HF main hardcodes MHA (num_key_value_groups=1). We follow.
        self.total_num_kv_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # gqkv_proj: 4-way fused [gate | q | k | v] matching the on-disk
        # `attn.gqkv_proj.weight` row layout. The weight_loader auto-splits the
        # fused disk tensor along the output dim by `output_sizes` (the
        # `loaded_shard_id is None` path). MHA only: GQA would need
        # QKVParallelLinear semantics for q/k/v shard replication.
        per_head_size = self.total_num_heads * self.head_dim
        self.gqkv_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [per_head_size] * 4,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gqkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # HF 5.9.0 stores rope params in `config.rope_parameters`
        # (e.g. {"rope_theta": 10000.0, "rope_type": "default"}); older/flat
        # configs may expose `rope_theta` directly. Support both.
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", 10000.0)
        rope_type = rope_parameters.get("rope_type", "default")
        # "default" rope = no scaling; pass the dict through otherwise.
        rope_scaling = None if rope_type in ("default", None) else rope_parameters
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )

        # One RadixAttention per recurrence step this stack runs at. Each gets a
        # unique `layer_id` (== global KV slot) so the recurrent forward writes
        # to disjoint cache slots. These modules hold no parameters.
        num_layers_per_stack = _num_layers_per_stack(config)
        self.attn = nn.ModuleDict()
        for step in _steps_used(config, stack_kind):
            global_idx = step * num_layers_per_stack + layer_idx_in_stack
            self.attn[str(step)] = RadixAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                scaling=self.scaling,
                num_kv_heads=self.num_kv_heads,
                layer_id=global_idx,
                attn_type=AttentionType.DECODER_BIDIRECTIONAL,
                quant_config=quant_config,
                prefix=add_prefix(f"attn.{step}", prefix),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        current_step: int,
    ) -> torch.Tensor:
        gqkv, _ = self.gqkv_proj(hidden_states)
        g, q, k, v = gqkv.split(
            [self.q_size, self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn[str(current_step)](q, k, v, forward_batch)
        # Sigmoid gate (HrmText / Qwen3Next style).
        attn_out = torch.sigmoid(g) * attn_out
        out, _ = self.o_proj(attn_out)
        return out


class HrmTextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx_in_stack: int,
        stack_kind: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = HrmTextAttention(
            config=config,
            layer_idx_in_stack=layer_idx_in_stack,
            stack_kind=stack_kind,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = HrmTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        # Parameterless RMSNorm (HF HrmTextRMSNorm has no weight).
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        current_step: int,
    ) -> torch.Tensor:
        # Pre-norm residual; norm a copy and add residual outside (matches HF).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            current_step=current_step,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HrmTextStack(nn.Module):
    """A single transformer stack -- instantiated twice (H and L)."""

    def __init__(
        self,
        config: PretrainedConfig,
        stack_kind: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        num_layers_per_stack = _num_layers_per_stack(config)
        self.layers = nn.ModuleList(
            [
                HrmTextDecoderLayer(
                    config=config,
                    layer_idx_in_stack=i,
                    stack_kind=stack_kind,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(num_layers_per_stack)
            ]
        )
        self.final_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        current_step_base: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                current_step=current_step_base,
            )
        return self.final_norm(hidden_states)


class HrmTextModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.L_module = HrmTextStack(
            config=config,
            stack_kind="L",
            quant_config=quant_config,
            prefix=add_prefix("L_module", prefix),
        )
        self.H_module = HrmTextStack(
            config=config,
            stack_kind="H",
            quant_config=quant_config,
            prefix=add_prefix("H_module", prefix),
        )
        # Frozen learned initial low-cycle state (disk key `model.z_L_init`).
        self.z_L_init = nn.Parameter(
            torch.zeros(config.hidden_size), requires_grad=False
        )

        # HF uses config.embedding_scale (= 1 / initializer_range), NOT
        # sqrt(hidden_size).
        self.embedding_scale = getattr(config, "embedding_scale", None)
        if self.embedding_scale is None:
            init_range = getattr(config, "initializer_range", 0.02)
            self.embedding_scale = 1.0 / init_range

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states_high_cycle = input_embeds
        else:
            hidden_states_high_cycle = self.embed_tokens(input_ids)
        hidden_states_high_cycle = hidden_states_high_cycle * self.embedding_scale

        hidden_states_low_cycle = (
            self.z_L_init.to(
                dtype=hidden_states_high_cycle.dtype,
                device=hidden_states_high_cycle.device,
            )
            .expand_as(hidden_states_high_cycle)
            .contiguous()
        )

        H_cycles = self.config.H_cycles
        L_cycles = self.config.L_cycles
        for high_cycle_idx in range(H_cycles):
            for low_cycle_idx in range(L_cycles):
                step = high_cycle_idx * (L_cycles + 1) + low_cycle_idx
                hidden_states_low_cycle = self.L_module(
                    positions=positions,
                    hidden_states=hidden_states_low_cycle + hidden_states_high_cycle,
                    forward_batch=forward_batch,
                    current_step_base=step,
                )
            step = high_cycle_idx * (L_cycles + 1) + L_cycles
            hidden_states_high_cycle = self.H_module(
                positions=positions,
                hidden_states=hidden_states_high_cycle + hidden_states_low_cycle,
                forward_batch=forward_batch,
                current_step_base=step,
            )

        return hidden_states_high_cycle


class HrmTextForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = HrmTextModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # On-disk attention weights use `.attn.`; our modules use `.self_attn.`
        # (the per-step RadixAttention modules under `.self_attn.attn.{step}`
        # hold no parameters, so they never receive weights). Disk tensors are
        # already fused -- gqkv_proj / gate_up_proj load via the fused-on-disk
        # auto-split path, so no stacked_params_mapping is needed.
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if ".attn." in name:
                name = name.replace(".attn.", ".self_attn.", 1)
            if name not in params_dict:
                # e.g. parameterless final_norm has no disk weight; skip strays.
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = HrmTextForCausalLM
