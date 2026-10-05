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
"""Inference-only MiniCPM model compatible with HuggingFace weights."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.lookahead import (
    SpardaPrefetchContext,
    clear_sparda_selection_cache,
    get_forecast_state,
    get_sparda_generation,
    get_sparda_prefetcher,
    get_sparda_request_context,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, set_weight_attrs
from sglang.srt.utils.hf_transformers_utils import get_rope_config


class MiniCPMMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
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
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        attn_use_rope: bool = True,
        use_output_gate: bool = False,
        attention_bias: bool = False,
        sparda_enabled: bool = False,
        num_layers: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attn_use_rope = attn_use_rope
        self.use_output_gate = use_output_gate
        self.sparda_enabled = sparda_enabled
        self.num_layers = num_layers

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        if self.sparda_enabled:
            # Phase one keeps Forecast weights replicated.  This preserves the
            # GQA mapping when num_kv_heads < TP size; the projection is small
            # compared with the backbone and avoids changing QKV sharding.
            self.q_future_proj = ReplicatedLinear(
                hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                params_dtype=torch.get_default_dtype(),
                prefix=add_prefix("q_future_proj", prefix),
            )
            # The first sparse layer has no previous-layer forecast.  SparDA
            # trains a separate selector for that layer; later layers use the
            # forecast published by their predecessor.
            self.q_curr_proj = (
                ReplicatedLinear(
                    hidden_size,
                    self.total_num_kv_heads * self.head_dim,
                    bias=False,
                    params_dtype=torch.get_default_dtype(),
                    prefix=add_prefix("q_curr_proj", prefix),
                )
                if layer_id == 0
                else None
            )
        else:
            self.q_future_proj = None
            self.q_curr_proj = None
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if self.attn_use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        if self.use_output_gate:
            self.o_gate = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=attention_bias,
                quant_config=quant_config,
                prefix=add_prefix("o_gate", prefix),
            )

    def _project_indexer_query(
        self,
        projection: Optional[nn.Module],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if projection is None:
            return None

        query, _ = projection(hidden_states)

        # SparDA's indexer query is in the same positional space as the KV
        # cache. The official MiniCPM implementation applies RoPE to q_future
        # and q_curr independently of the regular grouped-query projections.
        if self.attn_use_rope:
            orig_dtype = query.dtype
            query_fp32 = query.float()
            query, _ = self.rotary_emb(
                positions,
                query_fp32,
                query_fp32.clone(),
            )
            query = query.to(orig_dtype)

        query = query.view(-1, self.total_num_kv_heads, self.head_dim)

        # Match QKVParallelLinear's GQA partitioning.  When there are fewer KV
        # heads than TP ranks, each rank owns a replica of one logical head.
        tp_size = self.qkv_proj.tp_size
        tp_rank = self.qkv_proj.tp_rank
        if self.total_num_kv_heads >= tp_size:
            start = tp_rank * self.num_kv_heads
        else:
            start = tp_rank // self.qkv_proj.num_kv_head_replicas
        return query[:, start : start + self.num_kv_heads, :].contiguous()

    def _project_forecast(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self._project_indexer_query(self.q_future_proj, positions, hidden_states)

    def _project_current_selector(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self._project_indexer_query(self.q_curr_proj, positions, hidden_states)

    def _submit_forecast_prefetch(
        self,
        next_forecast: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        target_layer: Optional[int] = None,
    ) -> None:
        """Submit this layer's one-step forecast to the optional cache adapter."""
        if next_forecast is None:
            return
        if target_layer is None:
            target_layer = self.attn.layer_id + 1
        if self.num_layers is not None and target_layer >= self.num_layers:
            return
        prefetcher = get_sparda_prefetcher(forward_batch)
        if prefetcher is None or not forward_batch.rids:
            return

        if not forward_batch.forward_mode.is_decode_or_idle():
            # The current HiCache page resolver stages decode history.  Keep
            # chunked prefill on the Forecast selection path, but do not
            # submit a ticket that cannot describe its in-flight KV pages.
            return

        query_spans = [(i, i + 1) for i in range(len(forward_batch.rids))]

        request_contexts = get_sparda_request_context(forward_batch)
        for request_index, request_id in enumerate(forward_batch.rids):
            start, end = query_spans[request_index]
            if start == end:
                continue
            context = SpardaPrefetchContext(
                request=forward_batch,
                forward_batch=forward_batch,
                request_index=request_index,
                selector_backend=get_attn_backend(),
                forecast_batch=next_forecast,
            )
            if request_contexts is not None and request_index < len(request_contexts):
                context = SpardaPrefetchContext(
                    request=request_contexts[request_index],
                    forward_batch=forward_batch,
                    request_index=request_index,
                    selector_backend=get_attn_backend(),
                    forecast_batch=next_forecast,
                )
            prefetcher.prefetch_forecast_query(
                request_id,
                get_sparda_generation(forward_batch, request_index),
                target_layer,
                next_forecast[start:end],
                context=context,
            )

    def _restore_sparda_requests(self, forward_batch: ForwardBatch) -> None:
        """Restore host-backed rows before falling back to regular attention."""
        prefetcher = get_sparda_prefetcher(forward_batch)
        restore_request = getattr(prefetcher, "restore_request", None)
        request_contexts = get_sparda_request_context(forward_batch)
        if restore_request is None or request_contexts is None:
            return
        for request in request_contexts:
            if not restore_request(request):
                raise RuntimeError(
                    "SparDA request restoration failed; refusing to run "
                    "attention on host-backed KV"
                )

    def _wait_for_forecast_prefetch(self, forward_batch: ForwardBatch) -> bool:
        """Make the next-layer attention stream observe completed H2D copies."""
        prefetcher = get_sparda_prefetcher(forward_batch)
        if prefetcher is None or not forward_batch.rids:
            return True
        if not forward_batch.forward_mode.is_decode_or_idle():
            return True
        wait_for_layer = getattr(prefetcher, "wait_for_layer", None)
        if wait_for_layer is None:
            return True
        ready = True
        for request_index, request_id in enumerate(forward_batch.rids):
            try:
                request_ready = wait_for_layer(
                    request_id,
                    get_sparda_generation(forward_batch, request_index),
                    self.attn.layer_id,
                )
            except Exception:
                request_ready = False
            ready = request_ready and ready
        if not ready:
            cancel_for_layer = getattr(prefetcher, "cancel_for_layer", None)
            if cancel_for_layer is not None:
                for request_index, request_id in enumerate(forward_batch.rids):
                    cancel_for_layer(
                        request_id,
                        get_sparda_generation(forward_batch, request_index),
                        self.attn.layer_id,
                    )
        return ready

    def _consume_forecast_prefetch(self, forward_batch: ForwardBatch) -> None:
        """Release the page lease after this layer has consumed its KV pages."""
        prefetcher = get_sparda_prefetcher(forward_batch)
        if prefetcher is None or not forward_batch.rids:
            return
        consume_for_layer = getattr(prefetcher, "consume_for_layer", None)
        if consume_for_layer is None:
            return
        for request_index, request_id in enumerate(forward_batch.rids):
            consume_for_layer(
                request_id,
                get_sparda_generation(forward_batch, request_index),
                self.attn.layer_id,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.attn_use_rope:
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            q, k = self.rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        forecast_state = None
        forecast_for_attention = None
        current_selector = None
        next_forecast = None
        if self.sparda_enabled:
            forecast_state = get_forecast_state(forward_batch)
            forecast_for_attention = forecast_state.for_layer(self.attn.layer_id)
            current_selector = self._project_current_selector(positions, hidden_states)
            next_forecast = self._project_forecast(positions, hidden_states)
            if self.attn.layer_id == 0:
                self._submit_forecast_prefetch(
                    current_selector,
                    forward_batch,
                    target_layer=self.attn.layer_id,
                )
            self._submit_forecast_prefetch(next_forecast, forward_batch)
            if not self._wait_for_forecast_prefetch(forward_batch):
                # A cancelled or failed ticket must never make the attention
                # kernel consume a page whose H2D event was not observed.
                # Revert to the current-query/full loading path instead.
                self._restore_sparda_requests(forward_batch)
                forecast_for_attention = None
                current_selector = None

        selector_query = forecast_for_attention
        if selector_query is None and self.attn.layer_id == 0:
            selector_query = current_selector

        if selector_query is None:
            # Preserve the existing backend path for the first layer (and for
            # all requests when SparDA is disabled).  The extra kwarg would
            # otherwise force the tc-piecewise path through the eager adapter.
            attn_output = self.attn(q, k, v, forward_batch)
        else:
            attn_output = self.attn(
                q,
                k,
                v,
                forward_batch,
                forecast_query=selector_query,
            )

        if forecast_state is not None:
            forecast_state.publish(self.attn.layer_id, next_forecast)
            self._consume_forecast_prefetch(forward_batch)

        if self.use_output_gate:
            o_gate_output, _ = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate_output)

        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMLightningMixer(nn.Module):
    """Lightning attention mixer backed by the shared linear-attention backend.

    This is a wrapper that prepares inputs for the backend and handles
    the QKV projection, normalization, RoPE, and output processing,
    while delegating the recurrent computation through RadixAttention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_rope: bool = True,
        use_output_gate: bool = False,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_output_norm: bool = False,
        qk_norm: bool = True,
        scale: str | float = "1/sqrt(d)",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        if scale == "1/sqrt(d)":
            scaling = self.head_dim ** (-0.5)
        elif scale == "1/d":
            scaling = self.head_dim ** (-1.0)
        elif isinstance(scale, (int, float)):
            scaling = float(scale)
        else:
            raise ValueError(f"Unsupported lightning scale: {scale}")
        self.use_output_gate = use_output_gate
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.use_output_norm = use_output_norm

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if self.use_output_norm:
            self.o_norm = RMSNorm(self.num_heads * self.head_dim, eps=self.rms_norm_eps)
            set_weight_attrs(
                self.o_norm.weight, {"weight_loader": sharded_weight_loader(0)}
            )

        if self.use_output_gate:
            self.z_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=self.attention_bias,
                quant_config=quant_config,
                prefix=add_prefix("z_proj", prefix),
            )

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        if self.use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states.new_empty(hidden_states.shape[0], self.hidden_size)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, self.head_dim))
            k = self.k_norm(k.reshape(-1, self.head_dim))

        if self.use_rope:
            q = q.reshape(-1, self.num_heads * self.head_dim)
            k = k.reshape(-1, self.num_kv_heads * self.head_dim)
            orig_dtype = q.dtype
            q, k = q.float(), k.float()
            q, k = self.rotary_emb(positions, q, k)
            q, k = q.to(orig_dtype), k.to(orig_dtype)

        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        o = self.attn(q, k, v, forward_batch)

        if self.use_output_norm:
            o = self.o_norm(o)

        if self.use_output_gate:
            z, _ = self.z_proj(hidden_states)
            o = o * F.sigmoid(z)

        y, _ = self.o_proj(o)
        return y


class MiniCPMDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        if isinstance(config, MiniCPMHybridConfig):
            self.mixer_type = config.mixer_types[layer_id]
            attn_use_rope = config.attn_use_rope
            attn_use_output_gate = config.attn_use_output_gate
            attention_bias = config.attention_bias
        else:
            self.mixer_type = "minicpm4"
            attn_use_rope = True
            attn_use_output_gate = False
            attention_bias = False

        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if self.mixer_type == "minicpm4":
            self.self_attn = MiniCPMAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=getattr(config, "head_dim", None),
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                attn_use_rope=attn_use_rope,
                use_output_gate=attn_use_output_gate,
                attention_bias=attention_bias,
                sparda_enabled=getattr(config, "sparda_enabled", False),
                num_layers=config.num_hidden_layers,
                prefix=add_prefix("self_attn", prefix),
            )
        elif self.mixer_type == "lightning-attn":
            self.self_attn = MiniCPMLightningMixer(
                hidden_size=self.hidden_size,
                num_heads=config.lightning_nh,
                num_kv_heads=config.lightning_nkv,
                head_dim=config.lightning_head_dim,
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                use_rope=config.lightning_use_rope,
                use_output_gate=config.use_output_gate,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                use_output_norm=config.use_output_norm,
                qk_norm=config.qk_norm,
                scale=config.lightning_scale,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            raise ValueError(f"Unsupported mixer type: {self.mixer_type}")
        self.mlp = MiniCPMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        return hidden_states, None


class MiniCPMModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if getattr(self.config, "sparda_enabled", False):
            clear_sparda_selection_cache(forward_batch)
            get_forecast_state(forward_batch).reset()

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states = self.norm(hidden_states)
        if getattr(self.config, "sparda_enabled", False):
            prefetcher = get_sparda_prefetcher(forward_batch)
            offload_request = getattr(prefetcher, "offload_request_history", None)
            request_contexts = get_sparda_request_context(forward_batch)
            sparse_config = getattr(self.config, "sparse_config", None) or {}
            if (
                offload_request is not None
                and request_contexts is not None
                and forward_batch.forward_mode.is_decode_or_idle()
            ):
                for request in request_contexts:
                    offload_request(
                        request,
                        keep_device_tokens=int(sparse_config.get("window_size", 0)),
                        min_history_len=int(sparse_config.get("dense_len", 0)),
                    )
        return hidden_states


class MiniCPMSALAForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.num_experts = getattr(self.config, "num_experts", 0)
        self.quant_config = quant_config
        self.model = MiniCPMModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base
        self._sparda_indexer_loaded = False

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
        else:
            lm_head = self.lm_head
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            )
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        self._load_sparda_indexer_weights()

    def _load_sparda_indexer_weights(self) -> None:
        """Load Forecast projections from the standalone SparDA checkpoint."""
        if not getattr(self.config, "sparda_enabled", False):
            return
        if self._sparda_indexer_loaded:
            return

        indexer_path = getattr(self.config, "sparda_indexer_path", None)
        if not indexer_path:
            raise ValueError(
                "SparDA is enabled but config.sparda_indexer_path is missing."
            )
        path = Path(indexer_path)
        if not path.is_file():
            raise FileNotFoundError(f"SparDA indexer checkpoint does not exist: {path}")

        numpy_reconstruct = np.core.multiarray._reconstruct
        safe_numpy_globals = [
            numpy_reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
        ]
        with torch.serialization.safe_globals(safe_numpy_globals):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                f"SparDA indexer checkpoint must contain a state dict, got "
                f"{type(checkpoint).__name__}."
            )
        state_dict = checkpoint
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                state_dict = nested
                break

        params = {
            name: param
            for name, param in self.named_parameters()
            if name.endswith(("q_future_proj.weight", "q_curr_proj.weight"))
        }
        if not params:
            raise RuntimeError(
                "SparDA is enabled but the MiniCPM model has no q_future_proj "
                "parameters."
            )

        normalized_state = {}
        for name, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized_state[name.removeprefix("module.")] = value

        expected_aliases = set()
        for name in params:
            expected_aliases.update(
                (name, name.removeprefix("model."), f"model.{name}")
            )
        unexpected = sorted(
            name
            for name in normalized_state
            if name.endswith(("q_future_proj.weight", "q_curr_proj.weight"))
            and name not in expected_aliases
        )
        if unexpected:
            raise ValueError(
                "SparDA indexer checkpoint has unexpected Forecast weights: "
                + ", ".join(unexpected[:4])
                + (" ..." if len(unexpected) > 4 else "")
            )

        missing = []
        for name, param in params.items():
            candidates = (
                name,
                name.removeprefix("model."),
                f"model.{name}",
            )
            loaded_weight = next(
                (
                    normalized_state[candidate]
                    for candidate in candidates
                    if candidate in normalized_state
                ),
                None,
            )
            if loaded_weight is None:
                missing.append(name)
                continue
            if tuple(param.shape) != tuple(loaded_weight.shape):
                raise ValueError(
                    f"SparDA indexer shape mismatch for {name}: model expects "
                    f"{tuple(param.shape)}, checkpoint provides "
                    f"{tuple(loaded_weight.shape)}."
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        if missing:
            raise ValueError(
                "SparDA indexer checkpoint is missing Forecast weights: "
                + ", ".join(missing[:4])
                + (" ..." if len(missing) > 4 else "")
            )
        self._sparda_indexer_loaded = True


class MiniCPMForCausalLM(MiniCPMSALAForCausalLM):
    """Alias for MiniCPM checkpoints whose config uses the HF architecture name."""


EntryClass = [MiniCPMSALAForCausalLM, MiniCPMForCausalLM]
