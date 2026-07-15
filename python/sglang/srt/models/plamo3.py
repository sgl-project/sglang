# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
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
# ======================================================================
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn

from sglang.srt.configs.plamo3 import Plamo3Config
from sglang.srt.distributed import (
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers




PLAMO3_POST_MIXER_NORM_OFFSET = 1.0 / 5
PLAMO3_POST_MLP_NORM_OFFSET = 1.0 / (5**1.5)


def get_attention_sliding_window_size(config: "Plamo3Config") -> int:
    # SGLang attention uses an exclusive window size.
    return max(config.window_size - 1, 0)


class Plamo3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, offset: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(in_dtype)
        return (self.offset + self.weight) * x


class Plamo3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
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
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Plamo3Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Plamo3Config,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.head_dim = config.head_dim
        self.rotary_dim = self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"
        self.rope_theta = (
            config.rope_local_theta if self.is_sliding else config.rope_theta
        )
        rope_scaling = None if self.is_sliding else getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            # PLaMo3 config returns a nested dict keyed by attention type.
            key = "sliding_attention" if self.is_sliding else "full_attention"
            rope_scaling = dict(rope_scaling.get(key, {}))
            rope_scaling.pop("rope_theta", None)
            if rope_scaling.get("rope_type") == "default":
                rope_scaling = None
        self.rope_scaling = rope_scaling
        self.sliding_window = (
            get_attention_sliding_window_size(config) if self.is_sliding else None
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.q_norm = Plamo3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Plamo3RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch=forward_batch)

        output, _ = self.o_proj(attn_output)
        return output


class Plamo3DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Plamo3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Plamo3Attention(
            layer_id=layer_id,
            config=config,  # type: ignore[arg-type]
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Plamo3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.pre_mixer_norm = Plamo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mixer_norm = Plamo3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            offset=PLAMO3_POST_MIXER_NORM_OFFSET,
        )
        self.pre_mlp_norm = Plamo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = Plamo3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            offset=PLAMO3_POST_MLP_NORM_OFFSET,
        )
        self.is_sliding = self.self_attn.is_sliding
        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # Pipeline-parallel stages pass the pre-normalization hidden_states as
        # the residual. PLaMo3 layers keep the residual locally, so we only need
        # to return the current output as the residual for the next stage.
        residual = hidden_states

        hidden_states = self.pre_mixer_norm(hidden_states)
        attn_hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            **kwargs,
        )
        attn_hidden_states = self.post_mixer_norm(attn_hidden_states)
        hidden_states = residual + attn_hidden_states

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, hidden_states

class Plamo3TextModel(nn.Module):
    def __init__(
        self,
        config: Plamo3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Plamo3DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = Plamo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.gradient_checkpointing = False
        self.layers_to_capture: List[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids)
                if input_embeds is None
                else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states: List[torch.Tensor] = []

        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            layer_outputs = self.layers[i](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                **kwargs,
            )
            hidden_states, residual = layer_outputs

        if self.end_layer in self.layers_to_capture:
            aux_hidden_states.append(hidden_states)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Plamo3ForCausalLM(nn.Module):
    config_class = Plamo3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_up_proj"],
    }
    supported_lora_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    embedding_modules: dict[str, Any] = {}
    embedding_padding_modules: list[str] = []
    supports_lora = True

    def __init__(
        self,
        config: Plamo3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Plamo3TextModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if self.config.tie_word_embeddings and (
            self.model.pp_group.is_first_rank
            and self.model.pp_group.is_last_rank
        ):
            # Only tie weights when both embeddings and lm_head exist.
            # In pipeline parallelism, one or both are PPMissingLayer.
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self) -> int:
        return get_attention_sliding_window_size(self.config)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs: Any,
    ) -> Union[LogitsProcessor, PPProxyTensors]:
        model_output = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
            **kwargs,
        )

        if isinstance(model_output, PPProxyTensors):
            return model_output

        hidden_states = model_output
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],
        input_embeds: torch.Tensor | None = None,
    ):
        start, end = split_interval
        if start == 0:
            hidden_states = (
                self.model.embed_tokens(input_ids)
                if input_embeds is None
                else input_embeds
            )
            forward_batch.hidden_states = hidden_states
            forward_batch.model_specific_states = {
                "positions": positions,
            }

        for i in range(start, end):
            layer = self.model.layers[i]
            layer_output = layer(
                positions=forward_batch.model_specific_states["positions"],
                hidden_states=forward_batch.hidden_states,
                forward_batch=forward_batch,
            )
            forward_batch.hidden_states = layer_output[0]

        if end == self.model.config.num_hidden_layers:
            forward_batch.hidden_states = self.model.norm(forward_batch.hidden_states)
            return self.logits_processor(
                input_ids,
                forward_batch.hidden_states,
                self.lm_head,
                forward_batch,
            )
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def _layer_idx_from_name(name: str) -> Optional[int]:
            parts = name.split(".")
            if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
                try:
                    return int(parts[2])
                except ValueError:
                    pass
            return None

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace("model.layers.layers.", "model.layers.")

            # Skip layer weights that do not belong to this PP rank.
            layer_idx = _layer_idx_from_name(name)
            if layer_idx is not None and (
                layer_idx < self.start_layer or layer_idx >= self.end_layer
            ):
                continue

            # In pipeline parallelism, embed_tokens is a PPMissingLayer on
            # non-first ranks. When tied, map its checkpoint weight to lm_head
            # on the last PP rank.
            if name == "model.embed_tokens.weight":
                if not self.model.pp_group.is_first_rank:
                    if (
                        self.config.tie_word_embeddings
                        and self.model.pp_group.is_last_rank
                    ):
                        name = "lm_head.weight"
                    else:
                        continue

            name = name.replace(".mixer.", ".self_attn.")
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                continue
            if remapped_name != name:
                param = params_dict[remapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(remapped_name)
                continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                parts = name.split(".")
                if shard_name not in parts:
                    continue
                mapped_name = ".".join(
                    param_name if part == shard_name else part for part in parts
                )
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                if name == "lm_head.weight" and name not in params_dict:
                    continue
                # Skip rotary embedding buffers stored in the checkpoint
                # (cos_cached, sin_cached, inv_freq) — these are recomputed
                # at runtime by get_rope.
                if "rotary_emb" in name:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def set_eagle3_layers_to_capture(
        self, layer_ids: Optional[List[int]] = None
    ) -> None:
        # Plamo3 checkpoints do not implement EAGLE3 aux hidden capture.
        # Keep the entry point explicit so speculative paths fail fast with a
        # clear message instead of falling into a CUDA graph capture error.
        raise NotImplementedError(
            "Plamo3 does not support EAGLE3 auxiliary hidden state capture."
        )

    def get_embed(self):
        return self.model.embed_tokens.weight

    def get_embed_and_head(self):
        # This is used by speculative decoding to share the target model's
        # embeddings and LM head with the draft model.
        return self.model.embed_tokens.weight, self.lm_head.weight


EntryClass = Plamo3ForCausalLM
