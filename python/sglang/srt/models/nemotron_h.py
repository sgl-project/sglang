# Copyright 2023-2025 SGLang Team
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_h.py

"""Inference-only NemotronH model."""

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from sglang.srt.configs import NemotronHConfig
from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MLP
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import ReLU2
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    replace_prefix,
    replace_substrings,
)
from sglang.srt.utils import add_prefix, make_layers_non_pp
from sglang.utils import logger


class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size

        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ReLU2()

    def forward(self, x: torch.Tensor):
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class NemotronHMLPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.mixer = NemotronHMLP(
            config,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=f"{prefix}.mixer",
            layer_idx=layer_idx,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer.forward(hidden_states)
        return hidden_states, residual


class NemotronHMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_idx
        self.mixer = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.mamba_n_groups,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.mamba_hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        attn_backend = forward_batch.attn_backend
        assert isinstance(attn_backend, HybridLinearAttnBackend)
        assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)
        attn_backend.linear_attn_backend.forward(
            mixer=self.mixer,
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            output=output,
            use_triton_causal_conv=True,  # TODO: investigate need of `use_triton_causal_conv`
        )
        return output, residual


class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn.forward(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronHAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.mixer = NemotronHAttention(
            config,
            layer_idx,
            quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer.forward(
            hidden_states=hidden_states, forward_batch=forward_batch
        )
        return hidden_states, residual


Layers = (
    NemotronHAttentionDecoderLayer
    | NemotronHMLPDecoderLayer
    | NemotronHMambaDecoderLayer
)
ALL_DECODER_LAYER_TYPES: dict[str, type[Layers]] = {
    ATTENTION: NemotronHAttentionDecoderLayer,
    MLP: NemotronHMLPDecoderLayer,
    MAMBA: NemotronHMambaDecoderLayer,
}


class NemotronHModel(nn.Module):
    def __init__(
        self,
        *,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        lora_config = None
        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.hybrid_override_pattern[idx]]
            return layer_class(config, idx, quant_config=quant_config, prefix=prefix)

        self.layers = make_layers_non_pp(
            len(config.hybrid_override_pattern), get_layer, prefix=f"{prefix}.layers"
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        residual = None
        for layer in self.layers:
            if not isinstance(layer, Layers):
                raise ValueError(f"Unknown layer type: {type(layer)}")
            hidden_states, residual = layer.forward(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not get_pp_group().is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class NemotronHForCausalLM(nn.Module):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    remap_prefix = {"backbone": "model"}
    remap_substr = {"A_log": "A", "embeddings": "embed_tokens"}

    def __init__(
        self,
        *,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        lora_config = None
        self.config = config
        self.model = self._init_model(
            config=config, quant_config=quant_config, prefix=prefix
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config
                    else lora_config.lora_vocab_padding_size
                ),
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def _init_model(
        self,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return NemotronHModel(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model.forward(
            input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        updated_weights = []
        for name, loaded_weight in weights:
            name = replace_prefix(name, self.remap_prefix)
            name = replace_substrings(name, self.remap_substr)
            updated_weights.append((name, loaded_weight))
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in updated_weights:
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = [NemotronHForCausalLM]
