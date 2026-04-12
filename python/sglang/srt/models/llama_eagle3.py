"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix, is_hip

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

import copy
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP


logger = logging.getLogger(__name__)
_is_hip = is_hip()


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        if not hasattr(self.config, "parallel_drafting"):
            self.config.parallel_drafting = False
        if not hasattr(self.config, "mask_token_id") or self.config.mask_token_id is None:
            self.config.mask_token_id = (
                self.config.pad_token_id if self.config.pad_token_id is not None else 0
            )

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_scaling = rope_parameters
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
        self.is_mrope_enabled = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )
        self.parallel_drafting = bool(self.config.parallel_drafting)
        self.mask_token_id = int(self.config.mask_token_id)
        self.mask_hidden = nn.Parameter(torch.zeros(1, 1, self.fc.in_features))

        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def prepare_p_eagle_inputs(
        self,
        last_token_ids: torch.Tensor,
        fused_hidden_states: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        if last_token_ids.dim() == 1:
            last_token_ids = last_token_ids.unsqueeze(-1)
        if last_token_ids.dim() != 2 or last_token_ids.shape[1] != 1:
            raise ValueError("last_token_ids must have shape [batch, 1]")

        if fused_hidden_states.dim() != 3 or fused_hidden_states.shape[1] != 1:
            raise ValueError("fused_hidden_states must have shape [batch, 1, hidden*3]")

        if fused_hidden_states.shape[-1] != self.fc.in_features:
            raise ValueError(
                f"Expected fused hidden size {self.fc.in_features}, got {fused_hidden_states.shape[-1]}"
            )

        batch = last_token_ids.shape[0]
        device = last_token_ids.device
        hidden_dtype = fused_hidden_states.dtype
        if k == 1:
            all_hidden_states = fused_hidden_states
            input_ids = last_token_ids
        else:
            mask_hidden = self.mask_hidden.to(device=device, dtype=hidden_dtype).expand(
                batch, k - 1, -1
            )
            all_hidden_states = torch.cat([fused_hidden_states, mask_hidden], dim=1)
            mask_token_ids = torch.full(
                (batch, k - 1),
                self.mask_token_id,
                dtype=last_token_ids.dtype,
                device=device,
            )
            input_ids = torch.cat([last_token_ids, mask_token_ids], dim=1)

        # torch.nn.Embedding works fine on ROCm; CPU row-by-row loop was a
        # workaround for a stale env and caused catastrophic throughput (~0.5 t/s).
        embeds = self.embed_tokens(input_ids)
        projected_hidden_states = self.fc(all_hidden_states.to(self.fc.weight.dtype))
        return embeds, projected_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                assert embeds is not None
                embeds = torch.cat(
                    [
                        embeds[:-1],
                        self.embed_tokens(input_ids[-1].unsqueeze(0)),
                    ]
                )
            if embeds is None:
                embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states
        # Apply fc only when hidden_states are in TARGET aux-hidden space (fc.in_features = 7680
        # for a 3-layer EAGLE3 setup).  During DECODE DRAFT STEPS, hidden_states is the draft
        # model's own previous output (fc.out_features = 2560) and must NOT be projected again —
        # zero-padding 2560 to 7680 and projecting through fc produces garbage logits → 0% accept.
        if hidden_states.shape[-1] != self.fc.out_features:
            # RESCALE: the EAGLE3 head was trained on full-precision (fp16/bf16) aux hidden
            # states.  Q1_0_G128 dequantised hidden states have ~25× larger norms,
            # causing fc output to explode → 0% acceptance.  Scale back to training dist.
            # TODO: remove once EAGLE3 head is retrained on quantised hidden states.
            hidden_states = hidden_states * (1.0 / 25.0)
            hidden_states = torch.clamp(hidden_states, -100.0, 100.0)

            expected_in = self.fc.in_features
            current_in = hidden_states.shape[-1]
            if current_in != expected_in:
                if current_in < expected_in:
                    hidden_states = F.pad(hidden_states, (0, expected_in - current_in))
                else:
                    hidden_states = hidden_states[..., :expected_in]
            if hidden_states.dtype != self.fc.weight.dtype:
                hidden_states = hidden_states.to(self.fc.weight.dtype)
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMEagle3]
