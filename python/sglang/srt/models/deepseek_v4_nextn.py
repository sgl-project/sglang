"""Inference-only DeepSeek V4 NextN Speculative Decoding."""

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    dp_gather_partial,
    get_attention_dp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class DeepseekV4ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_eps = config.rms_norm_eps

        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("e_proj", prefix),
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("h_proj", prefix),
        )

        layer_name = "decoder"

        # Multi stream is disabled on MTP layer
        self.decoder = DeepseekV4DecoderLayer(
            config,
            layer_id=0,
            quant_config=quant_config,
            is_nextn=True,
            prefix=add_prefix(layer_name, prefix),
            alt_streams=None,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Coped from DeepSeekV4Model
    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            if (
                envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
                and envs.SGLANG_DSV4_MODE.get() == "2604"
            ):
                n_tokens = hidden_states.shape[0]
                d = self.config.hidden_size
                # spec_info.hidden_states: [n, hc*d] → reshape to [n*hc, d] for 2D kernels
                hc_flat = forward_batch.spec_info.hidden_states.view(
                    n_tokens * self.hc_mult, d
                )
                # hnorm + h_proj on each hc copy independently: [n*hc, d] → [n*hc, d]
                h_proj_out, _ = self.h_proj(self.hnorm(hc_flat))
                # reshape back: [n*hc, d] → [n, hc, d]
                h_proj_hidden_states = h_proj_out.view(n_tokens, self.hc_mult, d)

                # embed: [n, d] → enorm → e_proj → [n, d]
                e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))
                # broadcast [n, 1, d] + [n, hc, d] → [n, hc, d]
                hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states
            else:
                e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))
                h_proj_hidden_states, _ = self.h_proj(
                    self.hnorm(forward_batch.spec_info.hidden_states)
                )
                hidden_states = e_proj_hidden_states + h_proj_hidden_states
                hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:  # Pure TP attention
            input_ids_global = input_ids

        hidden_states = self.decoder(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )

        # decoder output: [n, hc, d] → flatten to [n, hc*d] for spec pipeline
        pre_hc_head = (
            hidden_states.flatten(1)
            if envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
            else None
        )

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.shared_head.norm(hidden_states)

        if pre_hc_head is not None:
            return hidden_states, pre_hc_head
        return hidden_states


class DeepseekV4ForCausalLMNextN(DeepseekV4ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.determine_num_fused_shared_experts()

        self.model = DeepseekV4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        result = self.model(input_ids, positions, forward_batch)
        pre_hc_head = None
        if (
            envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
        ):
            hidden_states, pre_hc_head = result
        else:
            hidden_states = result
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=pre_hc_head,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = [DeepseekV4ForCausalLMNextN]
