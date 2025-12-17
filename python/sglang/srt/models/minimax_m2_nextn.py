import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.minimax_m2 import MiniMaxM2DecoderLayer, MiniMaxM2ForCausalLM

logger = logging.getLogger(__name__)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


class MiniMaxM2MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.e_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.h_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_projection = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        # Minimax M2 has 1 MTP layers
        self.mtp_block = MiniMaxM2DecoderLayer(
            config=config, quant_config=quant_config, prefix=prefix, layer_id=layer_id
        )

    def forward(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:

        # logger.warning(f"(gaoji:m2_nextn:forward: input_embeds: {input_embeds.shape}\n"
        #       f"forward_batch.spec_info.hidden_states: {forward_batch.spec_info.hidden_states.shape}\n")
        hidden_states = self.linear_projection(
            torch.cat(
                (
                    self.h_norm(forward_batch.spec_info.hidden_states),
                    self.e_norm(input_embeds),
                ),
                dim=-1,
            )
        )

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )
        hidden_states = residual + hidden_states

        return hidden_states


class MiniMaxM2ForCausalLMNextN(MiniMaxM2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", None)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )  # mtp layers share the same embed_tokens
        self.mtp_layers = nn.ModuleList(
            [
                MiniMaxM2MultiTokenPredictorLayer(
                    config=config, quant_config=quant_config, prefix=prefix, layer_id=i
                )
                for i in range(self.num_mtp_layers)
            ]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_idx: int = 0,
    ) -> torch.Tensor:

        input_embeds = self.embed_tokens(input_ids)
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            logger.warning(
                f"(gaoji:m2_nextn:forward: layer_idx: {layer_idx}\n"
                f"input_ids: {input_ids.shape}\n"
                f"input_embeds: {input_embeds.shape}\n"
            )
        input_embeds[positions == 0] = 0
        hidden_states = self.mtp_layers[layer_idx](
            positions, forward_batch, input_embeds
        )  # TODO(zhongyu): support multiple MTP layers
        hidden_states = self.final_layernorm(hidden_states)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            name = self.map_model_name_to_mtp_param_name(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Only merge q/k/v or gate/up for MTP layers, ignore base model
                if weight_name not in name:
                    continue
                if not self.is_mtp_param(name):
                    break
                merged_name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if merged_name.endswith(".bias") and merged_name not in params_dict:
                    continue
                if merged_name not in params_dict:
                    break
                param = params_dict[merged_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Only accept weights that belong to MTP layers
                if not self.is_mtp_param(name):
                    continue
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def map_model_name_to_mtp_param_name(self, name: str) -> str:
        import re

        mtp_specific_params = ["e_norm", "h_norm", "linear_projection"]
        shared_params_name_map = {
            "model.norm": "final_layernorm",
            "model.embed_tokens": "embed_tokens",
        }
        for origin_name, mtp_name in shared_params_name_map.items():
            if origin_name in name:
                return name.replace(origin_name, mtp_name)
        pattern = r"model\.mtp_layers\.(\d+)\."
        group = re.match(pattern, name)
        if group is None:
            return name
        layer_id = group.group(1)
        for param in mtp_specific_params:
            if param in name:
                return name.replace(group.group(), f"mtp_layers.{layer_id}.")
        else:
            return name.replace(group.group(), f"mtp_layers.{layer_id}.mtp_block.")

    def is_mtp_param(self, name: str) -> bool:
        if (
            "mtp_layers" in name
            or "final_layernorm" in name
            or "embed_tokens" in name
            or "lm_head" in name
        ):
            return True
        return False

    def set_embed_and_head(self, embed, head):
        del self.embed_tokens.weight
        del self.lm_head.weight
        self.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [MiniMaxM2ForCausalLMNextN]
