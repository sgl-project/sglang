# Adapted from qwen2.py
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.managers.scheduler import StreamGuardResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix, is_cuda, is_flashinfer_available

Qwen3GuardModelConfig = None

logger = logging.getLogger(__name__)
_is_flashinfer_available = is_flashinfer_available()
_is_cuda = is_cuda()

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm


class Qwen3ForGuardModel(nn.Module):
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

    def __init__(
        self,
        config: Qwen3GuardModelConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self.norm_class = Qwen3MoeRMSNorm

        self.risk_level_category_pre = nn.Linear(
            config.hidden_size, config.guard_inner_size, bias=False
        )
        self.risk_level_category_layernorm = self.norm_class(
            config.guard_inner_size, eps=config.rms_norm_eps
        )
        self.risk_level_head = nn.Linear(
            config.guard_inner_size, config.num_risk_level, bias=False
        )
        self.category_head = nn.Linear(
            config.guard_inner_size, config.num_category, bias=False
        )
        self.query_risk_level_category_pre = nn.Linear(
            config.hidden_size, config.guard_inner_size, bias=False
        )
        self.query_risk_level_category_layernorm = self.norm_class(
            config.guard_inner_size, eps=config.rms_norm_eps
        )
        self.query_risk_level_head = nn.Linear(
            config.guard_inner_size, config.num_query_risk_level, bias=False
        )
        self.query_category_head = nn.Linear(
            config.guard_inner_size, config.num_query_category, bias=False
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            logits_to_keep = None
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            risk_level_category_x = self.risk_level_category_pre(
                hidden_states[:, slice_indices, :]
            )
            risk_level_category_x = self.risk_level_category_layernorm(
                risk_level_category_x
            )
            risk_level_logits = self.risk_level_head(risk_level_category_x)
            category_logits = self.category_head(risk_level_category_x)
            query_risk_level_category_x = self.query_risk_level_category_pre(
                hidden_states[:, slice_indices, :]
            )
            query_risk_level_category_x = self.query_risk_level_category_layernorm(
                query_risk_level_category_x
            )
            query_risk_level_logits = self.query_risk_level_head(
                query_risk_level_category_x
            )
            query_category_logits = self.query_category_head(
                query_risk_level_category_x
            )
            loss = None
            return StreamGuardResult(
                risk_level_logits=risk_level_logits,
                category_logits=category_logits,
                query_risk_level_logits=query_risk_level_logits,
                query_category_logits=query_category_logits,
                hidden_states=hidden_states,
                bid=-1,
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
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

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = Qwen3ForGuardModel
