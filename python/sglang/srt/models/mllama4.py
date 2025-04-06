# TODO: add Aapted from vllm/mllama4.py
import math
from collections.abc import Iterable, Mapping
from itertools import tee
from typing import List, Literal, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import BatchFeature, Llama4Config, Llama4VisionConfig
from transformers.image_utils import SizeDict
from transformers.modeling_outputs import BaseModelOutput

from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.utils import add_prefix

class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config: Llama4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:

        return self.language_model(input_ids, positions, forward_batch)

    def _separate_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        prefix: str,
    ) -> Tuple[Iterable[Tuple[str, torch.Tensor]], Iterable[Tuple[str, torch.Tensor]]]:
        weights1, weights2 = tee(weights, 2)

        def get_prefix_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, data in weights1:
                if name.startswith(prefix):
                    yield (name, data)

        def get_other_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, data in weights2:
                if not name.startswith(prefix):
                    yield (name, data)

        return get_prefix_weights(), get_other_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]

        updated_params: Set[str] = set()        
        params_dict = dict(self.named_parameters())
        
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )

        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )
        
        num_experts = self.config.text_config.num_local_experts
        
        for name, param in params_dict.items():
            print(name)
        
        for name, loaded_weight in weights:
            if name.startswith("vision_model"):
                continue
            if not name.startswith("language_model"):
                print("!!! buggy " + name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                print("!!! initializing " + name)
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".experts" in name:
                    print("loading " + name)
                    if ".gate_up_proj" in name:
                        name_list = [name.replace(".experts.gate_up_proj", ".experts.w13_weight")] * 2
                        loaded_weight_list = loaded_weight.chunk(2, dim=-1)
                        shard_id_list = ["w1", "w3"]
                    else:
                        name_list = [name.replace(".experts.down_proj", ".experts.w2_weight")]
                        loaded_weight_list = [loaded_weight]
                        shard_id_list = ["w2"]
                    for name, loaded_weight, shard_id in zip(name_list, loaded_weight_list, shard_id_list):
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in range(num_experts):
                            weight_loader(
                                param,
                                loaded_weight[expert_id],
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                    break
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                print("??? initializing " + name)
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
        return
        if True:
            # if "rotary_emb.inv_freq" in name or "projector" in name:
            #     continue
            # if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            #     # Models trained using ColossalAI may include these tensors in
            #     # the checkpoint. Skip them.
            #     continue
            # if name.startswith("model.vision_tower") and name not in params_dict:
            #     continue
            # if self.config.tie_word_embeddings and "lm_head.weight" in name:
            #     continue
            # # Handle FP8 kv-scale remapping
            # if "scale" in name:
            #     name = maybe_remap_kv_scale_name(name, params_dict)
            #     if name is None:
            #         continue
            # 
            # return

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
        
        language_model_weights, other_weights = self._separate_weights(
            weights, prefix=language_model_prefix
        )
        
        
        loader = AutoWeightsLoader(self)
        loaded_language_model_params = loader.load_weights(
            language_model_weights)
        assert loaded_language_model_params is not None
        updated_params.update(loaded_language_model_params)

        for name, loaded_weight in other_weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)

                weight_loader(param, loaded_weight)
                updated_params.add(name)
        return updated_params

EntryClass = Llama4ForConditionalGeneration
