# Copyright 2025 Qwen Team
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
# ==============================================================================
"""Inference-only Qwen3.5 MoE model compatible with HuggingFace weights."""
import logging
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

# Configs
from sglang.srt.configs.qwen3_next_vl import Qwen3NextVLMoEConfig

# Distributed
from sglang.srt.distributed.parallel_state import get_pp_group

# Environment
from sglang.srt.environ import envs
from sglang.srt.server_args import get_global_server_args

# Layers - Attention
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope

# Layers - Linear
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

# Layers - Others
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

# Model Executor
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)

# Models
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_5 import Qwen3_5LLMModel

# Utils
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    get_int_env_var,
    is_cuda,
    is_npu,
    make_layers,
    set_weight_attrs,
)
from sglang.srt.utils.hf_transformers_utils import get_processor


logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()

cached_get_processor = lru_cache(get_processor)

class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5 MoE Vision-Language Model."""
    
    def __init__(
        self,
        config: Qwen3NextVLMoEConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5LLMModel,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
    
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        # Debug: print all parameter names to understand the model structure
        # for name, param in params_dict.items():
        #     print(name, param.shape)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # print(name, param_name, weight_name, shard_id)
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # [TODO] Skip layers that are on other devices (check if sglang has a similar function)
                # if is_pp_missing_parameter(name, self):
                #     continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # print(name_mapped, 'is fused experts', is_fused_expert, loaded_weight.shape)
                    if is_fused_expert:
                        ### [TODO] no chunk?
                        # loaded_weight = loaded_weight.transpose(-1, -2)  # no bias
                        if "experts.gate_up_proj" in name:
                            # print(name, loaded_weight.shape)
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                # loaded_weight,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                # loaded_weight,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue
                                        
                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        print(f"Parameter {name} not found in params_dict")
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)
        
        # # Dump all loaded weights for verification
        # import os
        # dump_dir = "/cpfs01/user/yuche.lz/open3_5/weight_dump"
        # os.makedirs(dump_dir, exist_ok=True)
        
        # print(f"\n{'='*60}")
        # print(f"Dumping model weights to {dump_dir}")
        # print(f"{'='*60}")
        
        # # Save each parameter with its name, shape, and stats
        # weight_info = {}
        # for name, param in params_dict.items():
        #     if hasattr(param, 'data'):
        #         data = param.data
        #         weight_info[name] = {
        #             'shape': list(data.shape),
        #             'dtype': str(data.dtype),
        #             'mean': float(data.float().mean().item()) if data.numel() > 0 else 0,
        #             'std': float(data.float().std().item()) if data.numel() > 1 else 0,
        #             'min': float(data.float().min().item()) if data.numel() > 0 else 0,
        #             'max': float(data.float().max().item()) if data.numel() > 0 else 0,
        #             'loaded': name in loaded_params,
        #         }
        #         print(f"{name}: shape={data.shape}, dtype={data.dtype}, mean={weight_info[name]['mean']:.6f}, std={weight_info[name]['std']:.6f}, loaded={weight_info[name]['loaded']}")
        
        # # Save weight statistics as JSON
        # import json
        # with open(os.path.join(dump_dir, "weight_stats.json"), "w") as f:
        #     json.dump(weight_info, f, indent=2)
        
        # # Save actual weights as safetensors
        # try:
        #     from safetensors.torch import save_file
        #     weights_to_save = {}
        #     for name, param in params_dict.items():
        #         if hasattr(param, 'data'):
        #             weights_to_save[name] = param.data.contiguous().cpu()
        #     save_file(weights_to_save, os.path.join(dump_dir, "model_weights.safetensors"))
        #     print(f"\nWeights saved to {os.path.join(dump_dir, 'model_weights.safetensors')}")
        # except Exception as e:
        #     print(f"Failed to save safetensors: {e}")
        #     # Fallback to torch.save
        #     torch.save({name: param.data.cpu() for name, param in params_dict.items() if hasattr(param, 'data')}, 
        #                os.path.join(dump_dir, "model_weights.pt"))
        #     print(f"Weights saved to {os.path.join(dump_dir, 'model_weights.pt')}")
        
        # print(f"\nTotal parameters in model: {len(params_dict)}")
        # print(f"Loaded parameters: {len(loaded_params)}")
        # print(f"Missing parameters: {set(params_dict.keys()) - loaded_params}")
        # print(f"{'='*60}\n")
        
        return loaded_params

EntryClass = Qwen3_5MoeForConditionalGeneration