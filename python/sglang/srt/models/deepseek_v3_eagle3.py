"""
Copyright 2024 SGLang Team
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

from sglang.srt.utils import add_prefix
# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only DeepSeek V3-EAGLE3 model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
)
from sglang.srt.utils import BumpAllocator


class Eagle3MLP(nn.Module):
    """Eagle3专用的MLP层，用于将2*hidden_size转换为hidden_size"""
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.input_size = config.hidden_size * 2
        self.hidden_size = config.hidden_size
        
        
        self.input_norm = RMSNorm(self.input_size, eps=config.rms_norm_eps)
        
        self.proj = ReplicatedLinear(
            self.input_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )
        
    def forward(self, x):
        
        x = self.input_norm(x)
        output, _ = self.proj(x)
        return output


class DeepseekV3DecoderLayerEagle3(DeepseekV2DecoderLayer):
    """继承DeepSeek V2 DecoderLayer，类似LLaMA Eagle3的做法"""
    
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
        
        if hasattr(self.self_attn, 'attn_mha'):
            self.self_attn.attn_mha.layer_id = -1  # 禁用MHA的KV缓存
            
        # 确保MLA使用正确的layer_id
        if hasattr(self.self_attn, 'attn_mqa'):
            self.self_attn.attn_mqa.layer_id = layer_id
            
        # 设置标志，强制使用MLA
        self._force_mla_mode = True
        if not hasattr(self.self_attn, 'w_kc'):
            self.self_attn.w_kc = None
        if not hasattr(self.self_attn, 'w_vc'):
            self.self_attn.w_vc = None
        if not hasattr(self.self_attn, 'w_scale'):
            self.self_attn.w_scale = 1.0
        
        self.input_proj = Eagle3MLP(
            config, 
            quant_config=quant_config, 
            prefix=add_prefix("input_proj", prefix)
        )
        
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eagle3特有的forward逻辑"""
        
        if hasattr(self.self_attn, 'dispatch_attn_forward_method'):
            original_dispatch = self.self_attn.dispatch_attn_forward_method
            def force_mla_dispatch(forward_batch):
                from sglang.srt.models.deepseek_v2 import AttnForwardMethod
                return AttnForwardMethod.MLA
            self.self_attn.dispatch_attn_forward_method = force_mla_dispatch
        
        try:
            
            if self.self_attn.w_kc is None or self.self_attn.w_vc is None:
                self._ensure_mla_weights_initialized()
            
            residual = hidden_states
            embeds = self.input_layernorm(embeds)
            hidden_states = self.hidden_norm(hidden_states)

            concat_hidden = torch.cat([embeds, hidden_states], dim=-1)
            projected_hidden = self.input_proj(concat_hidden)
            
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=projected_hidden,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

            hidden_states = self.mlp(hidden_states, forward_batch)

            return hidden_states, residual

        finally:
            if hasattr(self.self_attn, 'dispatch_attn_forward_method'):
                self.self_attn.dispatch_attn_forward_method = original_dispatch


        

    def _ensure_mla_weights_initialized(self):
        """确保MLA权重被正确初始化"""
        self_attn = self.self_attn
        
        if self_attn.w_kc is not None and self_attn.w_vc is not None:
            return
        
        print(f"DEBUG: Initializing MLA weights for Eagle3 layer")
        
        from sglang.srt.utils import is_hip, is_cuda
        
        _is_cuda = is_cuda()
        _is_hip = is_hip()
        _is_fp8_fnuz = False
        
        if not _is_hip:
            try:
                from sgl_kernel import awq_dequantize
            except ImportError:
                from vllm._custom_ops import awq_dequantize
        else:
            from sglang.srt.layers.quantization.awq_triton import (
                awq_dequantize_triton as awq_dequantize,
            )
        
        from sglang.srt.layers.quantization.fp8_utils import (
            block_quant_to_tensor_quant,
            channel_quant_to_tensor_quant,
            normalize_e4m3fn_to_e4m3fnuz,
        )
        from sglang.srt.layers.quantization.int8_utils import (
            block_dequant as int8_block_dequant,
        )
        
        if hasattr(self_attn.kv_b_proj, "qweight"):
            if _is_cuda or _is_hip:
                w = awq_dequantize(
                    self_attn.kv_b_proj.qweight,
                    self_attn.kv_b_proj.scales,
                    self_attn.kv_b_proj.qzeros,
                ).T
            else:
                w = awq_dequantize(
                    self_attn.kv_b_proj.qweight,
                    self_attn.kv_b_proj.scales,
                    self_attn.kv_b_proj.qzeros,
                    0, 0, 0,
                ).T
        else:
            w = self_attn.kv_b_proj.weight

        if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            if hasattr(self_attn.kv_b_proj, "weight_scale_inv"):
                weight_scale = self_attn.kv_b_proj.weight_scale_inv
            elif hasattr(self_attn.kv_b_proj, "weight_scale"):
                weight_scale = self_attn.kv_b_proj.weight_scale
            else:
                weight_scale = None
            
            if weight_scale is not None:
                w, scale = channel_quant_to_tensor_quant(w, weight_scale)
                self_attn.w_scale = scale
        elif w.dtype == torch.int8:
            if hasattr(self_attn.kv_b_proj, "weight_scale"):
                w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(torch.bfloat16)
        
        w_kc, w_vc = w.unflatten(
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
        
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
        
        if hasattr(self_attn.kv_b_proj, "weight_scale") and self_attn.w_scale == 1.0:
            self_attn.w_scale = self_attn.kv_b_proj.weight_scale
            if _is_hip:
                self_attn.w_scale *= 2.0
        
        print(f"DEBUG: MLA weights initialized - w_kc: {self_attn.w_kc.shape}, w_vc: {self_attn.w_vc.shape}")


class DeepseekV3ModelEagle3(DeepseekV2Model):
    """继承DeepseekV2Model，确保使用正确的KV缓存配置"""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        original_num_layers = config.num_hidden_layers
        config.num_hidden_layers = 1  # 临时设置为1层
        
        super().__init__(config, quant_config, prefix)
        
        config.num_hidden_layers = original_num_layers
        
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.layers[0] = DeepseekV3DecoderLayerEagle3(
            config, 
            layer_id=0,
            quant_config=quant_config, 
            prefix=add_prefix("midlayer", prefix)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=embeds.device,
        )

        residual = None
        
        hidden_states, residual = self.layers[0](
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
            zero_allocator,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]

class DeepseekV3ForCausalLMEagle3(DeepseekV2ForCausalLM):
    """关键：继承DeepseekV2ForCausalLM，类似LLaMA Eagle3的模式"""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        
        super().__init__(config, quant_config, prefix)

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        del self.model  # 删除父类创建的model
        self.model = DeepseekV3ModelEagle3(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        
        if hasattr(config, 'draft_vocab_size') and config.draft_vocab_size != config.vocab_size:
            del self.lm_head
            if getattr(self.config, 'tie_word_embeddings', False):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.draft_vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue
            
            if name.startswith("midlayer."):
                name = name.replace("midlayer.", "layers.0.")
            
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name_full = f"model.{name}" if name not in params_dict else name
                if param_name_full in params_dict:
                    param = params_dict[param_name_full]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name_full = name if name in params_dict else f"model.{name}"
                if param_name_full in params_dict:
                    param = params_dict[param_name_full]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def post_load_weights(self):
        """调用父类的post_load_weights来正确初始化MLA权重"""
        # 直接调用父类的post_load_weights，它会处理所有层的MLA权重初始化
        super().post_load_weights()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_hot_token_id(self):
        return self.hot_token_id
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        model_output = self.model(input_ids, positions, forward_batch, input_embeds)

        # 正确处理模型输出
        if isinstance(model_output, tuple):
            hidden_states, aux_list = model_output
            aux_hidden_states = aux_list[0] if aux_list and self.capture_aux_hidden_states else None
        else:
            hidden_states = model_output
            aux_hidden_states = None

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )


EntryClass = [DeepseekV3ForCausalLMEagle3]