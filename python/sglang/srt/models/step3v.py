import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers, log_info_on_rank0
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.moe.topk import TopK


logger = logging.getLogger(__name__)

Step3vConfig = None

class Step3vMLP(nn.Module):
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
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Step3vMoEMLP(nn.Module):
    # Native 
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )
            
        self.topk = TopK(
            top_k=config.moe_top_k,
            renormalize=config.norm_expert_weight,
            use_grouped_topk=False,
        )
        
        self.experts = get_moe_impl_class()(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            output_size=config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if global_server_args_dict["enable_deepep_moe"]:
            raise NotImplementedError(
                "DeepEP MoE is not supported yet in Step2Mini model."
            )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        if not global_server_args_dict["enable_deepep_moe"]:
            return self.forward_normal(hidden_states)
        else:
            raise NotImplementedError(
                "DeepEP MoE is not supported yet in Step3v model."
            )

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)

class Step3vAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int, 
        share_q_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.all_tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        self.attn_tp_rank = attn_tp_rank
        self.layer_id = layer_id
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = share_q_dim if share_q_dim else head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        self.qkv_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.q_size, self.kv_size, self.kv_size],
            bias=False,
            quant_config=quant_config,
            tp_rank=0,      # In fact, we need a MergedReplicatedLinear
            tp_size=1,      
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        
        self.inter_norm = RMSNorm(self.q_size, eps=rms_norm_eps)
        
        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=f"{prefix}.wq",
        )
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
        self.iter = 0

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        self.iter += 1
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.inter_norm(q.contiguous())
        q, _ = self.wq(q)
        q, k = self.rotary_emb(positions, q, k)
        # torch.save([q,k,v], f"fa3qkv_{self.attn_tp_rank}.pt")
        # torch.save([q,k,v], f"dpattn_qkv_{self.all_tp_rank}_{self.iter}.pt")
        # torch.save([q,k,v], f"fa3qkv_{self.attn_tp_rank}_{self.layer_id}.pt")
        attn_output = self.attn(q, k, v, forward_batch)
        # torch.save([attn_output], f"flinferattn_{self.attn_tp_rank}_{self.layer_id}.pt")
        # torch.save([attn_output], f"fa3attn_{self.attn_tp_rank}_{self.layer_id}.pt")
        output, _ = self.o_proj(attn_output)
        return output
    
class Step3vDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Step3vConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.n_shared_experts = 1
        self.num_fused_shared_experts = (
            0
            if global_server_args_dict["disable_shared_experts_fusion"]
            else self.n_shared_experts
        )
        self.num_fused_shared_experts = 0
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = Step3vAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads, 
            num_kv_heads=1,
            head_dim=head_dim,
            share_q_dim=config.share_q_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(',')]
        else:
            # Default to 1dense.
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
        
        self.use_moe = False

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_id = layer_id
        self.is_layer_sparse = True if layer_id in moe_layers_idx else False
        self.is_previous_layer_sparse = True if layer_id - 1 in moe_layers_idx else False

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=self.is_previous_layer_sparse,
        )

        if not self.is_layer_sparse:
            self.mlp = Step3vMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.use_moe = True
            if self.num_fused_shared_experts == 0:
                self.moe = Step3vMoEMLP(
                    layer_id = layer_id,
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp", prefix),
                )
                self.share_expert = Step3vMLP(
                    hidden_size = config.hidden_size,
                    intermediate_size = config.share_expert_dim,
                    hidden_act="silu",
                    quant_config=quant_config,
                    prefix=add_prefix("share_expert", prefix),
                )
            else:
                self.moe = Step3vMoEMLP(
                    layer_id = layer_id,
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp", prefix),
                )
                    
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def moe_mlp_forward(self, hidden_states):
        if not self.num_fused_shared_experts:
            h = hidden_states.clone()
            hidden_states = self.moe(hidden_states)
            hidden_states += self.share_expert(h)
        else:
            hidden_states = self.moe(hidden_states)
        return hidden_states

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        if self.use_moe:
            hidden_states = self.moe_mlp_forward(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        
        return hidden_states, residual


class Step3vModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Step3vDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    

class Step3vForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Step3vModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        #self.vision_model = StepCLIPVisionTransformer()
        # TODO: after textmodel is ok.
        self.n_shared_experts = 1
        # self.num_fused_shared_experts = (
        #     0
        #     if global_server_args_dict["disable_shared_experts_fusion"]
        #     else self.n_shared_experts
        # )
        self.num_fused_shared_experts = 0
        self.config.tie_word_embeddings = False
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        #TODO: 
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", 0),
            (".qkv_proj", ".k_proj", 1),
            (".qkv_proj", ".v_proj", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        
        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")
        
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.moe_num_experts + self.num_fused_shared_experts,
        )
                
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        
        def match_expert_and_shard_ids(name_path: str, weight_path: str) -> bool:
            name_parts = name_path.split('.')
            weight_parts = weight_path.split('.')
            # print("shard_id", name_parts[4], weight_parts[2])
            shard_id_matches = name_parts[4] == weight_parts[2]
            return shard_id_matches



        for name, loaded_weight in weights:
            exclude = False
            for exclude_name in ["vit_downsampler", "vision_model", "vit_large_projector"]:
                if exclude_name in name:
                    exclude = True
            if exclude:
                continue
            #TODO: support vision model
            if self.num_fused_shared_experts > 0 and "share" in name:
                # assert False
                FLAG = 0 
                name = name.replace("share_expert", "moe")
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if expert_id != self.config.moe_num_experts or not match_expert_and_shard_ids(name, weight_name):
                        continue
                
                    part_name = weight_name.split('.')[-2]
                    fake_weight_name = name.replace(part_name, weight_name[:-1])
                    actual_param_name = name.replace(part_name+'.', param_name)
                    param = params_dict[actual_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    print("actual_param_name", actual_param_name)
                    print("name ", name, expert_id, shard_id    )
                    # loaded_params.add(actual_param_name)
                    FLAG = 1
                    break
                # assert FLAG == 1
                continue
        
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'gate.' not in name and 'moe' in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                # Check if param has too many zeros
                # assert zero_ratio < 0.25, f"Parameter {name} has {zero_ratio:.2%} zeros (threshold: 25%)"
                loaded_params.add(name)
                break
            else:
                if "moe" not in name:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)   
                else:
                    if 'gate.' in name:
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                        continue
                    
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping      
                        if expert_id == self.config.moe_num_experts:
                            continue
                        if not match_expert_and_shard_ids(name, weight_name):
                            continue
                        part_name = weight_name.split('.')[-2]
                        fake_weight_name = name.replace(part_name, weight_name[:-1])
                        actual_param_name = name.replace(part_name+'.', param_name)
                        param = params_dict[actual_param_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(actual_param_name)
                        # Don't break here, because this 'loaded_weight' includes all the weights for this layer
                
        print(params_dict.keys()-loaded_params)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.moe_num_experts,
            num_groups=None,
        )

class MMGPTStep3vForCausalLM(Step3vForConditionalGeneration):
    pass

EntryClass = MMGPTStep3vForCausalLM