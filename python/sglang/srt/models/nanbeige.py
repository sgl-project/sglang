import logging
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.configs import NanbeigeConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

class NanbeigeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        NanbeigeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class NanbeigeMLP(nn.Module):
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
    
    def forward(self, x, use_reduce_scatter: bool = False):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        return x
    
class NanbeigeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: NanbeigeConfig, 
                 layer_id: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0, "num_attention_heads must be divisible by tp_size."
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)

        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads >= tp_size, "num_key_value_heads must be greater than tp_size."
        assert self.total_num_kv_heads % tp_size == 0, "num_key_value_heads must be divisible by tp_size."
        self.num_kv_heads = config.num_key_value_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.total_layers = config.num_hidden_layers
        self.num_loops = config.num_loops


        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = nn.ModuleList()
        base_layer_id = layer_id
        for loop_idx in range(self.num_loops):
            layer_id = base_layer_id + loop_idx * self.total_layers
            self.attn.append(
                RadixAttention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("attn", prefix),
                )
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        loop_idx: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn[loop_idx](q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output
    
class NanbeigeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NanbeigeConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()

        self.self_attn = NanbeigeAttention(config=config, 
                                           layer_id=layer_id, 
                                           quant_config=quant_config, 
                                           prefix=add_prefix("self_attn", prefix))

        self.mlp = NanbeigeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        loop_idx: int,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            loop_idx=loop_idx,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
    
class NanbeigeModel(nn.Module):
    def __init__(
        self,
        config: NanbeigeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = NanbeigeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        pp_size = self.pp_group.world_size
        assert pp_size == 1, "The NanbeigeModel only supports a pipeline parallelism (PP) value of 1."

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix)
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to NanbeigeDecoderLayer
        decoder_layer_type = decoder_layer_type or NanbeigeDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embedding(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for loop_idx in range(self.config.num_loops):
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual if residual is not None else hidden_states
                    )
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    loop_idx,
                    residual,
                )

            # Match the reference HF semantics for Nanbeige "loop models":
            # - At the end of each full loop (except the last), convert the
            #   (hidden_states, residual) representation into real hidden_states
            #   by applying the missing residual addition.
            # - If skip_loop_final_norm=False, HF applies RMSNorm after each loop
            #   (including intermediate loops) before entering the next loop.
            if loop_idx != self.config.num_loops - 1:
                if residual is not None:
                    hidden_states = hidden_states + residual
                    residual = None
                if not self.config.skip_loop_final_norm:
                    hidden_states = self.norm(hidden_states)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states
    
            
class NanbeigeForCausalLM(nn.Module):
    def __init__(
        self,
        config: NanbeigeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = NanbeigeModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens
    
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
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states
    
    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer
        
    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        assert False, "NanbeigeModel does not support split_prefill."

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
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

            for param_name, weight_name, shard_id in stacked_params_mapping:
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

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class NanbeigeForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: NanbeigeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = NanbeigeModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "NanbeigeForSequenceClassification is only used for embedding"

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        logits = self.score(hidden_states)
        pooled_logits = self.pooler(logits, forward_batch).embeddings

        return EmbeddingPoolerOutput(pooled_logits)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Filter out lm_head weights of NanbeigeForCausalLM
        filtered_weights = [
            (name, w) for name, w in weights if not name.startswith("lm_head")
        ]
        return NanbeigeForCausalLM.load_weights(self, filtered_weights)


EntryClass = [NanbeigeForCausalLM, NanbeigeForSequenceClassification]