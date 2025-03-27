import math
from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Phi3Config
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@torch.jit.script
def gegelu(input, limit: Optional[float] = None):
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]
    if limit is not None:
        a_gelu = torch.where(
            torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
        )
        a_linear = torch.where(
            torch.isinf(a_linear),
            a_linear,
            a_linear.clamp(min=-limit, max=limit),
        )
    out_gelu = quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1)


class Phi3SmallMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        assert (
            self.config.hidden_act == "gegelu"
        ), "Only `gegelu` is supported for the 4.7 series of models .."
        self.hidden_size = config.hidden_size
        self.gegelu_limit = config.gegelu_limit
        self.intermediate_size = config.intermediate_size

        self.up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            2 * [self.intermediate_size],
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x):
        gate_up, _ = self.up_proj(x)
        x = gegelu(gate_up)
        x, _ = self.down_proj(x)
        return x


class Phi3SmallSelfAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.sparse_block_size = config.blocksparse_block_size
        self.homo_heads = config.blocksparse_homo_head_pattern
        self.local_blocks = config.blocksparse_num_local_blocks
        self.vert_stride = config.blocksparse_vert_stride

        assert (
            config.blocksparse_block_size == config.blocksparse_triton_kernel_block_size
        )

        self.hidden_size = config.hidden_size
        # Number of Query Heads
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        # Number of total Key Value Heads before tensor parallel
        self.num_key_value_heads = config.num_key_value_heads
        self.num_q_per_kv = self.num_heads // self.num_key_value_heads
        if self.tp_size > 1:
            assert self.num_key_value_heads % self.tp_size == 0
        self.num_kv_heads_per_partion = max(1, self.num_key_value_heads // self.tp_size)
        self.num_heads_per_partition = self.num_heads // self.tp_size

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_embedding_base = config.rope_embedding_base
        self.rope_position_scale = config.rope_position_scale
        self.is_causal = True

        norm_factor = None
        if config.mup_use_scaling:
            norm_factor = self.head_dim / config.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(self.head_dim)
        self.scale = 1 / norm_factor

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if getattr(self.config, "rope_scaling", None) is not None:
            rope_scaling = self.config.rope_scaling
            for key in rope_scaling:
                if isinstance(rope_scaling[key], list):
                    rope_scaling[key] = tuple(rope_scaling[key])

            if "factor" not in rope_scaling:
                rope_scaling["factor"] = self.rope_position_scale
        else:
            rope_scaling = {
                "rope_type": "linear",
                "factor": self.rope_position_scale,
            }

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_embedding_base,
            rope_scaling=rope_scaling,
        )

        # blocksparse params
        self.blocksparse_block_size = config.blocksparse_block_size
        self.blocksparse_num_local_blocks = config.blocksparse_num_local_blocks
        self.blocksparse_vert_stride = config.blocksparse_vert_stride

        use_dense_attn = (
            getattr(self.config, "dense_attention_every_n_layers", None)
            and (self.layer_id + 1) % self.config.dense_attention_every_n_layers == 0
        )

        bs_params = None
        if not use_dense_attn:
            bs_params = {
                "max_seqlen": self.max_position_embeddings,
                "num_heads": self.num_heads_per_partition,
                "num_kv_heads": self.num_kv_heads_per_partion,
                "block_size": self.sparse_block_size,
                "local_blocks": self.local_blocks,
                "vert_stride": self.vert_stride,
                "homo_head": self.homo_heads,
            }

        self.attn = RadixAttention(
            self.num_heads_per_partition,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads_per_partion,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        qkv, _ = self.query_key_value(hidden_states)

        qkv = qkv.view(qkv.shape[:-1] + (-1, (self.num_q_per_kv + 2), self.head_dim))
        q, k, v = qkv.split([self.num_q_per_kv, 1, 1], dim=-2)

        # NOTE: this is required by RotaryEmbed, which indeed does not have to
        # TODO: allow 3D QK for rotary forward
        q = q.reshape(-1, self.head_dim * self.num_heads_per_partition)
        k = k.reshape(-1, self.head_dim * self.num_kv_heads_per_partion)
        v = v.reshape(-1, self.head_dim * self.num_kv_heads_per_partion)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.dense(attn_output)

        return output


class Phi3SmallDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Phi3SmallSelfAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Phi3SmallMLP(
            config,
            quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Phi3SmallModel(nn.Module):

    def __init__(
        self,
        config: Phi3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.mup_embedding_multiplier = config.mup_embedding_multiplier
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Phi3SmallDecoderLayer(
                config,
                int(prefix.split(".")[-1]),
                quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor],
    ) -> Union[torch.Tensor]:

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        if (
            self.mup_embedding_multiplier is not None
            and self.mup_embedding_multiplier > 0.0
        ):
            hidden_states = hidden_states * self.mup_embedding_multiplier

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states, forward_batch=forward_batch)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class Phi3SmallForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: Phi3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):

        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.model = Phi3SmallModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.vocab_size = config.vocab_size
        self.mup_width_multiplier = config.mup_width_multiplier
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # tokens in tiktoken but not used
        if hasattr(config, "dummy_token_indices"):
            device = self.lm_head.weight.device
            self.register_buffer(
                "dummy_token_indices",
                torch.LongTensor(config.dummy_token_indices).to(device),
                persistent=False,
            )
        else:
            self.dummy_token_indices = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(
            input_ids, self.lm_head, hidden_states, sampling_metadata
        )
        if self.dummy_token_indices is not None and logits is not None:
            logits.index_fill_(-1, self.dummy_token_indices, -torch.inf)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds,
        )

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )

        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = Phi3SmallForCausalLM
