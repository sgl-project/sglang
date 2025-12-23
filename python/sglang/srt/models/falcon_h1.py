import logging
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class FalconH1MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: int,
        mlp_multipliers: List[float],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.layer_id = layer_id

        self.intermediate_size = intermediate_size
        self.tp_size = get_tensor_model_parallel_world_size()

        self.gate_multiplier, self.down_multiplier = mlp_multipliers

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        gate_up[:, : self.intermediate_size // self.tp_size] *= self.gate_multiplier

        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        x = x * self.down_multiplier
        return x


class FalconH1HybridAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: FalconH1Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.layer_id = layer_id

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        self.d_ssm = (
            int(config.mamba_expand * config.hidden_size)
            if config.mamba_d_ssm is None
            else config.mamba_d_ssm
        )

        self.mamba = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            use_rms_norm=config.mamba_rms_norm,
            prefix=f"{prefix}.mixer",
        )

        # FalconH1 all layers are dense and have no nextn now
        self.is_layer_sparse = False
        is_previous_layer_sparse = False
        is_next_layer_sparse = False

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.feed_forward = FalconH1MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
            mlp_multipliers=config.mlp_multipliers,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.pre_ff_layernorm,
            allow_reduce_scatter=True,
        )

        self.alt_stream = alt_stream
        self.key_multiplier = config.key_multiplier

        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.ssm_in_multiplier = config.ssm_in_multiplier

        self.attention_in_multiplier = config.attention_in_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.groups_time_state_size = self.mamba.n_groups * config.mamba_d_state
        self.zxbcdt_multipliers = config.ssm_multipliers
        self._init_mup_vector()

    def _init_mup_vector(self):
        """
        Non learnable per-block scaling vector composed of element-wise
        multipliersapplied to each separate contiguous block of the output
        of the linear projection (in_proj) before further processing
        (gating, convolution, SSM):

            - Z block:  [0 : d_ssm]                      → zxbcdt_multipliers[0]
            - X block:  [d_ssm : 2 * d_ssm]              → zxbcdt_multipliers[1]
            - B block:  [2 * d_ssm : 2 * d_ssm + G * S]  → zxbcdt_multipliers[2]
            - C block:  [2 * d_ssm + G * S : 2 * d_ssm + 2 * G * S]
                        → zxbcdt_multipliers[3]
            - dt block: [2 * d_ssm + 2 * G * S : end]    → zxbcdt_multipliers[4]

        where:
            - d_ssm:     Dimension of state-space model latent
            - G:         Number of groups (n_groups)
            - S:         SSM state size per group
            - All indices are divided by tp_size to support tensor parallelism
        """
        vector_shape = (
            2 * self.d_ssm + 2 * self.groups_time_state_size + self.config.mamba_n_heads
        ) // self.tp_size
        mup_vector = torch.ones(1, vector_shape)
        # Z vector 0 -> d_ssm
        mup_vector[:, : self.d_ssm // self.tp_size] *= self.zxbcdt_multipliers[0]
        # X vector d_ssm -> 2 * d_ssm
        mup_vector[
            :, (self.d_ssm // self.tp_size) : (2 * self.d_ssm // self.tp_size)
        ] *= self.zxbcdt_multipliers[1]
        # B vector 2 * d_ssm -> 2 * d_ssm + (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm)
            // self.tp_size : (2 * self.d_ssm + self.groups_time_state_size)
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[2]
        # C vector 2 * d_ssm + (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm + self.groups_time_state_size)
            // self.tp_size : (2 * self.d_ssm + 2 * self.groups_time_state_size)
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[3]
        # dt vector 2 * d_ssm + 2 * (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state) + n_heads
        mup_vector[
            :,
            (2 * self.d_ssm + 2 * self.groups_time_state_size) // self.tp_size :,
        ] *= self.zxbcdt_multipliers[4]

        self.register_buffer("mup_vector", mup_vector, persistent=False)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k = k * self.key_multiplier
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            # Attention block
            attention_hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states * self.attention_in_multiplier,
                forward_batch=forward_batch,
            )
            attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

            attn_backend = forward_batch.attn_backend
            assert isinstance(attn_backend, HybridLinearAttnBackend)
            assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)
            # Mamba block
            mamba_hidden_states = torch.empty_like(hidden_states)
            attn_backend.linear_attn_backend.forward(
                self.mamba,
                hidden_states * self.ssm_in_multiplier,
                mamba_hidden_states,
                layer_id=self.layer_id,
                mup_vector=self.mup_vector,
            )
            mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

            hidden_states = attention_hidden_states + mamba_hidden_states

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.feed_forward(
            hidden_states, forward_batch, use_reduce_scatter
        )

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "falcon_h1": FalconH1HybridAttentionDecoderLayer,
}


class FalconH1Model(nn.Module):
    def __init__(
        self,
        config: FalconH1Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.embedding_multiplier = config.embedding_multiplier

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.infer_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        if inputs_embeds is not None:
            hidden_states = inputs_embeds * self.embedding_multiplier
        else:
            hidden_states = self.embed_tokens(input_ids) * self.embedding_multiplier

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.final_layernorm(hidden_states)
            else:
                hidden_states, _ = self.final_layernorm(hidden_states, residual)

        return hidden_states


class FalconH1ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: FalconH1Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank
        self.quant_config = quant_config
        self.model = FalconH1Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.lm_head = self.lm_head.float()
        self.lm_head_multiplier = config.lm_head_multiplier
        self.logits_processor = LogitsProcessor(
            config, logit_scale=self.lm_head_multiplier
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq" in name:
                continue

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            if "A_log" in name:
                name = name.replace("A_log", "A")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # if is_pp_missing_parameter(name, self):
                #     continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


EntryClass = FalconH1ForCausalLM
