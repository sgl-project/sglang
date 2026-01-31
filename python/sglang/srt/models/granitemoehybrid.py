from typing import Iterable, Optional

import torch
from torch import nn
from transformers.models.granitemoeshared import GraniteMoeSharedConfig

from sglang.srt.configs.granitemoehybrid import GraniteMoeHybridConfig
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.utils import make_layers

from .granitemoe import GraniteMoeMoE


# in vLLM this is in a separate file, but keeping it here for decoupling
class GraniteMoeSharedMLP(nn.Module):
    def __init__(
        self,
        config: GraniteMoeSharedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.shared_intermediate_size
        self.input_linear = MergedColumnParallelLinear(
            input_size=self.input_size,
            output_sizes=[self.hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_linear",
        )
        self.output_linear = RowParallelLinear(
            self.hidden_size,
            self.input_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_linear",
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.input_linear(hidden_states)
        x = self.act_fn(gate_up)
        x, _ = self.output_linear(x)
        return x


class GraniteMoeHybridMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.mamba = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.block_sparse_moe = None
        if getattr(config, "num_local_experts", 0) > 0:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_idx,
                quant_config=quant_config,
                tp_size=get_tensor_model_parallel_world_size(),
                prefix=f"{prefix}.block_sparse_moe",
            )

        self.shared_mlp = (
            None
            if getattr(config, "shared_intermediate_size", 0) == 0
            else GraniteMoeSharedMLP(
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: ForwardBatch,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = torch.empty_like(hidden_states)
        attn_backend = forward_batch.attn_backend
        assert isinstance(attn_backend, HybridLinearAttnBackend)
        assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)
        attn_backend.linear_attn_backend.forward(
            mixer=self.mamba,
            layer_id=self.layer_idx,
            hidden_states=hidden_states,
            output=output,
            use_triton_causal_conv=True,
        )

        hidden_states = residual + output * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            if self.block_sparse_moe is not None:
                hidden_states = self.block_sparse_moe(hidden_states)
            # else: skip
        else:
            # create a copy since block_sparse_moe modifies in-place
            if self.block_sparse_moe is not None:
                moe_hidden_states = hidden_states.clone()
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
                hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
                del moe_hidden_states
            else:
                hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual


class GraniteMoeHybridAttention(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.causal = True
        self.hidden_size = config.hidden_size
        self.attention_bias = config.attention_bias
        self.attention_multiplier = config.attention_multiplier
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads

        # TensorParallel logic
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_key_value_heads = max(1, self.total_num_kv_heads // tp_size)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.position_embedding_type == "rope":

            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,  # its not in the config
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                rope_scaling=config.rope_scaling,
            )
        else:
            self.rotary_emb = None

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.attention_multiplier,
            num_kv_heads=self.num_key_value_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=-1,
        )

        if self.rotary_emb is not None:
            query, key = self.rotary_emb(positions, query, key)

        hidden_states = self.attn(query, key, value, forward_batch=forward_batch)
        del query, key, value

        hidden_states = self.o_proj(hidden_states)[0]
        return hidden_states


class GraniteMoeHybridAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.self_attn = GraniteMoeHybridAttention(
            config,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.block_sparse_moe = None
        if getattr(config, "num_local_experts", 0) > 0:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_idx,
                quant_config=quant_config,
                tp_size=get_tensor_model_parallel_world_size(),
                prefix=f"{prefix}.block_sparse_moe",
            )

        self.shared_mlp = (
            None
            if getattr(config, "shared_intermediate_size", 0) == 0
            else GraniteMoeSharedMLP(
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: ForwardBatch | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            if self.block_sparse_moe is not None:
                hidden_states = self.block_sparse_moe(hidden_states)
            # else: skip
        else:
            # create a copy since block_sparse_moe modifies in-place
            if self.block_sparse_moe is not None:
                moe_hidden_states = hidden_states.clone()
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
                hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
                del moe_hidden_states
            else:
                hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": GraniteMoeHybridAttentionDecoderLayer,
    "mamba": GraniteMoeHybridMambaDecoderLayer,
}


class GraniteMoeHybridModel(nn.Module):
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.embedding_multiplier = config.embedding_multiplier

        def get_layer(idx: int, prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[config.layer_types[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from the model."""
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
                hidden_states = hidden_states * self.embedding_multiplier
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                forward_batch,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class GraniteMoeHybridForCausalLM(
    nn.Module,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "conv1d": ["conv1d"],
        "in_proj": ["in_proj"],
        "input_linear": ["input_linear"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.capture_aux_hidden_states = False
        self.pp_group = get_pp_group()

        self.quant_config = quant_config
        self.config = config
        self.model = GraniteMoeHybridModel(
            config=config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(
            config,
            logit_scale=1 / self.config.logits_scaling,
        )

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
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

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # layers.0.block_sparse_moe.expert_0.input_linear.input_scale
        ckpt_gate_proj_name = "gate_proj"
        ckpt_down_proj_name = "down_proj"
        ckpt_up_proj_name = "up_proj"
        num_experts = self.config.num_local_experts

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "block_sparse_moe.experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "block_sparse_moe.experts.w2_"
                ),
                f"block_sparse_moe.experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        def _load(n, p):
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, p)
            loaded_params.add(n)

        def _load_shard(n, p, shard_id):
            # Skip layers on other devices.
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, p, shard_id)
            loaded_params.add(n)

        def _load_expert(n, p, name, shard_id, expert_id):
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, p, name, shard_id=shard_id, expert_id=expert_id)
            loaded_params.add(n)

        def _load_quant_expert(name, loaded_weight):
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping

                if weight_name not in name:
                    continue

                name_mapped = name.replace(weight_name, param_name)

                # Skip layers on other devices.
                # if is_pp_missing_parameter(name_mapped, self):
                #     continue

                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                success = False

                if weight_loader is not None:
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )

                if success:
                    return name_mapped
            return None

        for n, p in weights:
            if "A_log" in n:
                n = n.replace("A_log", "A")

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(n)
            ):
                # Loading kv cache quantization scales
                loaded_weight = p
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                _load(scale_name, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if _load_quant_expert(n, p):
                continue

            # Logic analogous to: https://github.com/vllm-project/vllm/blob/f49e5aff11c986ed4d45202b1716c5d74786efa9/vllm/model_executor/models/granitemoeshared.py#L215
            # Mapping different experts' layout:
            #  from HF (input_linear, output_linear, router)
            #  to vLLM (experts_w13({e}.w1, {e}.w2), experts_w3({e}.w3), gate)
            # The renaming and parameter loading logic is the same for weight
            # and weight_scale tensors so we can reuse them without issues.
            if n.endswith(".block_sparse_moe.input_linear.weight") or n.endswith(
                ".block_sparse_moe.input_linear.weight_scale"
            ):
                for e in range(p.size(0)):
                    w1_name = n.replace(
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w1.weight",
                    )
                    w3_name = n.replace(
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w3.weight",
                    )
                    w1_param, w3_param = p[e].chunk(2, dim=0)
                    _load_expert(
                        n.replace(".input_linear.", ".experts.w13_"),
                        w1_param,
                        w1_name,
                        shard_id="w1",
                        expert_id=e,
                    )
                    _load_expert(
                        n.replace(".input_linear.", ".experts.w13_"),
                        w3_param,
                        w3_name,
                        shard_id="w3",
                        expert_id=e,
                    )
            elif n.endswith(".block_sparse_moe.output_linear.weight") or n.endswith(
                ".block_sparse_moe.output_linear.weight_scale"
            ):
                for e in range(p.size(0)):
                    w2_name = n.replace(
                        ".block_sparse_moe.output_linear.weight",
                        f".block_sparse_moe.experts.{e}.w2.weight",
                    )
                    w2_param = p[e]
                    _load_expert(
                        n.replace(".output_linear.", ".experts.w2_"),
                        w2_param,
                        w2_name,
                        shard_id="w2",
                        expert_id=e,
                    )
            elif n.endswith(".block_sparse_moe.router.layer.weight"):
                gate_name = n.replace(
                    ".block_sparse_moe.router.layer.weight",
                    ".block_sparse_moe.gate.weight",
                )
                _load(gate_name, p)
            else:
                loaded = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name in n:
                        _load_shard(
                            n.replace(weight_name, param_name), p, shard_id=shard_id
                        )
                        loaded = True
                if not loaded:
                    _load(n, p)

        return loaded_params


EntryClass = [GraniteMoeHybridForCausalLM]
