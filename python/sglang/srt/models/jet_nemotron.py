from typing import Iterable, List, Optional

import einops
import torch
import torch.nn as nn

from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.jet_nemotron.dynamic_conv import (
    DynamicShortConvolution,
)
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
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model
from sglang.srt.utils import add_prefix


class JetBlock(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        jet_block_config = JetBlockConfig(
            **self.config.efficient_attention_config[self.config.layer_types[layer_id]]
        )

        hidden_size = self.config.hidden_size
        num_heads = jet_block_config.num_heads
        head_k_dim = jet_block_config.head_dim
        total_k_dim = num_heads * head_k_dim
        head_v_dim = int(head_k_dim * jet_block_config.expand_v)
        total_v_dim = num_heads * head_v_dim
        conv_size = jet_block_config.conv_size

        self.qkvabz_proj = MergedColumnParallelLinear(
            hidden_size,
            [
                total_k_dim,
                total_k_dim,
                total_v_dim,
                num_heads,
                num_heads,
                total_v_dim,
            ],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkvabz_proj", prefix),
        )

        self.o_proj = RowParallelLinear(total_v_dim, hidden_size, bias=False)

        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(num_heads))

        self.dynamic_conv1d = DynamicShortConvolution(
            quant_config=quant_config,
            prefix=add_prefix("dynamic_conv1d", prefix),
            hidden_size=total_v_dim,
            kernel_size=conv_size,
            generator_input_size=hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
        )

        self.o_norm = RMSNormGated(
            head_v_dim,
            eps=float(jet_block_config.norm_eps),
        )

        # Attributes.
        self.conv_size = conv_size
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.total_k_dim = total_k_dim
        self.total_v_dim = total_v_dim

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkvabz, _ = self.qkvabz_proj(hidden_states)
        q, k, v, a, b, z = qkvabz.split(
            [
                self.total_k_dim,
                self.total_k_dim,
                self.total_v_dim,
                self.num_heads,
                self.num_heads,
                self.total_v_dim,
            ],
            dim=-1,
        )

        q = nn.functional.silu(q)
        q = einops.rearrange(
            q, "l (h d) -> 1 l h d", h=self.num_heads, d=self.head_k_dim
        )

        k = nn.functional.silu(k)
        k = einops.rearrange(
            k, "l (h d) -> 1 l h d", h=self.num_heads, d=self.head_k_dim
        )

        kwargs = {
            "dynamic_conv": self.dynamic_conv1d,
            "head_v_dim": self.head_v_dim,
            "a": a,
            "b": b,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "layer_id": self.layer_id,
            "hidden_states": hidden_states,
        }

        o = forward_batch.attn_backend.forward(
            q=q, k=k, v=v, layer=None, forward_batch=forward_batch, **kwargs
        ).squeeze(0)

        z = einops.rearrange(z, "l (h d) -> l h d", h=self.num_heads)

        o = self.o_norm(o, z)

        o = einops.rearrange(o, "l h d -> l (h d)")

        o, _ = self.o_proj(o)

        return o


class JetNemotronAttention(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.q_size = self.config.num_attention_heads * self.head_dim
        self.kv_size = self.config.num_key_value_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.config.hidden_size,
            self.head_dim,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.config.num_attention_heads * self.head_dim,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.config.max_position_embeddings,
            base=int(self.config.rope_theta),
            rope_scaling=self.config.rope_scaling,
        )

        match self.config.layer_types[layer_id]:
            case "attn":
                sliding_window_size = -1

            case "swa":
                sliding_window_size = self.config.efficient_attention_config["swa"][
                    "window_size"
                ]

            case _:
                raise NotImplementedError

        self.attn = RadixAttention(
            self.config.num_attention_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class JetNemotronDecoderLayer(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        alt_stream: torch.cuda.Stream | None = None,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        match config.layer_types[layer_id]:
            case "attn" | "swa":
                self.self_attn = JetNemotronAttention(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                    layer_id=layer_id,
                )

            case "jet":
                self.self_attn = JetBlock(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                    layer_id=layer_id,
                )

            case _:
                raise NotImplementedError

        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Self Attention
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, None


class JetNemotronForCausalLM(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        self.model = Qwen2Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=JetNemotronDecoderLayer,
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
    ) -> EmbeddingPoolerOutput | LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            # (param_name, shard_weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("qkvabz_proj", "q_proj", 0),
            ("qkvabz_proj", "k_proj", 1),
            ("qkvabz_proj", "v_proj", 2),
            ("qkvabz_proj", "a_proj", 3),
            ("qkvabz_proj", "b_proj", 4),
            ("qkvabz_proj", "g_proj", 5),
        ]

        params_dict = dict(self.named_parameters())
        for weight_name, loaded_weight in weights:
            # Handle stacked parameters first.
            for (
                param_name_part,
                shard_weight_name_part,
                shard_id,
            ) in stacked_params_mapping:
                if shard_weight_name_part not in weight_name.split("."):
                    continue

                param_name = weight_name.replace(
                    shard_weight_name_part, param_name_part
                )

                if param_name not in params_dict:
                    # Fall back to direct match if no such stacked parameter.
                    continue

                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break

            else:
                param_name = weight_name

                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

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
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = layer_ids


EntryClass = JetNemotronForCausalLM
