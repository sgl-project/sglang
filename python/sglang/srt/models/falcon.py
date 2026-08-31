import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader


class FalconAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.use_alibi = config.alibi

        if config.new_decoder_architecture:
            self.num_kv_heads = config.num_kv_heads

        elif config.multi_query:
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = self.num_heads

        self.scaling = self.head_dim**-0.5

        self.attention = RadixAttention(
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            layer_id=layer_id,
        )
        self.parallel_attn = config.parallel_attn

        self.rotary_emb = None

        if self.use_alibi:
            self.alibi_slopes = _get_alibi_slopes(self.num_heads).cuda() * self.scaling
        else:
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=2048,
                base=10000,
                is_neox_style=True,
                rope_scaling=None,
            )

        self.attn_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=getattr(config, "bias", False),
        )

        self.attn_dense = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=getattr(config, "bias", False),
        )

    def forward(self, hidden_states, positions, forward_batch):
        qkv, _ = self.attn_proj(hidden_states)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k, v = qkv.split(
            [q_size, kv_size, kv_size],
            dim=-1,
        )

        v = v.contiguous()

        if self.use_alibi:
            alibi_slopes = self.alibi_slopes.view(1, self.num_heads, 1).expand(
                q.shape[0], -1, -1
            )

            hidden_states = self.attention(
                q,
                k,
                v,
                forward_batch,
                score_mod=_falcon_alibi_score_mod,
                aux_tensors=[alibi_slopes],
            )
        else:
            q, k = self.rotary_emb(positions, q, k)

            hidden_states = self.attention(
                q,
                k,
                v,
                forward_batch,
            )

        hidden_states, _ = self.attn_dense(hidden_states)

        return hidden_states


class FalconMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size

        intermediate_size = (
            getattr(config, "ffn_hidden_size", None)
            or getattr(config, "intermediate_size", None)
            or self.hidden_size * 4
        )

        self.up_projection = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=intermediate_size,
            bias=getattr(config, "bias", False),
        )

        self.down_projection = RowParallelLinear(
            input_size=intermediate_size,
            output_size=self.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.gelu = nn.GELU()

    def forward(self, hidden_states):
        hidden_states, _ = self.up_projection(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states, _ = self.down_projection(hidden_states)

        return hidden_states


class FalconDecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()

        self.attention = FalconAttention(config, layer_id)
        self.mlp = FalconMLP(config)

        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.parallel_attn = config.parallel_attn

    def forward(self, hidden_states, positions, forward_batch):
        residual = hidden_states
        if not self.parallel_attn:
            attn_input = self.input_layernorm(hidden_states)
            attn_output = self.attention(
                attn_input,
                positions,
                forward_batch,
            )

            hidden_states = residual + attn_output

            residual = hidden_states
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)

            return residual + mlp_output
        else:
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.attention(
                hidden_states,
                positions,
                forward_batch,
            )
            mlp_output = self.mlp(hidden_states)

            return residual + mlp_output + attn_output


class FalconModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                FalconDecoderLayer(config, layer_id=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(self, input_ids, positions, forward_batch):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                forward_batch,
            )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class FalconForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()

        self.logits_processor = LogitsProcessor(config)
        self.transformer = FalconModel(config)

        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_ids, positions, forward_batch):
        hidden_states = self.transformer(
            input_ids,
            positions,
            forward_batch,
        )

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )

    @torch.no_grad()
    def load_weights(self, weights_iterable):
        for name, loaded_weight in weights_iterable:
            if "transformer.word_embeddings.weight" in name:
                self.transformer.embed_tokens.weight.copy_(loaded_weight)

                weight_loader = getattr(
                    self.lm_head.weight,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(self.lm_head.weight, loaded_weight)

            elif "transformer.ln_f.weight" in name:
                self.transformer.final_layernorm.weight.copy_(loaded_weight)

            elif "transformer.ln_f.bias" in name:
                self.transformer.final_layernorm.bias.copy_(loaded_weight)

            elif "lm_head.weight" in name:
                weight_loader = getattr(
                    self.lm_head.weight,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(self.lm_head.weight, loaded_weight)

            elif "transformer.h." in name:
                layer_id = int(name.split(".")[2])
                layer = self.transformer.layers[layer_id]

                name_inside_layer = ".".join(name.split(".")[3:])

                if name_inside_layer == "input_layernorm.weight":
                    layer.input_layernorm.weight.copy_(loaded_weight)

                elif name_inside_layer == "input_layernorm.bias":
                    layer.input_layernorm.bias.copy_(loaded_weight)

                elif name_inside_layer == "post_attention_layernorm.weight":
                    layer.post_attention_layernorm.weight.copy_(loaded_weight)

                elif name_inside_layer == "post_attention_layernorm.bias":
                    layer.post_attention_layernorm.bias.copy_(loaded_weight)

                elif name_inside_layer == "self_attention.query_key_value.weight":
                    loaded_weight = _reorder_falcon_qkv_weight(
                        loaded_weight,
                        layer.attention.num_heads,
                        layer.attention.num_kv_heads,
                        layer.attention.head_dim,
                    )

                    weight_loader = getattr(
                        layer.attention.attn_proj.weight,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.attention.attn_proj.weight,
                        loaded_weight,
                    )

                elif name_inside_layer == "self_attention.query_key_value.bias":
                    loaded_weight = _reorder_falcon_qkv_bias(
                        loaded_weight,
                        layer.attention.num_heads,
                        layer.attention.num_kv_heads,
                        layer.attention.head_dim,
                    )

                    weight_loader = getattr(
                        layer.attention.attn_proj.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.attention.attn_proj.bias,
                        loaded_weight,
                    )

                elif name_inside_layer == "self_attention.dense.bias":
                    weight_loader = getattr(
                        layer.attention.attn_dense.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.attention.attn_dense.bias,
                        loaded_weight,
                    )

                elif name_inside_layer == "self_attention.dense.weight":
                    weight_loader = getattr(
                        layer.attention.attn_dense.weight,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.attention.attn_dense.weight,
                        loaded_weight,
                    )

                elif name_inside_layer == "mlp.dense_h_to_4h.bias":
                    weight_loader = getattr(
                        layer.mlp.up_projection.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.mlp.up_projection.bias,
                        loaded_weight,
                    )

                elif name_inside_layer == "mlp.dense_4h_to_h.bias":
                    weight_loader = getattr(
                        layer.mlp.down_projection.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.mlp.down_projection.bias,
                        loaded_weight,
                    )

                elif name_inside_layer == "mlp.dense_h_to_4h.weight":
                    weight_loader = getattr(
                        layer.mlp.up_projection.weight,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.mlp.up_projection.weight,
                        loaded_weight,
                    )

                elif name_inside_layer == "mlp.dense_4h_to_h.weight":
                    weight_loader = getattr(
                        layer.mlp.down_projection.weight,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        layer.mlp.down_projection.weight,
                        loaded_weight,
                    )


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))

    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )

    powers = torch.arange(
        1,
        1 + closest_power_of_2,
        dtype=torch.int32,
    )
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )

        num_remaining_heads = min(
            closest_power_of_2,
            total_num_heads - closest_power_of_2,
        )

        extra_powers = torch.arange(
            start=1,
            end=1 + 2 * num_remaining_heads,
            step=2,
            dtype=torch.int32,
        )

        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)],
            dim=0,
        )

    return slopes


def _reorder_falcon_qkv_bias(
    bias,
    num_heads,
    num_kv_heads,
    head_dim,
):
    q_per_kv = num_heads // num_kv_heads

    bias = bias.view(
        num_kv_heads,
        q_per_kv + 2,
        head_dim,
    )

    q = bias[:, :q_per_kv, :].reshape(num_heads * head_dim)

    k = bias[:, q_per_kv, :].reshape(num_kv_heads * head_dim)

    v = bias[:, q_per_kv + 1, :].reshape(num_kv_heads * head_dim)

    return torch.cat([q, k, v], dim=0)


def _reorder_falcon_qkv_weight(
    weight,
    num_heads,
    num_kv_heads,
    head_dim,
):
    in_dim = weight.shape[1]
    q_per_kv = num_heads // num_kv_heads

    weight = weight.view(
        num_kv_heads,
        q_per_kv + 2,
        head_dim,
        in_dim,
    )

    q = weight[:, :q_per_kv, :, :].reshape(
        num_heads * head_dim,
        in_dim,
    )

    k = weight[:, q_per_kv, :, :].reshape(
        num_kv_heads * head_dim,
        in_dim,
    )

    v = weight[:, q_per_kv + 1, :, :].reshape(
        num_kv_heads * head_dim,
        in_dim,
    )

    return torch.cat([q, k, v], dim=0)


@triton.jit
def _falcon_alibi_score_mod(
    qk,
    q_pos,
    kv_pos,
    q_idx,
    head,
    mask,
    Aux0,
    aux0_stride_t,
    aux0_stride_h,
    aux0_len,
):
    slope = tl.load(Aux0 + q_idx * aux0_stride_t + head * aux0_stride_h)
    return qk + slope * kv_pos


EntryClass = [FalconForCausalLM]
