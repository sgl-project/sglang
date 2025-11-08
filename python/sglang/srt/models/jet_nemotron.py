from collections.abc import Iterable
from typing import cast

import einops
import torch
import torch.nn as nn

from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
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


class DynamicShortConvolutionKernelGenerator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.w1 = ColumnParallelLinear(
            input_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )

        self.act = nn.SiLU()

        self.w2 = ColumnParallelLinear(
            hidden_size,
            output_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


class DynamicShortConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        generator_input_size: int,
        generator_reduction: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        generator_hidden_size = hidden_size // generator_reduction

        self.kernel_generator = DynamicShortConvolutionKernelGenerator(
            input_size=generator_input_size,
            hidden_size=generator_hidden_size,
            output_size=hidden_size * kernel_size,
            quant_config=quant_config,
            prefix=add_prefix("kernel_generator", prefix),
        )

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

    def forward(
        self,
        x: torch.Tensor,  # (cu_seq_len, hidden_size)
        *,
        conv_state: torch.Tensor,  # (batch_size, hidden_size, kernel_size - 1)
        generator_input: torch.Tensor,  # (cu_seq_len, generator_input_size)
        seq_lens: torch.Tensor,  # (batch_size,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (cu_seq_len, hidden_size)
            conv_state: (batch_size, hidden_size, kernel_size - 1)
            generator_input: (cu_seq_len, generator_input_size)
            seq_lens: (batch_size,)

        Returns:
            out: (cu_seq_len, hidden_size)
            conv_state: (batch_size, hidden_size, kernel_size - 1)
        """

        x_seqs = self._continuous_to_seqs(x, seq_lens=seq_lens)
        conv_state = einops.rearrange(conv_state, "b d k -> b k d")
        x_seqs = [torch.cat([conv_state[i], x_seqs[i]]) for i in range(len(x_seqs))]
        x = self._seqs_to_batch(
            x_seqs
        )  # (batch_size, max_seq_len + kernel_size - 1, hidden_size)

        x = einops.rearrange(x, "b l d -> b d l")

        new_conv_state = x[
            :, :, -(self.kernel_size - 1) :
        ]  # (batch_size, hidden_size, kernel_size - 1)

        x = x.unfold(
            dimension=-1, size=self.kernel_size, step=1
        )  # (batch_size, hidden_size, max_seq_len, kernel_size)
        x = einops.rearrange(x, "b d l k -> b l d k")

        kernels = self.kernel_generator(
            generator_input
        )  # (cu_seq_len, hidden_size * kernel_size)
        kernels = einops.rearrange(
            kernels,
            "l (d k) -> l d k",
            d=self.hidden_size,
            k=self.kernel_size,
        )
        kernels = self._seqs_to_batch(
            self._continuous_to_seqs(kernels, seq_lens=seq_lens)
        )  # (batch_size, max_seq_len, hidden_size, kernel_size)

        out = (x * kernels).sum(dim=-1)  # (batch_size, max_seq_len, hidden_size)

        out = self._batch_to_continuous(
            out, seq_lens=seq_lens
        )  # (cu_seq_len, hidden_size)

        out = nn.functional.silu(out)

        return out, new_conv_state

    def _batch_to_continuous(
        self,
        x: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([x[i, -seq_lens[i] :] for i in range(seq_lens.size(0))])

    def _continuous_to_seqs(
        self,
        x: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            x[(seq_lens[:i].sum()) : (seq_lens[: i + 1].sum())]
            for i in range(seq_lens.size(0))
        ]

    def _seqs_to_batch(
        self,
        seqs: list[torch.Tensor],
    ) -> torch.Tensor:
        return nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_side="left",
        )


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
        assert isinstance(forward_batch.attn_backend, HybridLinearAttnBackend)
        assert isinstance(
            forward_batch.attn_backend.linear_attn_backend, MambaAttnBackendBase
        )
        linear_attn_backend = forward_batch.attn_backend.linear_attn_backend
        forward_metadata = linear_attn_backend.forward_metadata
        layer_cache = linear_attn_backend.req_to_token_pool.mamba2_layer_cache(
            self.layer_id
        )

        qkvabz, _ = self.qkvabz_proj(hidden_states)
        q, k, v, a, beta, z = qkvabz.split(
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
        q = einops.rearrange(q, "l (h d) -> l h d", h=self.num_heads, d=self.head_k_dim)

        k = nn.functional.silu(k)
        k = einops.rearrange(k, "l (h d) -> l h d", h=self.num_heads, d=self.head_k_dim)

        conv_cache = layer_cache.conv
        assert isinstance(conv_cache, torch.Tensor)
        v, new_conv_state = self.dynamic_conv1d(
            v,
            conv_state=conv_cache[
                forward_metadata.mamba_cache_indices, -self.total_v_dim :, :
            ],
            generator_input=hidden_states,
            seq_lens=(
                forward_batch.extend_seq_lens
                if forward_batch.extend_seq_lens is not None
                else torch.ones(
                    (forward_batch.batch_size,),
                    dtype=torch.long,
                )
            ),
        )
        conv_cache[forward_metadata.mamba_cache_indices, -self.total_v_dim :, :] = (
            new_conv_state
        )
        v = einops.rearrange(v, "l (h d) -> l h d", h=self.num_heads, d=self.head_v_dim)

        g = -self.A_log.float().exp() * nn.functional.softplus(a.float() + self.dt_bias)

        beta = nn.functional.sigmoid(beta)

        o = fused_recurrent_gated_delta_rule_update(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state_source=layer_cache.temporal,
            initial_state_indices=forward_metadata.mamba_cache_indices,
            cu_seqlens=cast(torch.LongTensor, forward_metadata.query_start_loc),
            use_qk_l2norm_in_kernel=True,
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

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
    ) -> EmbeddingPoolerOutput | LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

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


EntryClass = JetNemotronForCausalLM
