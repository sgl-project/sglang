from collections.abc import Iterable
from typing import cast

import einops
import torch
import torch.nn as nn

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
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
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model
from sglang.srt.runtime_context import mamba_cache_chunk_size
from sglang.srt.utils import add_prefix, is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()

if _is_npu:
    import torch_npu


class JetRMSNormGated(RMSNormGated):
    def forward(self, x, z=None):
        if not _is_npu:
            return super().forward(x, z)
        # The Triton gated-norm tile exceeds the 910B4 UB for Jet's heads.
        # Keep normalization and gating in fp32, then round once at the end.
        out = torch_npu.npu_rms_norm(x.float(), self.weight.float(), self.eps)[0]
        if z is not None:
            out = out * nn.functional.silu(z.float())
        return out.to(x.dtype)


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
        checkpoint_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (cu_seq_len, hidden_size)
            conv_state: (batch_size, hidden_size, kernel_size - 1)
            generator_input: (cu_seq_len, generator_input_size)
            seq_lens: (batch_size,)
            checkpoint_lens: optional consumed-token counts to snapshot per sequence

        Returns:
            out: (cu_seq_len, hidden_size)
            conv_state: (batch_size, hidden_size, kernel_size - 1)
            checkpoint_state: same shape as conv_state, or None
        """
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x, conv_state, conv_state if checkpoint_lens is not None else None

        # Ragged windows stay on device: Python slices with device scalars
        # synchronize the forward stream and stall overlapped scheduling.
        seq_lens = seq_lens.to(device=x.device, dtype=torch.long)
        batch_size = seq_lens.numel()
        window = self.kernel_size - 1
        starts = seq_lens.cumsum(0) - seq_lens
        seq_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=x.device),
            seq_lens,
            output_size=num_tokens,
        )
        history = conv_state.transpose(1, 2).reshape(-1, self.hidden_size)
        values = torch.cat((history, x), dim=0)
        history_size = batch_size * window
        token_offsets = torch.arange(num_tokens, device=x.device) - starts[seq_ids]
        positions = token_offsets[:, None] + torch.arange(-window, 1, device=x.device)
        indices = torch.where(
            positions < 0,
            seq_ids[:, None] * window + positions + window,
            history_size + starts[seq_ids, None] + positions,
        )
        inputs = values[indices].transpose(1, 2)
        kernels = self.kernel_generator(generator_input).reshape(
            num_tokens, self.hidden_size, self.kernel_size
        )
        out = nn.functional.silu((inputs * kernels).sum(dim=-1))

        def state_at(lengths):
            positions = lengths[:, None] + torch.arange(-window, 0, device=x.device)
            indices = torch.where(
                positions < 0,
                torch.arange(batch_size, device=x.device)[:, None] * window
                + positions
                + window,
                history_size + starts[:, None] + positions,
            )
            return values[indices].transpose(1, 2)

        new_conv_state = state_at(seq_lens)
        checkpoint_state = None
        if checkpoint_lens is not None:
            # Untracked rows may carry -1; they are masked out by the caller.
            checkpoint_lens = checkpoint_lens.clamp(min=0).minimum(seq_lens)
            checkpoint_state = state_at(checkpoint_lens)
        return out, new_conv_state, checkpoint_state


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

        self.o_norm = JetRMSNormGated(
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
        assert isinstance(get_attn_backend(), HybridLinearAttnBackend)
        assert isinstance(get_attn_backend().linear_attn_backend, MambaAttnBackendBase)
        linear_attn_backend = get_attn_backend().linear_attn_backend
        forward_metadata = linear_attn_backend.forward_metadata
        layer_cache = linear_attn_backend.state_layer_cache(self.layer_id)

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

        conv_cache = layer_cache.conv[0]
        assert isinstance(conv_cache, torch.Tensor)
        # Normalize window-first caches to a channel-first view for reads and writes.
        if conv_cache.shape[-2] == self.conv_size - 1:
            conv_cache = conv_cache.transpose(1, 2)

        checkpoint_lens = None
        checkpoint_indices = None
        if (
            forward_batch.forward_mode.is_extend()
            and forward_metadata.has_mamba_track_mask
        ):
            chunk_size = mamba_cache_chunk_size()
            checkpoint_lens = (
                (forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens)
                // chunk_size
                * chunk_size
            )
            checkpoint_lens = torch.where(
                forward_batch.mamba_track_mask, checkpoint_lens, -1
            )
            checkpoint_indices = forward_metadata.mamba_track_indices

        v, new_conv_state, checkpoint_conv_state = self.dynamic_conv1d(
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
                    device=hidden_states.device,
                )
            ),
            checkpoint_lens=checkpoint_lens,
        )
        conv_cache[forward_metadata.mamba_cache_indices, -self.total_v_dim :, :] = (
            new_conv_state
        )
        if checkpoint_conv_state is not None:
            rows = forward_metadata.mamba_track_mask_indices
            conv_cache[
                forward_metadata.conv_states_mask_indices, -self.total_v_dim :, :
            ] = checkpoint_conv_state[rows]
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
            checkpoint_seq_lens=checkpoint_lens,
            checkpoint_state_indices=checkpoint_indices,
        ).squeeze(0)

        if forward_batch.forward_mode.is_decode():
            linear_attn_backend._track_mamba_state_decode(
                forward_batch,
                layer_cache.conv[0],
                layer_cache.temporal,
                forward_metadata.mamba_cache_indices,
                layer_id=self.layer_id,
            )

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

        rope_theta, rope_scaling = get_rope_config(self.config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.config.max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
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
        start_layer: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.start_layer = start_layer

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

    @staticmethod
    def stage_facts(config: JetNemotronConfig, layer_id: int):
        """None: the layer adds its own residual, so it declares no stages.
        The model's shared declaration function (see make_layers)."""
        return ()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
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

        return hidden_states


class JetNemotronModel(Qwen2Model):
    """Each decoder layer adds its own residual, so the stack runs without a
    residual stream and ends with the final norm alone."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](positions, hidden_states, forward_batch)
        if hidden_states.shape[0] != 0:
            hidden_states = self.norm(hidden_states)
        return hidden_states


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

        self.model = JetNemotronModel(
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

    def get_attention_sliding_window_size(self) -> int | None:
        swa = self.config.efficient_attention_config.get("swa")
        if swa is None or "swa" not in self.config.layer_types:
            return None
        return swa["window_size"]

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


EntryClass = JetNemotronForCausalLM
