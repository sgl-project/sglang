"""
LFM2 (Liquid Foundation Model 2) implementation for SGLang.

This is a hybrid architecture with both attention and short conv layers.
- Attention layers use standard KV cache (RadixAttention)
- Conv layers use MambaPool for state caching (via HybridReqToTokenPool)

The model uses a gated 1D causal convolution (kernel=3) instead of attention
in some layers, providing linear memory complexity for those layers.

Uses optimized causal_conv1d kernels from the mamba package for fast inference.
"""

import logging
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.mamba.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_update as causal_conv1d_update_triton,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
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
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.short_conv_backend import (
        ShortConvMetadata,
    )

logger = logging.getLogger(__name__)


class Lfm2MLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        intermediate_size = config.intermediate_size

        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(
                    config.block_ffn_dim_multiplier * intermediate_size
                )
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1)
                    // config.block_multiple_of
                )

        self.w1 = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )
        self.w3 = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w3", prefix),
        )
        self.w2 = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.w1(x)
        up, _ = self.w3(x)
        out, _ = self.w2(F.silu(gate) * up)
        return out


class Lfm2Attention(nn.Module):
    """Grouped-query attention with RoPE and Q/K layernorm."""

    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.total_num_heads
        )
        self.scaling = self.head_dim**-0.5

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        else:
            rope_theta = getattr(config, "rope_theta", 1000000.0)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_scaling=rope_parameters or getattr(config, "rope_scaling", None),
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.num_local_q_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads

        self.attn = RadixAttention(
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.reshape(T, self.num_local_q_heads, self.head_dim)
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim)

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_q_heads, self.head_dim
        )
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch)
        out, _ = self.out_proj(attn_out)
        return out


def register_shortconv_verify_buffers(conv: nn.Module) -> None:
    """Pre-size the per-module tape slot-index buffer used by TARGET_VERIFY.

    Shared by Lfm2ShortConv and Lfm2MoeShortConv (lfm2_moe.py imports it).
    """
    conv.register_buffer(
        "_intermediate_state_indices",
        torch.arange(256, dtype=torch.int32),
        persistent=False,
    )


def shortconv_target_verify(
    conv: nn.Module,
    Bx: torch.Tensor,
    meta: "ShortConvMetadata",
    draft_token_num: int,
    arch: str,
) -> torch.Tensor:
    """Depthwise short conv over a DFlash TARGET_VERIFY block.

    Shared by Lfm2ShortConv and Lfm2MoeShortConv (lfm2_moe.py imports it).
    Runs the triton update kernel over the [bs, block] layout, recording the
    per-step conv windows into the speculative tape
    (``MambaPool.SpeculativeState.intermediate_conv_window``) so the conv
    state can roll back to the accept boundary after verification.
    """
    assert isinstance(meta.layer_cache, MambaPool.SpeculativeState), (
        f"{arch} TARGET_VERIFY requires --mamba-radix-cache-strategy extra_buffer."
    )
    bs = meta.cache_indices.shape[0]
    Bx_reshaped = Bx.view(bs, draft_token_num, -1).transpose(1, 2)
    if conv._intermediate_state_indices.shape[0] < bs:
        conv._intermediate_state_indices = torch.arange(
            bs, dtype=torch.int32, device=Bx.device
        )
    conv_out = causal_conv1d_update_triton(
        Bx_reshaped,
        meta.layer_cache.conv[0],
        conv.conv_weight,
        conv.conv_bias,
        activation=None,
        conv_state_indices=meta.cache_indices,
        intermediate_conv_window=meta.layer_cache.intermediate_conv_window[0],
        intermediate_state_indices=conv._intermediate_state_indices[:bs],
    )
    return conv_out.transpose(1, 2).reshape(bs * draft_token_num, -1)


class Lfm2ShortConv(nn.Module):
    """
    Gated short convolution layer using optimized causal_conv1d kernels.

    Architecture: in_proj -> split(B, C, x) -> Bx -> conv1d -> C*conv_out -> out_proj
    - Uses double gating: B (before conv) and C (after conv)
    - Fixed-size cache: stores last (kernel_size - 1) tokens
    - Uses causal_conv1d_fn for prefill and causal_conv1d_update for decode
    - Supports tensor parallelism: hidden dimension is sharded across TP ranks
    """

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel = int(config.conv_L_cache)
        self.use_bias = bool(config.conv_bias)
        self.hidden_size = config.hidden_size

        tp_size = get_parallel().tp_size
        self.hidden_size_per_partition = self.hidden_size // tp_size

        # Use MergedColumnParallelLinear so each output (B, C, x) is sharded separately
        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,  # B, C, x each get hidden_size
            bias=self.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # Conv weights sharded along hidden dimension: (hidden_size/tp, kernel_size)
        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": sharded_weight_loader(0)})
        if self.use_bias:
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition))
            set_weight_attrs(
                self.conv_bias, {"weight_loader": sharded_weight_loader(0)}
            )
        else:
            self.register_parameter("conv_bias", None)

        register_shortconv_verify_buffers(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        # The backend owns the per-request conv-state plumbing (slot indices,
        # prefix mask, cu-seqlens, cuda-graph buffers); this layer just runs its
        # depthwise conv against the returned handle.
        meta = get_attn_backend().conv_state_metadata(self.layer_idx, forward_batch)
        conv_state = meta.layer_cache.conv[0]

        # Project and split into gates: B (pre-conv), C (post-conv), x (input)
        proj, _ = self.in_proj(hidden_states)
        B_gate, C_gate, x = proj.chunk(3, dim=-1)
        Bx = B_gate * x

        if forward_batch.forward_mode.is_decode():
            # Decode: single token per request, use optimized update kernel
            conv_out = causal_conv1d_update(
                Bx,
                conv_state,
                self.conv_weight,
                self.conv_bias,
                activation=None,
                conv_state_indices=meta.cache_indices,
            )
        elif forward_batch.forward_mode.is_target_verify():
            conv_out = shortconv_target_verify(
                self, Bx, meta, forward_batch.spec_info.draft_token_num, "LFM2"
            )
        else:
            # Prefill: multiple tokens, use varlen kernel
            Bx_t = Bx.transpose(0, 1).contiguous()
            conv_out = causal_conv1d_fn(
                Bx_t,
                self.conv_weight,
                self.conv_bias,
                query_start_loc=meta.query_start_loc,
                cache_indices=meta.cache_indices,
                has_initial_state=meta.has_initial_state,
                conv_states=conv_state,
                activation=None,
            ).transpose(0, 1)

        output, _ = self.out_proj(C_gate * conv_out)
        return output


class Lfm2DecoderLayer(nn.Module):
    """Decoder layer - either attention or conv based on config."""

    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type == "full_attention"
        # Set by Lfm2Model.set_dflash_layers_to_capture for DFlash aux capture.
        self._is_layer_to_capture = False

        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.conv = Lfm2ShortConv(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )

        self.feed_forward = Lfm2MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward(
        self,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not forward_batch.forward_mode.is_idle():
            if captured_last_layer_outputs is not None:
                captured_last_layer_outputs.append(hidden_states)

            residual = hidden_states
            normed = self.operator_norm(hidden_states)

            if self.is_attention_layer:
                hidden_states = self.self_attn(positions, normed, forward_batch)
            else:
                hidden_states = self.conv(normed, forward_batch)

            hidden_states = hidden_states + residual
            hidden_states = hidden_states + self.feed_forward(
                self.ffn_norm(hidden_states)
            )

        return hidden_states, residual


class Lfm2Model(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Count attention layers for KV cache sizing
        self.num_attention_layers = sum(
            1 for lt in config.layer_types if lt == "full_attention"
        )

        def get_layer(idx: int, prefix: str, **kwargs):
            return Lfm2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers_to_capture: List[int] = []

    def set_dflash_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = list(layers_to_capture)
        for layer_id in self.layers_to_capture:
            # A tap on the final layer (layer_id == len(self.layers)) is
            # captured after the loop in forward(); only mark real layers.
            if layer_id < len(self.layers):
                self.layers[layer_id]._is_layer_to_capture = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )

        residual = None
        aux_hidden_states: List[torch.Tensor] = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                captured_last_layer_outputs=(
                    aux_hidden_states if layer._is_layer_to_capture else None
                ),
            )

        if (
            not forward_batch.forward_mode.is_idle()
            and len(self.layers) in self.layers_to_capture
        ):
            aux_hidden_states.append(hidden_states)

        hidden_states = self.embedding_norm(hidden_states)
        if not aux_hidden_states:
            return hidden_states
        return hidden_states, aux_hidden_states


class Lfm2ForCausalLM(nn.Module):
    """LFM2 for causal language modeling with hybrid attention/conv architecture."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank

        self.quant_config = quant_config
        self.model = Lfm2Model(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.num_attention_layers = self.model.num_attention_layers

    def get_num_kv_cache_layers(self) -> int:
        return self.num_attention_layers

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @property
    def capture_aux_hidden_states(self) -> bool:
        return bool(self.model.layers_to_capture)

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError("DFLASH requires explicit layer_ids for aux hidden capture.")

        # Mark layer L to capture its input, which is the output of layer L-1.
        self.model.set_dflash_layers_to_capture([val + 1 for val in layer_ids])

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        aux_hidden_states = None
        # Capture-enabled idle batches return a bare tensor (layers skip the
        # append on IDLE), so narrow on the actual return shape.
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        embed_tokens_weight = None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens.weight" in name:
                embed_tokens_weight = loaded_weight

            # Handle conv weight/bias naming: HF uses conv.conv, we use conv_weight/conv_bias
            if ".conv.conv.weight" in name:
                name = name.replace(".conv.conv.weight", ".conv.conv_weight")
                loaded_weight = loaded_weight.squeeze(1)  # (D, 1, K) -> (D, K)
            if ".conv.conv.bias" in name:
                name = name.replace(".conv.conv.bias", ".conv.conv_bias")

            # Handle QKV stacking
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Handle tied lm_head weight
        if "lm_head.weight" not in loaded_params and "lm_head.weight" in params_dict:
            if embed_tokens_weight is not None:
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, embed_tokens_weight)
                loaded_params.add("lm_head.weight")

        return loaded_params


EntryClass = [Lfm2ForCausalLM]
