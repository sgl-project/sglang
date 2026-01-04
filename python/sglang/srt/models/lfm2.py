import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
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
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu

logger = logging.getLogger(__name__)


class Lfm2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: float | None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id=0,
    ) -> None:
        super().__init__()

        if auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)

            if ffn_dim_multiplier is not None:
                intermediate_size = int(ffn_dim_multiplier * intermediate_size)
            intermediate_size = multiple_of * (
                (intermediate_size + multiple_of - 1) // multiple_of
            )

        self.layer_id = layer_id

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
        )
        return x


class Lfm2Attention(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads

        self.max_position_embeddings = max_position_embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rotary_dim=self.head_dim,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.rope_theta = rope_theta

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        n_tokens, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(n_tokens, self.num_heads, self.head_dim).contiguous()
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim).contiguous()

        q = q.view(-1, self.head_dim)
        q = self.q_layernorm(q)
        q = q.view(n_tokens, self.num_heads, self.head_dim)

        k = k.view(-1, self.head_dim)
        k = self.k_layernorm(k)
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(n_tokens, self.num_heads * self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads * self.head_dim)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


def apply_mask_to_padding_states(
    x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if attention_mask is not None:
        if attention_mask.dim() == 2:

            x = x * attention_mask.unsqueeze(-1)
    return x


class Lfm2ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias
        self.layer_id = layer_id

        self.in_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj" if prefix else "in_proj",
        )

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
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

        if hasattr(forward_batch, "attention_mask"):
            hidden_states = apply_mask_to_padding_states(
                hidden_states, forward_batch.attention_mask
            )

        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.split(
            [self.hidden_size, self.hidden_size, self.hidden_size], dim=-1
        )

        B = B.unsqueeze(-1)
        C = C.unsqueeze(-1)
        x = x.unsqueeze(-1)

        Bx = B * x
        Bx = Bx.transpose(1, 2)

        conv_cache = layer_cache.conv[0]
        assert isinstance(conv_cache, torch.Tensor)

        is_decode = (
            forward_batch.extend_seq_lens is None
            or (forward_batch.extend_seq_lens == 1).all()
        )

        if is_decode:

            conv_weights = self.conv.weight.view(
                self.conv.weight.size(0), self.conv.weight.size(2)
            )
            batch_conv_state = conv_cache[
                forward_metadata.mamba_cache_indices, : self.hidden_size, :
            ]
            current_input = Bx.squeeze(1)
            conv_out = causal_conv1d_update(
                current_input,
                batch_conv_state,
                conv_weights,
                self.conv.bias if self.bias else None,
                None,
            )
            conv_cache[forward_metadata.mamba_cache_indices, : self.hidden_size, :] = (
                batch_conv_state
            )
            conv_out = conv_out.unsqueeze(1)
        else:
            conv_weights = self.conv.weight.view(
                self.conv.weight.size(0), self.conv.weight.size(2)
            )
            Bx_for_conv = Bx.squeeze(1)
            seq_lens = forward_batch.extend_seq_lens
            split_Bx = torch.split(Bx_for_conv, seq_lens.tolist())

            for i, seq_bx in enumerate(split_Bx):
                cache_idx = forward_metadata.mamba_cache_indices[i]
                seq_len = seq_bx.shape[0]
                if seq_len >= self.kernel_size - 1:
                    new_conv_state = seq_bx[-(self.kernel_size - 1) :, :]
                else:
                    padding = torch.zeros(
                        self.kernel_size - 1 - seq_len,
                        self.hidden_size,
                        dtype=seq_bx.dtype,
                        device=seq_bx.device,
                    )
                    new_conv_state = torch.cat([padding, seq_bx], dim=0)

                conv_cache[cache_idx, : self.hidden_size, :] = new_conv_state.t()

            conv_outs = []
            for seq_bx in split_Bx:
                seq_bx_conv = seq_bx.unsqueeze(0).transpose(1, 2)
                conv_out_seq = causal_conv1d_fn(
                    seq_bx_conv,
                    conv_weights,
                    self.conv.bias if self.bias else None,
                    activation=None,
                )

                conv_outs.append(conv_out_seq.squeeze(0).transpose(0, 1))
            conv_out = torch.cat(conv_outs, dim=0)
            conv_out = conv_out.unsqueeze(1)

        conv_out = conv_out.transpose(1, 2)
        y = C * conv_out
        y = y.squeeze(-1)
        y, _ = self.out_proj(y)

        return y


class Lfm2AttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.prefix = prefix
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.block_dim

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)

        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )

        max_position_embeddings = getattr(config, "max_position_embeddings", 128000)
        self.self_attn = Lfm2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.feed_forward = Lfm2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.block_ff_dim,
            multiple_of=config.block_multiple_of,
            auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=config.block_ffn_dim_multiplier,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
            layer_id=layer_id,
        )

        self.operator_norm = RMSNorm(self.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        return self.feed_forward(hidden_states), residual


class Lfm2ShortConvDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.block_dim
        self.conv = Lfm2ShortConv(
            hidden_size=self.hidden_size,
            kernel_size=config.conv_L_cache,
            bias=config.conv_bias,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=f"{prefix}.conv",
        )

        self.feed_forward = Lfm2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.block_ff_dim,
            multiple_of=config.block_multiple_of,
            auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=config.block_ffn_dim_multiplier,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
            layer_id=layer_id,
        )

        self.operator_norm = RMSNorm(self.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        output = self.conv(
            hidden_states,
            forward_batch,
        )

        hidden_states, residual = self.ffn_norm(output, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class Lfm2Model(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(idx, prefix):
            layer_type = config.layer_types[idx]
            if layer_type == "full_attention":
                return Lfm2AttentionDecoderLayer(
                    config=config,
                    layer_id=idx,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            else:
                return Lfm2ShortConvDecoderLayer(
                    config=config,
                    layer_id=idx,
                    quant_config=quant_config,
                    prefix=prefix,
                )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:

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
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                if residual is not None:
                    aux_hidden_states.append(hidden_states + residual)
                else:
                    aux_hidden_states.append(hidden_states)

            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.embedding_norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class Lfm2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))

        if self.pp_group.is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = None

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def _init_model(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> Lfm2Model:
        return Lfm2Model(config, quant_config=quant_config, prefix=prefix)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
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
                ret = self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:

                ret = hidden_states
        else:
            ret = hidden_states

        return ret

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".w1", 0),
            (".gate_up_proj", ".w3", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            layer_id = self._get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if ".mlp." in name:
                name = name.replace(".mlp.", ".feed_forward.")

            if "self_attn.out_proj" in name:
                name = name.replace("self_attn.out_proj", "self_attn.o_proj")

            if "feed_forward.w2" in name:
                name = name.replace("feed_forward.w2", "feed_forward.down_proj")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def _get_layer_id(self, name: str) -> Optional[int]:
        if "layers" not in name:
            return None

        parts = name.split(".")
        try:
            layers_idx = parts.index("layers")
            if layers_idx + 1 < len(parts):
                return int(parts[layers_idx + 1])
        except (ValueError, IndexError):
            return None

        return None


EntryClass = [Lfm2ForCausalLM]
