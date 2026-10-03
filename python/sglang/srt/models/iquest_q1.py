import logging
from typing import Iterable, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import PretrainedConfig
from triton.language.extra import libdevice

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layer_boundary import (
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    UNFUSED_NORM_READOUT,
    UnfusedNormReadout,
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import post_experts_all_reduce
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["num_tokens"])
def _apply_learned_sink_kernel(
    query,
    sink_key,
    attn_output,
    lse,
    result,
    num_tokens,
    query_stride_t: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_d: tl.constexpr,
    sink_stride_h: tl.constexpr,
    sink_stride_d: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_d: tl.constexpr,
    lse_stride_t: tl.constexpr,
    lse_stride_h: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    tokens = rows // NUM_HEADS
    heads = rows % NUM_HEADS
    dims = tl.arange(0, BLOCK_D)
    mask = (tokens[:, None] < num_tokens) & (dims[None, :] < HEAD_DIM)
    q = tl.load(
        query
        + tokens[:, None] * query_stride_t
        + heads[:, None] * query_stride_h
        + dims[None, :] * query_stride_d,
        mask,
        0,
    )
    sink = tl.load(
        sink_key
        + (heads[:, None] // QUERIES_PER_KV) * sink_stride_h
        + dims[None, :] * sink_stride_d,
        mask,
        0,
    ).to(q.dtype)
    products = q.to(tl.float32) * sink.to(tl.float32)
    if HEAD_DIM == 128:
        # Keep the reference reduction grouping to limit FP32 rounding drift.
        even, odd = tl.split(products.reshape(BLOCK_ROWS, HEAD_DIM // 2, 2))
        first, third = tl.split(even.reshape(BLOCK_ROWS, HEAD_DIM // 4, 2))
        second, fourth = tl.split(odd.reshape(BLOCK_ROWS, HEAD_DIM // 4, 2))
        sink_dot = tl.sum(((first + second) + third) + fourth, axis=1)
    else:
        sink_dot = tl.sum(products, axis=1)
    sink_logit = sink_dot * SCALE
    normal_lse = tl.load(
        lse + tokens * lse_stride_t + heads * lse_stride_h,
        tokens < num_tokens,
        0,
    ).to(tl.float32)
    factor = tl.div_rn(1.0, 1.0 + libdevice.exp(sink_logit - normal_lse))
    values = tl.load(
        attn_output
        + tokens[:, None] * output_stride_t
        + heads[:, None] * output_stride_h
        + dims[None, :] * output_stride_d,
        mask,
        0,
    ).to(tl.float32)
    tl.store(
        result + rows[:, None] * HEAD_DIM + dims[None, :],
        values * factor[:, None],
        mask,
    )


def _apply_learned_sink(
    query: torch.Tensor,
    sink_key: torch.Tensor,
    attn_output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = query.shape
    assert attn_output.shape == query.shape
    assert sink_key.ndim == 2 and sink_key.shape[1] == head_dim
    assert num_heads % sink_key.shape[0] == 0
    assert lse.shape == (num_tokens, num_heads)
    result = torch.empty(query.shape, dtype=attn_output.dtype, device=query.device)
    if num_tokens == 0:
        return result
    # Larger tiles reduce prefill overhead but cost more on short decode batches.
    block_rows = 64 if num_tokens * num_heads >= 32768 else 4
    _apply_learned_sink_kernel[(triton.cdiv(num_tokens * num_heads, block_rows),)](
        query,
        sink_key,
        attn_output,
        lse,
        result,
        num_tokens,
        *query.stride(),
        *sink_key.stride(),
        *attn_output.stride(),
        *lse.stride(),
        NUM_HEADS=num_heads,
        QUERIES_PER_KV=num_heads // sink_key.shape[0],
        HEAD_DIM=head_dim,
        SCALE=scale,
        BLOCK_ROWS=block_rows,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        # Avoid fused multiply-add rounding differences from the reference.
        enable_fp_fusion=False,
    )
    return result


class IQuestQ1RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The draft passes FP32 residuals with BF16 weights, which the fused CUDA
        # RMSNorm rejects; it also rounds BF16 inputs differently.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        result = (self.weight * hidden_states).to(input_dtype)
        return result

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IQuestQ1MoEBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ):
        super().__init__()
        self.reduce_results = reduce_results
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=True,
            layer_id=layer_id,
        )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=self.top_k,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states.float())
        topk_output = self.topk(hidden_states, router_logits)
        output = self.experts(hidden_states, topk_output)
        if self.reduce_results:
            output = post_experts_all_reduce(output)
        return output.reshape(original_shape)


class IQuestQ1DenseMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.dense_intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.dense_intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class IQuestQ1Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ):
        super().__init__()
        parallel = get_parallel()
        attn_tp_size = parallel.attn_tp_size
        attn_tp_rank = parallel.attn_tp_rank
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % attn_tp_size == 0
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = (
            self.head_dim**-0.5
            if config.softmax_scale is None
            else config.softmax_scale
        )
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = IQuestQ1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = IQuestQ1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.is_sliding = (
            config.layer_types[layer_id] == "sliding_attention"
            and config.sliding_window is not None
        )
        rope_theta = (
            config.swa_rope_theta
            if self.is_sliding
            else config.rope_parameters["rope_theta"]
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=int(
                    self.head_dim
                    * config.rope_parameters.get("partial_rotary_factor", 1.0)
                ),
                max_position=config.max_position_embeddings,
                base=rope_theta,
                rope_scaling=config.rope_parameters,
                is_neox_style=True,
            )
            if layer_id not in config.no_rope_layers
            else None
        )
        sliding_window_size = config.sliding_window - 1 if self.is_sliding else -1
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=sliding_window_size,
        )
        self.enable_sink_attention = config.enable_sink_attention
        if self.enable_sink_attention:
            self.sink_k = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.head_dim), requires_grad=False
            )
            set_weight_attrs(self.sink_k, {"weight_loader": self._sink_k_loader})

    def _sink_k_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(0)
        if self.total_num_kv_heads >= self.attn_tp_size:
            start = self.attn_tp_rank * self.num_kv_heads
        else:
            replicas = self.attn_tp_size // self.total_num_kv_heads
            start = self.attn_tp_rank // replicas
        local_weight = loaded_weight[start : start + self.num_kv_heads]
        assert local_weight.shape == param.shape, (
            f"Unexpected sink_k shard shape {tuple(local_weight.shape)} for "
            f"rank-local parameter {tuple(param.shape)}"
        )
        param.data.copy_(local_weight)

    def _apply_zero_value_sink(
        self, q: torch.Tensor, attn_output: torch.Tensor, lse: torch.Tensor
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.head_dim)
        return _apply_learned_sink(
            q,
            self.sink_k,
            attn_output.view(q.shape),
            lse,
            self.scaling,
        ).view(-1, self.q_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        if self.enable_sink_attention:
            attn_output, lse = self.attn(q, k, v, forward_batch, return_lse=True)
            attn_output = self._apply_zero_value_sink(q, attn_output, lse)
        else:
            attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class _NormReplacesResidualRead(UnfusedNormReadout):
    """After the update, the norm of the stream is both the stage's input and
    the new stream: IQuest-Q1 layers after the first add their attention
    output to the normalized input."""

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        hidden_states, _ = super().read(
            residual, norm, quant_format, post_residual_addition
        )
        return hidden_states, hidden_states


_NORM_REPLACES_RESIDUAL = _NormReplacesResidualRead()


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    return layer_id not in (config.mlp_only_layers or [])


class IQuestQ1DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = IQuestQ1Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            reduce_results=False,
        )
        if not _is_moe_layer(config, layer_id):
            self.mlp = IQuestQ1DenseMLP(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,
            )
        else:
            self.mlp = IQuestQ1MoEBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,
            )
        self.is_first_layer = layer_id == 0
        if self.is_first_layer:
            self.ffn_out_norm = IQuestQ1RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.attn_out_scale = config.first_layer_attn_out_scale
            self.ffn_out_scale = config.first_layer_ffn_out_scale
        else:
            self.attn_out_scale = config.attn_out_scale
            self.ffn_out_scale = config.ffn_out_scale
        # Intentional: layer 0 adds the raw input and an FFN output norm, later layers
        # add the normalized input without one; the MTP draft follows layer 0.
        moe = _is_moe_layer(config, layer_id)
        self.attn_boundary, self.ffn_boundary = make_stages(
            (
                declare_attn(
                    read=UNFUSED_NORM_READOUT
                    if self.is_first_layer
                    else _NORM_REPLACES_RESIDUAL,
                    output_transform=OutputTransform(self._attn_output),
                ),
                self.attention_norm,
            ),
            (
                declare_ffn(
                    sparse=moe,
                    next_layer_sparse=_is_moe_layer(config, layer_id + 1),
                    read=UNFUSED_NORM_READOUT,
                    output_transform=OutputTransform(self._ffn_output),
                ),
                self.feed_forward_norm,
            ),
            previous=declare_ffn(
                sparse=_is_moe_layer(config, layer_id - 1), next_layer_sparse=moe
            )
            if layer_id != 0
            else None,
            terminal=layer_id == config.num_hidden_layers - 1,
        )

    def _attn_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.attn_out_norm(attn_output) * self.attn_out_scale

    def _ffn_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        if self.is_first_layer:
            return self.ffn_out_norm(mlp_output) * self.ffn_out_scale
        return mlp_output * self.ffn_out_scale

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        attn_output = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.attn_boundary.finish(attn_output, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        mlp_output = self.mlp(hidden_states)
        return self.ffn_boundary.finish(mlp_output, forward_batch)


class IQuestQ1Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
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
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: IQuestQ1DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states = residual_batch.fold(hidden_states, forward_batch)
        return self.norm(residual_batch.take_output(hidden_states, forward_batch))


_FUSED_SHARD_LAYOUT = (
    ("qkv_proj", ("q", "k", "v")),
    ("gate_up_proj", (0, 1)),
)


def _is_created_by_quantization(
    name: str, param: torch.nn.Parameter, quant_config: Optional[QuantizationConfig]
) -> bool:
    if getattr(param, "_skip_weight_check", False):
        return True
    return (
        isinstance(quant_config, Fp8Config)
        and not quant_config.is_checkpoint_fp8_serialized
        and name.rsplit(".", 1)[-1]
        in {
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        }
    )


def check_all_params_loaded(
    params_dict: dict,
    loaded: set,
    model: str,
    loaded_shards: Optional[dict] = None,
    quant_config: Optional[QuantizationConfig] = None,
) -> None:
    missing = sorted(
        name
        for name, param in params_dict.items()
        if name not in loaded
        and not _is_created_by_quantization(name, param, quant_config)
    )
    partial = []
    for name, shards in sorted((loaded_shards or {}).items()):
        for projection, expected in _FUSED_SHARD_LAYOUT:
            if not name.endswith(f".{projection}.weight"):
                continue
            absent = [shard for shard in expected if shard not in shards]
            if absent:
                partial.append(f"{name} without {absent}")
    if not missing and not partial:
        return
    problems = []
    if missing:
        problems.append(
            f"{len(missing)} parameters were not found in the checkpoint, "
            f"e.g. {missing[:8]}"
        )
    if partial:
        problems.append(
            f"{len(partial)} fused parameters were only partially loaded, "
            f"e.g. {partial[:8]}"
        )
    message = f"{model}: " + "; ".join(problems)
    if quant_config is not None and not partial:
        logger.warning(
            "%s; verify whether these parameters are generated by quantization.",
            message,
        )
        return
    raise ValueError(message)


def load_iquest_q1_weights(
    model: nn.Module, weights: Iterable[Tuple[str, torch.Tensor]]
):
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(model.named_parameters())
    expert_params_mapping = None
    loaded = set()
    loaded_shards = {}
    for name, loaded_weight in weights:
        if (
            name == "lm_head.weight"
            and name not in params_dict
            and "model.embed_tokens.weight" in params_dict
        ):
            name = "model.embed_tokens.weight"
        if "rotary_emb.inv_freq" in name:
            continue
        if name.endswith("experts.fc"):
            param_name = name.replace("experts.fc", "experts.w13_weight")
            if param_name not in params_dict:
                raise ValueError(f"Unexpected IQuest Q1 weight: {name}")
            param = params_dict[param_name]
            loaded.add(param_name)
            for expert_id in range(model.config.num_experts):
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight[expert_id][: model.config.intermediate_size],
                    param_name,
                    shard_id="w1",
                    expert_id=expert_id,
                )
                weight_loader(
                    param,
                    loaded_weight[expert_id][model.config.intermediate_size :],
                    param_name,
                    shard_id="w3",
                    expert_id=expert_id,
                )
            continue
        if name.endswith("experts.proj"):
            param_name = name.replace("experts.proj", "experts.w2_weight")
            if param_name not in params_dict:
                raise ValueError(f"Unexpected IQuest Q1 weight: {name}")
            param = params_dict[param_name]
            loaded.add(param_name)
            for expert_id in range(model.config.num_experts):
                param.weight_loader(
                    param,
                    loaded_weight[expert_id],
                    param_name,
                    shard_id="w2",
                    expert_id=expert_id,
                )
            continue
        if ".experts." in name:
            if expert_params_mapping is None:
                expert_params_mapping = FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name="gate_proj",
                    ckpt_down_proj_name="down_proj",
                    ckpt_up_proj_name="up_proj",
                    num_experts=model.config.num_experts,
                )
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    raise ValueError(f"Unexpected IQuest Q1 weight: {name}")
                param = params_dict[mapped_name]
                param.weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded.add(mapped_name)
                break
            else:
                raise ValueError(f"Unexpected IQuest Q1 weight: {name}")
            continue
        if "router.weight" in name:
            name = name.replace("router.weight", "gate.weight")
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            mapped_name = name.replace(weight_name, param_name)
            if mapped_name not in params_dict:
                continue
            param = params_dict[mapped_name]
            param.weight_loader(param, loaded_weight, shard_id)
            if mapped_name not in loaded or mapped_name in loaded_shards:
                loaded_shards.setdefault(mapped_name, set()).add(shard_id)
            loaded.add(mapped_name)
            break
        else:
            if name not in params_dict:
                raise ValueError(f"Unexpected IQuest Q1 weight: {name}")
            param = params_dict[name]
            getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
            loaded.add(name)
            loaded_shards.pop(name, None)
    check_all_params_loaded(
        params_dict,
        loaded,
        type(model).__name__,
        loaded_shards=loaded_shards,
        quant_config=model.quant_config,
    )


class IQuestQ1ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = IQuestQ1Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = (
            self.model.embed_tokens
            if config.tie_word_embeddings
            else ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        )
        self.logits_processor = LogitsProcessor(config, logit_scale=config.logit_scale)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        load_iquest_q1_weights(
            self,
            (
                (name, weight)
                for name, weight in weights
                if not name.startswith("mtp_layers.")
            ),
        )


EntryClass = [IQuestQ1ForCausalLM]
