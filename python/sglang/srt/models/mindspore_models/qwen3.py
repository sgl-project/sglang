# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import logging
import math
import os
from functools import lru_cache
from typing import Iterable, Optional, Tuple, Type, Union

import mindspore as ms
import numpy as np
import torch
from mindspore import Parameter, Tensor, dtype, jit, mint, mutable, nn, ops
from mindspore.ops.operations.nn_ops import (
    FlashAttentionScore,
    PagedAttention,
    ReshapeAndCache,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.mindspore_models.mindspore_model_base import MindSporeModelBase
from sglang.srt.models.mindspore_models.utils import tensor_torch2ms

logger = logging.getLogger(__name__)

Qwen3Config = None


def _get_tp_group_name():
    return get_tp_group().unique_name


class MsNativeAttnBackend(nn.Cell):
    """Paged Attention Manager."""

    def __init__(self, n_heads, head_dim, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.flash_attention = FlashAttentionScore(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            next_tokens=0,
            input_layout="TH",
        )
        self.paged_attention = PagedAttention(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            kv_head_num=self.n_kv_heads,
        )
        self.reshape_and_cache = ReshapeAndCache()

    # pylint: disable=W0613
    def construct(
        self,
        key,
        value,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        k_scale=None,
        v_scale=None,
    ):
        if k_scale is not None:
            key = key / k_scale
        if v_scale is not None:
            value = value / v_scale
        cache_out = self.reshape_and_cache(
            key, value, key_cache, value_cache, out_cache_loc
        )
        key = ops.depend(key, cache_out)

        return key

    def extend(
        self,
        query,
        key,
        value,
        attn_mask=None,
        alibi_mask=None,
        prefix=None,
        padding_mask=None,
        q_seq_lens=None,
        batch_valid_length=None,
    ):
        _, _, _, output = self.flash_attention(
            query,
            key,
            value,
            alibi_mask,
            None,
            padding_mask,
            attn_mask,
            prefix,
            q_seq_lens,
            batch_valid_length,
        )
        return output

    def decode(
        self,
        query,
        batch_valid_length,
        attn_mask=None,
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        block_tables=None,
    ):
        output = self.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            None,
            None,
            attn_mask,
            q_seq_lens,
        )
        return output


class VocabParallelEmbedding(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        # self.sequence_parallel = config.sequence_parallel
        self.tensor_parallel_group_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, tp_rank, self.tensor_parallel_group_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.weight = Parameter(
            mint.zeros(
                (self.num_embeddings_per_partition, self.embedding_dim),
                dtype=config.param_dtype,
            ),
            requires_grad=False,
        )

        tp_group_name = _get_tp_group_name()
        self.all_reduce = ops.AllReduce(group=tp_group_name)
        self.reduce_scatter_tensor = ops.ReduceScatter(group=tp_group_name)

        self.max_index_per_partition = Tensor(
            self.num_embeddings_per_partition - 1, dtype=dtype.int32
        )
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.Gather()

    def construct(self, x):
        if self.tensor_parallel_group_size > 1:
            displaced_x = mint.sub(x, self.vocab_start_index)
            down_truncated_x = mint.nn.functional.relu(displaced_x)
            truncated_x = mint.minimum(down_truncated_x, self.max_index_per_partition)
            input_mask = mint.eq(displaced_x, truncated_x)
            input_mask = self.expand_dims(input_mask, -1)
        else:
            input_mask = None
            truncated_x = x
        output_parallel = self.gather(self.weight, truncated_x, 0)
        if self.tensor_parallel_group_size > 1:
            output_parallel = mint.mul(output_parallel, input_mask)
        output = self.all_reduce(output_parallel)
        return output

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        copy_dim = 0
        shard_size = param.shape[copy_dim]
        start_idx = tp_rank * shard_size
        weight = weight.narrow(copy_dim, start_idx, shard_size).contiguous()
        param.set_data(tensor_torch2ms(weight))
        return None

    def _vocab_range_from_global_vocab_size(self, global_vocab_size, rank, world_size):
        if global_vocab_size % world_size != 0:
            raise ValueError(
                f"The vocabulary size is {global_vocab_size},"
                f"which is not divisible by size of tensor parallel({world_size})."
            )
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l


class RMSNorm(nn.Cell):
    def __init__(self, norm_dim: int, eps: float, param_dtype: Optional[Type]) -> None:
        super().__init__()

        self.weight = Parameter(mint.ones(norm_dim, dtype=param_dtype))
        self.rms_norm = ops.RmsNorm(eps)

    def construct(
        self, x: Tensor, residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
        output = self.rms_norm(x, self.weight)[0]
        if residual is None:
            return output
        return output, residual


class Qwen3ColParallelLinear(nn.Cell):
    def __init__(
        self, input_size: int, output_size: int, param_dtype: Optional[Type], bias: bool
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.param_dtype = param_dtype
        self.input_size = input_size
        self.output_size = output_size // self.tp_size
        self.enable_bias = bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros((self.output_size, self.input_size), dtype=self.param_dtype),
            requires_grad=False,
        )
        setattr(self.weight, "weight_load", self.weight_load)

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(mint.zeros(self.output_size, dtype=self.param_dtype))
            setattr(self.bias, "weight_load", self.weight_load)

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
        return x.view(*origin_shape[:-1], -1)

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        copy_dim = 0
        shard_size = param.shape[copy_dim]
        start_idx = tp_rank * shard_size
        weight = weight.narrow(copy_dim, start_idx, shard_size).contiguous()

        param.set_data(tensor_torch2ms(weight))
        return None


class Qwen3RowParallelLinear(nn.Cell):
    def __init__(
        self, input_size: int, output_size: int, param_dtype: Optional[Type], bias: bool
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.param_dtype = param_dtype
        self.input_size = input_size // self.tp_size
        self.output_size = output_size
        self.enable_bias = bias

        self.matmul = ops.MatMul(transpose_b=True)
        self.weight = Parameter(
            mint.zeros((self.output_size, self.input_size), dtype=self.param_dtype),
            requires_grad=False,
        )
        setattr(self.weight, "weight_load", self.weight_load)

        if self.enable_bias:
            self.bias_add = ops.Add()
            self.bias = Parameter(mint.zeros(self.output_size, dtype=self.param_dtype))
            setattr(self.bias, "weight_load", self.weight_load)
        tp_group_name = _get_tp_group_name()
        self.all_reduce = ops.AllReduce(group=tp_group_name)

    def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
        origin_shape = input.shape
        x = self.matmul(input.view(-1, origin_shape[-1]), self.weight)
        if self.enable_bias:
            x = self.bias_add(x, self.bias)
        x = self.all_reduce(x)
        return x.view(*origin_shape[:-1], -1)

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        copy_dim = 1
        shard_size = param.shape[copy_dim]
        start_idx = tp_rank * shard_size
        weight = weight.narrow(copy_dim, start_idx, shard_size).contiguous()

        param.set_data(tensor_torch2ms(weight))
        return None


class Qwen3MLP(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.up_proj = Qwen3ColParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            bias=False,
        )
        self.gate_proj = Qwen3ColParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            param_dtype=config.param_dtype,
            bias=False,
        )
        self.down_proj = Qwen3RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            bias=False,
        )
        self.act_fn = ops.silu

    def construct(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: np.dtype
) -> np.ndarray:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = np.clip(linear_func, 0, 1)
    return ramp_func


class InferYaRNScalingRotaryEmbedding(nn.Cell):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)

        super().__init__()

        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, scaling_factor: float) -> Tensor:
        pos_freqs = self.base ** (
            np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(
                low,
                high,
                self.rotary_dim // 2,
                dtype=np.float32,  # type: ignore[arg-type]
            )
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.scaling_factor)
        t = np.arange(self.max_position_embeddings * self.scaling_factor).astype(
            np.float32
        )
        self.freqs = Tensor(freqs.reshape(1, 1, 1, -1), dtype=self.dtype)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) * self.mscale  # (seq_len, head_dim)
        freqs_sin = np.sin(emb) * self.mscale  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()

        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = self.gather(self.freqs_cos, positions.view(-1), 0)
            freqs_sin = self.gather(self.freqs_sin, positions.view(-1), 0)

        return self.rotary_embedding_op(
            query, key, freqs_cos, freqs_sin, batch_valid_length
        )


# Adapt from: https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/layers/rotary_embedding.py
class Qwen3RotaryEmbedding(nn.Cell):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        dtype: Optional[Type],
    ) -> None:
        super().__init__()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()

        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        freqs_base = mint.arange(0, self.rotary_dim, 2).astype(
            np.float32
        )  # (head_dim // 2, )
        freqs = 1.0 / (base ** (freqs_base / self.rotary_dim))  # (head_dim // 2, )
        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.base)
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)  # (seq_len, head_dim)
        freqs_sin = np.sin(emb)  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()

        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = self.gather(self.freqs_cos, positions.view(-1), 0)
            freqs_sin = self.gather(self.freqs_sin, positions.view(-1), 0)

        return self.rotary_embedding_op(
            query, key, freqs_cos, freqs_sin, batch_valid_length
        )


class Qwen3Attention(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = float(self.head_dim**-0.5)
        self.rope_theta = int(config.rope_theta)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling["rope_type"]
            self.rope_factor = config.rope_scaling["factor"]
            self.rope_max_position_embeddings = config.rope_scaling[
                "original_max_position_embeddings"
            ]
        else:
            self.rope_type = "default_rope"

        self.attn = MsNativeAttnBackend(
            config.num_attention_heads // self.tp_size,
            config.head_dim,
            config.num_key_value_heads // self.tp_size,
        )

        self.q_proj = Qwen3ColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.q_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(
            norm_dim=config.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )
        self.k_proj = Qwen3ColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
        )
        self.k_norm = RMSNorm(
            norm_dim=config.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )
        self.v_proj = Qwen3ColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
        )
        self.o_proj = Qwen3RowParallelLinear(
            input_size=self.q_size,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=config.attention_bias,
        )
        self.rotary_emb = None
        if self.rope_type == "yarn":
            self.rotary_emb = InferYaRNScalingRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.rope_max_position_embeddings,
                base=self.rope_theta,
                is_neox_style=True,
                scaling_factor=self.rope_factor,
                dtype=self.param_dtype,
            )
        else:
            self.rotary_emb = Qwen3RotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.max_position,
                base=self.rope_theta,
                dtype=self.param_dtype,
            )

    def construct(
        self,
        hidden_state: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        token_lens, hidden_dim = hidden_state.shape

        q = self.q_proj(hidden_state).view(-1, self.head_dim).contiguous()
        k = self.k_proj(hidden_state).view(-1, self.head_dim).contiguous()
        v = (
            self.v_proj(hidden_state)
            .view(-1, self.kv_size // self.tp_size)
            .contiguous()
        )

        q = self.q_norm(q).view(-1, self.q_size // self.tp_size)
        k = self.k_norm(k).view(-1, self.kv_size // self.tp_size)

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        k = k.contiguous()
        v = v.contiguous()

        key_out = self.attn(
            k,
            v,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
        )
        q = ops.depend(q, key_out)

        if is_prefill:
            attn_output = self.attn.extend(
                q, k, v, attn_mask, None, None, None, q_seq_lens, batch_valid_length
            )
        else:
            attn_output = self.attn.decode(
                q,
                batch_valid_length,
                attn_mask,
                q_seq_lens,
                key_cache,
                value_cache,
                block_tables,
            )

        output = self.o_proj(attn_output).view(token_lens, -1)
        return output


class Qwen3DecoderLayer(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config)
        self.mlp = Qwen3MLP(config=config)
        self.input_layernorm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    def construct(
        self,
        hidden_state: Tensor,
        residual: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        layer_idx: int,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
        else:
            hidden_state, residual = self.input_layernorm(hidden_state, residual)
        hidden_state = self.self_attn(
            hidden_state=hidden_state,
            positions=positions,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            layer_idx=layer_idx,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )
        hidden_state, residual = self.post_attention_layernorm(hidden_state, residual)
        hidden_state = self.mlp(hidden_state)

        return hidden_state, residual


class Qwen3Model(nn.Cell):
    r"""
    qwen3 model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(config=config)

        self.layers = nn.CellList()

        for i in range(self.num_hidden_layers):
            layer = Qwen3DecoderLayer(config=config)
            self.layers.append(layer)

        self.norm = RMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
        )

    # pylint: disable=W0613
    @jit
    def construct(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        batch_valid_length=None,
        is_prefill=True,
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        block_tables=None,
    ):
        """
        Forward of qwen model.
        """
        hidden_state = self.embed_tokens(input_ids)
        residual = None
        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            hidden_state, residual = layer(
                hidden_state=hidden_state,
                residual=residual,
                positions=position_ids,
                batch_valid_length=batch_valid_length,
                is_prefill=is_prefill,
                layer_idx=i,
                attn_mask=attention_mask,
                q_seq_lens=q_seq_lens,
                key_cache=key_cache[i],
                value_cache=value_cache[i],
                out_cache_loc=out_cache_loc,
                block_tables=block_tables,
            )

        hidden_state, _ = self.norm(hidden_state, residual)

        return hidden_state


class GatherLastDim(nn.Cell):
    def __init__(self):
        super().__init__()
        tp_group_name = _get_tp_group_name()
        self.all_gather = ops.AllGather(group=tp_group_name)
        self.world_size = get_tensor_model_parallel_world_size()
        self.split = ops.Split(axis=0, output_num=self.world_size)

    def construct(self, input: Tensor) -> Tensor:
        output = self.all_gather(input)
        tensor_list = self.split(output)
        output = ops.cat(tensor_list, axis=-1)
        return output


class Qwen3ForCausalLM(MindSporeModelBase):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prev_prefill = False

        self.config = config

        param_dtype = dtype.bfloat16
        if hasattr(dtype, self.config.torch_dtype):
            param_dtype = getattr(dtype, self.config.torch_dtype)
        setattr(self.config, "param_dtype", param_dtype)

        self.model = Qwen3Model(self.config)

        self.lm_head = Qwen3ColParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False,
        )
        self.all_gather = GatherLastDim()
        self.gather = ops.Gather()

        # for best performance of MindSpore for Qwen3
        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = (
            "FlashAttentionScore,PagedAttention"
        )
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "RmsNorm"

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=dtype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=dtype.int64)

        head_size = self.config.head_dim
        # use pa, if use ifa, the shape should (None, None, head_size)
        kv_cache_shape = (None, None, None, head_size)

        kv_cache_dtype = self.config.param_dtype

        num_layers = self.config.num_hidden_layers

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

        dyn_out_cache_loc = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dynamic_attention_mask = Tensor(
            shape=[None, None], dtype=self.config.param_dtype
        )
        dyn_batch_valid_length = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_q_seq_lens = Tensor(
            shape=[
                None,
            ],
            dtype=dtype.int32,
        )
        dyn_block_tables = Tensor(shape=[None, None], dtype=dtype.int32)
        # dyn_intermediate_tensors = None
        # dyn_inputs_embeds = None
        self.model.set_inputs(
            input_ids=dyn_input_ids,
            position_ids=dyn_position_ids,
            attention_mask=dynamic_attention_mask,
            batch_valid_length=dyn_batch_valid_length,
            is_prefill=is_prefill,
            q_seq_lens=dyn_q_seq_lens,
            key_cache=dyn_key_caches,
            value_cache=dyn_value_caches,
            out_cache_loc=dyn_out_cache_loc,
            block_tables=dyn_block_tables,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        param_dict = self.parameters_dict()

        for name, weight in weights:
            if name in param_dict:
                param = param_dict[name]
                if hasattr(param, "weight_load"):
                    weight_load = getattr(param, "weight_load")
                    weight_load(param, weight)
                    param.set_data(param.move_to("Ascend"))
                else:
                    param.set_data(tensor_torch2ms(weight).move_to("Ascend"))
                # Make sure the weight is loaded on device, so the kv cache calculation is correct.

    def construct(self, **model_inputs) -> Tensor:
        q_seq_lens = model_inputs["q_seq_lens"]
        is_prefill = model_inputs["is_prefill"]

        if self.prev_prefill != is_prefill:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        hidden_state = self.model(**model_inputs)

        # TODO: In pure decode scenarios, cumsum and gather operations will be redundant .
        q_seq_lens = mint.cumsum(q_seq_lens, 0)
        hidden_state = self.gather(hidden_state, q_seq_lens - 1, 0)

        logits = self.lm_head(hidden_state)
        logits = self.all_gather(logits)
        logits = ops.cast(logits, dtype.float32)
        logits = ops.reshape(logits, (-1, logits.shape[-1]))
        return logits
