# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only MiniCPM model compatible with HuggingFace weights."""

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from cachetools import Cache
from einops import rearrange
from torch import nn
from transformers import DynamicCache
from transformers.modeling_flash_attention_utils import flash_attn_func
from transformers.utils import is_flash_attn_greater_or_equal_2_10

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
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
from sglang.srt.utils import add_prefix, logger


class MiniCPMMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # set rope as fp32 instead of bf16
        self.rotary_emb.cos_sin_cache = self.rotary_emb._compute_cos_sin_cache()
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
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        orig_dtype = q.dtype
        q, k = q.float(), k.float()
        q, k = self.rotary_emb(positions, q, k)
        q, k = q.to(orig_dtype), k.to(orig_dtype)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, moba_chunk_size, kernel_stride):
    """
    计算需要 MOBA 注意力的 chunks，支持 stride。
    返回:
        - filtered_indices: 用于直接索引 kv 的索引。
        - cu_seqlens_compressed: 压缩后的累积序列长度。
    """
    # 1. 计算每个序列的长度
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. 计算每个序列的 chunk 起始位置 (考虑 stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (
        max_seq_len - moba_chunk_size
    ) // kernel_stride + 1  # 修正公式
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = (
        seq_starts[:, None] + chunk_start_offsets[None, :]
    )  # [batch_size, max_num_chunks_per_seq]

    # 3. 过滤掉超出序列长度的 chunk 和非完整大小的 chunk
    chunk_end_in_seq = chunk_start_in_seq + moba_chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (
        seq_starts[:, None] + batch_sizes[:, None]
    )  # 完整 chunk

    # 4. 根据 valid_chunk_mask 过滤有效的 chunk 起始位置
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. 生成 filtered_indices
    chunk_indices = torch.arange(0, moba_chunk_size, device=cu_seqlen.device)[
        None, :
    ]  # [1, moba_chunk_size]
    filtered_indices = (
        valid_chunk_starts[:, None] + chunk_indices
    )  # [num_valid_chunks, moba_chunk_size]
    filtered_indices = filtered_indices.view(-1)  # 展平为一维索引

    # 6. 计算压缩后的累积序列长度
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(
        dim=1
    )  # 每个 batch 的有效 chunk 数量
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del (
        num_filtered_chunks_per_batch,
        chunk_start_offsets,
        seq_starts,
        chunk_end_in_seq,
        valid_chunk_mask,
        chunk_indices,
    )
    return filtered_indices, cu_seqlens_compressed


class CompressKV(torch.nn.Module):
    def __init__(
        self,
        head_num_k,
        head_dim,
        kernel_size,
        compress_func,
        add_pos_embed=False,
        kernel_stride=16,
    ):
        """
        压缩KV模块，支持多种压缩方式
        Args:
            head_num_k: KV头的数量
            head_dim: 每个头的维度
            kernel_size: 每个chunk的大小
            compress_func: 压缩方式（如meanpool, mlp, conv1d等）
            add_pos_embed: 是否添加位置编码
            stride: 分块时的步长
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.compress_func = compress_func
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride  # 新增stride参数

        # 定义不同的压缩方式
        if compress_func == "mlp" or compress_func == "mlp+residual":
            self.kv_compress = nn.Sequential(
                nn.Linear(kernel_size * 2, kernel_size * 4),
                nn.ReLU(),
                nn.Linear(kernel_size * 4, 2),
            )
        elif compress_func == "conv1d":
            self.k_conv = nn.Conv1d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=kernel_size,
            )
            self.v_conv = nn.Conv1d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=kernel_size,
            )
        elif compress_func == "weighted_sum":
            self.weight_net_v = nn.Linear(self.head_dim, 1)
            self.weight_net_k = nn.Linear(self.head_dim, 1)
        elif compress_func == "weighted_sum+proj":
            self.weight_net_v = nn.Linear(self.head_dim, 1)
            self.weight_net_k = nn.Linear(self.head_dim, 1)
            self.k_proj = nn.Linear(self.head_dim, self.head_dim)
            self.v_proj = nn.Linear(self.head_dim, self.head_dim)

        if add_pos_embed:
            # 修改位置编码层：为每个头创建独立的位置编码
            self.pos_embed = nn.Embedding(
                kernel_size,
                head_num_k * head_dim,  # 维度扩展为 [kernel_size, num_heads * head_dim]
            )
        else:
            self.pos_embed = None

    def forward(self, kv: torch.Tensor, cu_seqlens):
        """
        前向传播，压缩KV
        Args:
            kv: 输入的KV张量
            cu_seqlens: 累积序列长度
        Returns:
            compress_k: 压缩后的K
            compress_v: 压缩后的V
            cu_seqlens_compressed: 压缩后的累积序列长度
        """

        # 计算chunk相关信息，支持stride
        filtered_kv_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        # 提取过滤后的kv
        filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

        # 分块
        filtered_kv = filtered_kv.view(
            filtered_kv.shape[0] // self.kernel_size,
            self.kernel_size,
            2,
            self.head_num_k,
            self.head_dim,
        )  # [l, block_size,2,h,d]
        if self.pos_embed is not None:
            positions = torch.arange(self.kernel_size, device=kv.device)
            pos_emb = self.pos_embed(positions)  # [kernel_size, num_heads * head_dim]

            # 重塑形状以匹配多头结构
            pos_emb = pos_emb.view(
                self.kernel_size,
                self.head_num_k,  # 使用实际头数参数（需在__init__中保存）
                self.head_dim,
            )  # [kernel_size, num_heads, head_dim]

            # 添加维度用于广播
            pos_emb = pos_emb.reshape(
                1, self.kernel_size, 1, self.head_num_k, self.head_dim
            )  # [1, block_size, 1, num_heads, head_dim]
            filtered_kv = filtered_kv + pos_emb

        if self.compress_func == "meanpool":
            compressed_kv = filtered_kv.mean(dim=1)
            compress_k = compressed_kv[
                :, 0, :, :
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
            compress_v = compressed_kv[
                :, 1, :, :
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
        elif self.compress_func == "mlp":

            filtered_kv = filtered_kv.permute(0, 3, 4, 2, 1).reshape(
                filtered_kv.shape[0], self.head_num_k, self.head_dim, -1
            )
            compressed_kv = self.kv_compress(filtered_kv)
            compress_k = compressed_kv[
                :, :, :, 0
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
            compress_v = compressed_kv[
                :, :, :, 1
            ]  # .reshape(-1, self.head_num_k, self.head_dim)
        elif self.compress_func == "mlp+residual":
            mean_kv = filtered_kv.mean(dim=1)
            mlp_kv = self.kv_compress(
                filtered_kv.permute(0, 3, 4, 2, 1).reshape(
                    filtered_kv.shape[0], self.head_num_k, self.head_dim, -1
                )
            ).permute(
                0, 3, 1, 2
            )  # [l, h,d,2]->[l,2,h,d]
            compressed_kv = mean_kv + mlp_kv
            compress_k = compressed_kv[:, 0, :, :]
            compress_v = compressed_kv[:, 1, :, :]
        elif self.compress_func == "conv1d":
            k = filtered_kv[:, :, 0, :, :]
            k = rearrange(k, "l block_size h d -> (l h) d block_size")  # 只能3维
            v = filtered_kv[:, :, 1, :, :]
            v = rearrange(v, "l block_size h d -> (l h) d block_size")
            compress_k = self.k_conv(k).squeeze(-1)  # [(l h), d]
            compress_v = self.v_conv(v).squeeze(-1)  # [(l h), d]
            compress_k = rearrange(compress_k, "(l h) d -> l h d", h=self.head_num_k)
            compress_v = rearrange(compress_v, "(l h) d -> l h d", h=self.head_num_k)

        elif self.compress_func == "weighted_sum":
            k = filtered_kv[:, :, 0, :, :]
            k = rearrange(k, "l block_size h d -> l h block_size d")
            v = filtered_kv[:, :, 1, :, :]
            v = rearrange(v, "l block_size h d -> l h block_size d")
            weight_k = torch.softmax(
                self.weight_net_k(k), dim=2
            )  # [l, h, block_size, 1]
            weight_v = torch.softmax(
                self.weight_net_v(v), dim=2
            )  # [l, h, block_size, 1]

            compress_k = (weight_k * k).sum(dim=2)  # [l, h, d]
            compress_v = (weight_v * v).sum(dim=2)  # [l, h, d]
        elif self.compress_func == "weighted_sum+proj":
            k = filtered_kv[:, :, 0, :, :]
            k = rearrange(k, "l block_size h d -> l h block_size d")
            v = filtered_kv[:, :, 1, :, :]
            v = rearrange(v, "l block_size h d -> l h block_size d")
            weight_k = torch.softmax(
                self.weight_net_k(k), dim=2
            )  # [l, h, block_size, 1]
            weight_v = torch.softmax(
                self.weight_net_v(v), dim=2
            )  # [l, h, block_size, 1]

            compress_k = (weight_k * self.k_proj(k)).sum(dim=2)  # [l, h, d]
            compress_v = (weight_v * self.v_proj(v)).sum(dim=2)  # [l, h, d]

        else:
            raise ValueError(f"Unsupported compress type: {self.compress_func}")

        del filtered_kv
        if "compressed_kv" in locals():
            del compressed_kv

        return compress_k, compress_v, cu_seqlens_compressed


class DynamicCacheQKV(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        if num_hidden_layers is None:
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self.query_cache: List[torch.Tensor] = []
        else:
            self.key_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.value_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
            self.query_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                self.query_cache[layer_idx],
            )
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                self.query_cache[layer_idx],
            )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        query_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if query_states is None:
            raise ValueError("query_states must be provided for DynamicCacheQKV")

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.query_cache.append(query_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.query_cache[layer_idx] = query_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            self.query_cache[layer_idx] = torch.cat(
                [self.query_cache[layer_idx], query_states], dim=-2
            )

        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            self.query_cache[layer_idx],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx or (
            len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []
        ):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    # @classmethod
    # def from_legacy_cache(
    #     cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    # ) -> "DynamicCacheQKV":
    #     """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
    #     backward compatibility."""
    #     cache = cls(num_hidden_layers)
    #     if past_key_values is not None:
    #         for layer_idx in range(len(past_key_values)):
    #             key_states, value_states, query_status = past_key_values[layer_idx]
    #             cache.update(key_states, value_states, query_status,layer_idx)
    #     return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search.
        """
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
                self.query_cache[idx] = self.query_cache[idx][..., :max_length, :]

    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int
    ) -> List["DynamicCacheQKV"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCacheQKV(num_hidden_layers)
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [
                tensor[i : i + split_size] for tensor in self.key_cache
            ]
            current_split.value_cache = [
                tensor[i : i + split_size] for tensor in self.value_cache
            ]
            current_split.query_cache = [
                tensor[i : i + split_size] for tensor in self.query_cache
            ]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(
        cls, splits: List["DynamicCacheQKV"], num_hidden_layers: int
    ) -> "DynamicCacheQKV":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls(num_hidden_layers)
        for idx in range(len(splits[0])):
            key_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            value_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            query_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                layer_query = torch.cat(query_cache, dim=0)
                cache.update(layer_keys, layer_values, idx, query_states=layer_query)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.query_cache[layer_idx] = self.query_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
            self.query_cache[layer_idx] = self.query_cache[layer_idx][indices, ...]


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    parallel_topk_compute: Union[str, bool] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Compute attention output and topk block idx used in topk_sparse_attention.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch.
        max_seqlen_k (int): max k len of the batch.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.
        parallel_topk_compute (str, optional): Only set it to False when the sequence length is too long. This can avoid a current bug.
            We'll fix this issue later. Defaults to auto, it will be set to False when the sequence length is greater than 32k and True otherwise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention
    """
    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    attn_output, lse = CompressedAttention.apply(
        q,
        k,
        v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

    # do not select topk index
    if topk <= 0:
        warnings.warn("topk <= 0, returned topk_idx will be None")
        return attn_output, None

    assert topk >= init_blocks  # + local_blocks
    with torch.no_grad():
        num_k_heads, num_q_heads = k.shape[1], q.shape[1]
        num_shared_q_heads = num_q_heads // num_k_heads
        batch_size = cu_seqlens_q.shape[0] - 1
        q_idx = torch.cat(
            [
                torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                for i in range(batch_size)
            ],
            dim=0,
        )
        q_idx = q_idx // block_size
        # whether to use parallel version
        if parallel_topk_compute == "auto":
            parallel_topk_compute = cu_seqlens_q[-1] <= 32768
        # parallel version
        if parallel_topk_compute:
            # recompute score
            score = _get_attention_score(
                q,
                k,
                lse,
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )
            # transform score to block-wise score
            score = transform_score(
                score,
                kernel_size,
                kernel_stride,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                init_blocks,
                local_blocks,
            )
            # get topk
            topk = min(topk, score.shape[-1])
            topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
            # print(cu_seqlens_q)
            # breakpoint()
            topk_idx[topk_idx >= q_idx[None, :, None]] = -1
            topk_idx = topk_idx.to(torch.int32)
        # non parallel version, avoid some current bugs when sequence length is too long
        # FIXME: need to fix later
        else:
            topk_idx_list = []
            for h in range(num_k_heads):
                # recompute score
                score = _get_attention_score(
                    q[:, h * num_shared_q_heads : (h + 1) * num_shared_q_heads],
                    k[:, h : h + 1],
                    lse[h * num_shared_q_heads : (h + 1) * num_shared_q_heads],
                    kernel_size,
                    kernel_stride,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    sm_scale,
                )
                # transform score to block-wise score
                score = transform_score(
                    score,
                    kernel_size,
                    kernel_stride,
                    block_size,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    init_blocks,
                    local_blocks,
                )
                # get topk
                topk = min(topk, score.shape[-1])
                topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
                topk_idx[topk_idx >= q_idx[None, :, None]] = -1
                topk_idx = topk_idx.to(torch.int32)
                topk_idx_list.append(topk_idx)
            topk_idx = torch.cat(topk_idx_list, dim=0)
    return attn_output, topk_idx


class MiniCPMFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        # !  -------nsa-------
        self.kernel_size = 32
        self.kernel_stride = 16
        compress_type = "meanpool"
        self.init_blocks = 1

        self.block_size = 64

        self.window_size = 2048

        self.local_blocks = self.window_size // self.block_size
        # local_blocks
        self.topk = 32
        self.gate_proj = torch.nn.Sequential(
            torch.nn.Linear(self.head_dim, 2),  # 对应三个专家
            torch.nn.Sigmoid(),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # set rope as fp32 instead of bf16
        self.rotary_emb.cos_sin_cache = self.rotary_emb._compute_cos_sin_cache()
        self.compress_kv = CompressKV(
            self.num_key_value_heads,
            self.head_dim,
            compress_func=compress_type,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            add_pos_embed=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "现在只支持batch_size=1"

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(v.to(torch.float32), seq_len=kv_seq_len)
        # if key_states.shape[-2] == 1:  #这里是possition ids的问题
        #     position_ids = torch.tensor([[kv_seq_len-1]], device=key_states.device, dtype=position_ids.dtype)
        #     # breakpoint()

        q, k = q.float(), k.float()
        q, k = self.rotary_emb(positions, q, k)

        # if past_key_value is not None:

        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        new_k = k
        new_v = v
        new_q = q
        try:
            past_k, past_v, past_q = past_key_value.__getitem__(self.layer_idx)
        except Exception as e:
            # If the cache is empty, we need to create a new one
            past_k, past_v, past_q = k, v, q
            new_k, new_v = None, None

        k, v, q = past_key_value.update(
            k,
            v,
            self.layer_idx,
            cache_kwargs,
            query_states=q,
        )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MiniCPMRMSNorm handles it correctly)

        input_dtype = q.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        # if past_key_value is None or new_k is None:
        #     attn_output = self._flash_attention_forward(
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask,
        #         q_len,
        #         dropout=dropout_rate,
        #         original_hidden_states=hidden_states,
        #     )
        # else:
        # breakpoint()
        attn_output = self._flash_attention_forward_with_kv_cache(
            q,
            k,
            v,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            original_hidden_states=hidden_states,
            past_k=past_k,
            past_v=past_v,
            new_k=new_k,
            new_v=new_v,
            new_q=new_q,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _flash_attention_forward_with_kv_cache(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        original_hidden_states=None,
        past_k=None,
        past_v=None,
        new_k=None,
        new_v=None,
        new_q=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            query_length = query_states.shape[1]
            batch_size = query_states.shape[0]

            # query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            #     query_states, key_states, value_states, attention_mask, query_length=query_length
            # )

            # ! 这里的attention_mask是没有包括最后一个,所以不准
            assert batch_size == 1, "现在只支持batch_size=1"
            query_states = query_states.squeeze(0)
            key_states = key_states.squeeze(0)
            value_states = value_states.squeeze(0)
            original_hidden_states = original_hidden_states.squeeze(0)
            cu_seqlens_q = cu_seqlens_k = torch.tensor(
                [0, query_length], device=query_states.device, dtype=torch.int32
            )
            max_seqlen_in_batch_q = max_seqlen_in_batch_k = query_length
            attn_output = self.nsa_forward_with_kv_cache(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                original_hidden_states=original_hidden_states,
                past_k=past_k,
                past_v=past_v,
                new_k=new_k,
                new_v=new_v,
                new_q=new_q,
                batch_size=batch_size,
            )
            # attn_output_unpad = flash_attn_varlen_func(
            #     query_states,
            #     key_states,
            #     value_states,
            #     cu_seqlens_q=cu_seqlens_q,
            #     cu_seqlens_k=cu_seqlens_k,
            #     max_seqlen_q=max_seqlen_in_batch_q,
            #     max_seqlen_k=max_seqlen_in_batch_k,
            #     dropout_p=dropout,
            #     softmax_scale=softmax_scale,
            #     causal=causal,
            # )

            # attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    # !  -------nsa-------
    def nsa_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        original_hidden_states=None,
    ):
        kv = torch.stack((key_layer, value_layer), dim=1)
        compressed_k, compressed_v, compressed_cu_seqlens = self.compress_kv(
            kv, cu_seqlens_k
        )
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        compressed_attn_output, topk_idx = compressed_attention(
            query_layer,
            compressed_k,
            compressed_v,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_in_batch_q,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
        )
        if (
            globals()["debug_token"] == globals()["token_now"]
            and globals()["save_no_cache"]
        ):
            torch.save(
                compressed_attn_output,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/compressed_attn_output.pt",
            )
            torch.save(
                topk_idx,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/topk_idx.pt",
            )
            torch.save(
                compressed_k,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/compressed_k.pt",
            )
            torch.save(
                compressed_v,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/compressed_v.pt",
            )
            torch.save(
                compressed_cu_seqlens,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/compressed_cu_seqlens.pt",
            )
            torch.save(
                compressed_seqlens,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/compressed_seqlens.pt",
            )
            torch.save(
                cu_seqlens_q,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/cu_seqlens_q.pt",
            )
            torch.save(
                cu_seqlens_k,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/cu_seqlens_k.pt",
            )
            torch.save(
                query_layer,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/query_layer.pt",
            )
            torch.save(
                key_layer,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/key_layer.pt",
            )
            torch.save(
                value_layer,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/value_layer.pt",
            )
        # raise ValueError('debug')
        # topk_idx_np = topk_idx.cpu().numpy()

        # # 设置numpy的打印选项，这里将阈值设为1000000
        # np.set_printoptions(threshold=1000000)

        # # 打印numpy数组
        # print(topk_idx_np)
        del compressed_k, compressed_v, compressed_cu_seqlens, kv, compressed_seqlens
        nheads_k = key_layer.shape[1]
        head_mask_type = torch.tensor(
            [1] * nheads_k, device=query_layer.device, dtype=torch.int32
        )
        streaming_info = torch.tensor(
            [0, 0] * nheads_k, device=query_layer.device, dtype=torch.int32
        )
        exact_streaming = False

        repeat_times = 1
        if repeat_times > 1:
            query_layer_repeat = query_layer.repeat_interleave(repeat_times, dim=-2)
        else:
            query_layer_repeat = query_layer
        topk_attn_output = block_sparse_attn_func(
            query_layer_repeat,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            topk_idx,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            self.attention_dropout,
            deterministic=False,
            softmax_scale=None,
            is_causal=True,
            exact_streaming=False,
            return_attn_probs=False,
            block_window_size=self.window_size // self.block_size,
        )
        # import pdb; pdb.set_trace()
        # raise ValueError('debug')
        if repeat_times > 1:
            topk_attn_output = topk_attn_output.view(
                topk_attn_output.shape[0],
                topk_attn_output.shape[1] // repeat_times,
                repeat_times,
                -1,
            ).mean(dim=-2)
        return topk_attn_output
        if original_hidden_states is not None:
            gate_input = original_hidden_states.squeeze(1).view(
                query_layer.shape[0], -1, query_layer.shape[-1]
            )  # 分头
        else:
            gate_input = query_layer
        # breakpoint()
        gate_scores = self.gate_proj(gate_input)
        attn_output = (
            # gate_scores[..., 0:1] * sliding_attn_output   # compressed_attn_output
            gate_scores[..., 0:1] * compressed_attn_output  # topk_attn_output
            + gate_scores[..., 1:2] * topk_attn_output
        )
        if (
            globals()["debug_token"] == globals()["token_now"]
            and globals()["save_no_cache"]
        ):
            torch.save(
                topk_attn_output,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/topk_attn_output.pt",
            )
            torch.save(
                gate_scores,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/gate_scores.pt",
            )
            torch.save(
                gate_input,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/gate_input.pt",
            )
            torch.save(
                attn_output,
                "/home/zhouzihan/project/LongContext/patch_nsa_model/.vscode/debug_kvcache/no_cache/attn_output.pt",
            )
            raise ValueError("debug")
        del (
            gate_input,
            compressed_attn_output,
            query_layer,
            key_layer,
            value_layer,
            original_hidden_states,
        )
        del gate_scores, topk_attn_output
        return attn_output

    # !  -------nsa-------
    def nsa_forward_with_kv_cache(
        self,
        query_layer,
        key_layer,
        value_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        original_hidden_states=None,
        past_k=None,
        past_v=None,
        new_k=None,
        new_v=None,
        new_q=None,
        batch_size=None,
    ):
        # breakpoint()
        kv = torch.stack((key_layer, value_layer), dim=1)
        compressed_k, compressed_v, compressed_cu_seqlens = self.compress_kv(
            kv, cu_seqlens_k
        )
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        compressed_attn_output, topk_idx = compressed_attention(
            query_layer,
            compressed_k,
            compressed_v,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_in_batch_q,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
        )
        compressed_attn_output = compressed_attn_output[-1].unsqueeze(0).unsqueeze(0)
        # raise ValueError('debug')
        # topk_idx_np = topk_idx.cpu().numpy()

        # # 设置numpy的打印选项，这里将阈值设为1000000
        # np.set_printoptions(threshold=1000000)

        # # 打印numpy数组
        # print(topk_idx_np)
        del compressed_k, compressed_v, compressed_cu_seqlens, kv, compressed_seqlens
        nheads_k = key_layer.shape[1]
        head_mask_type = torch.tensor(
            [1] * nheads_k, device=query_layer.device, dtype=torch.int32
        )
        streaming_info = torch.tensor(
            [0, 0] * nheads_k, device=query_layer.device, dtype=torch.int32
        )
        exact_streaming = False

        repeat_times = 1
        past_k = past_k.transpose(1, 2).contiguous()
        past_v = past_v.transpose(1, 2).contiguous()
        if new_k is not None:
            new_k = new_k.transpose(1, 2).contiguous()
        if new_v is not None:
            new_v = new_v.transpose(1, 2).contiguous()
        new_q = new_q.transpose(1, 2).contiguous()
        if repeat_times > 1:
            new_q = new_q.repeat_interleave(repeat_times, dim=-2)
        else:
            new_q = new_q
        # ! 暂时
        # assert batch_size == 1, '只支持batch_size =1'

        cache_batch_idx = torch.arange(
            batch_size, device=query_layer.device, dtype=torch.int32
        )
        current_seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        new_topk_idx = []
        if new_k is not None:
            for i in range(batch_size):
                new_topk_idx.append(
                    topk_idx[:, current_seqlens_k[i] - 1, :].unsqueeze(1)
                )
            topk_idx = torch.stack(new_topk_idx, dim=0)
        else:
            # prefilling
            for i in range(batch_size):
                if i == 0:
                    start = 0
                else:
                    start = current_seqlens_k[i - 1]
                new_topk_idx.append(topk_idx[:, start : current_seqlens_k[i], :])
            topk_idx = torch.stack(new_topk_idx, dim=0)
        seqlen_k = key_layer.shape[0]  # ! 只考虑单个batch
        seqlens_k = torch.full(
            (batch_size,), seqlen_k - 1, dtype=torch.int32, device=new_q.device
        )

        # print(cache_batch_idx)
        # topk_idx,_ = torch.sort(topk_idx,dim=-1, descending=True) #! 需要逆序
        # past_k = key_layer.unsqueeze(0)[:, :-1].contiguous()
        # past_v = value_layer.unsqueeze(0)[:, :-1].contiguous()
        past_k = torch.cat(
            [past_k, torch.zeros_like(new_k, dtype=new_k.dtype)], dim=1
        )  # 填充多一个
        past_v = torch.cat(
            [past_v, torch.zeros_like(new_v, dtype=new_v.dtype)], dim=1
        )  # 填充多一个
        # TODO: triton kernel for qvk
        topk_attn_output, softmax_lse = block_sparse_attn_kvcache_func(
            q=new_q,  # [batch_size, seqlen_q, nheads, d]
            k_cache=past_k,  # [batch_size, max_seqlen_k, nheads_k, d]
            v_cache=past_v,  # [batch_size, max_seqlen_k, nheads_k, d]
            m_block_dim=16,
            n_block_dim=64,
            head_mask_type=head_mask_type,
            streaming_info=None,  # streaming_info,
            topk_idx=topk_idx,
            k=new_k,  # [batch_size, 1, nheads_k, d]
            v=new_v,  # [batch_size, 1, nheads_k, d]
            seqlens_k=seqlens_k,  # current_seqlens_k-1 ,#! 这边要对齐kv cahce的长度, # Current positions in cache
            rotary_cos=None,  # No rotary embeddings
            rotary_sin=None,  # No rotary embeddings
            cache_batch_idx=cache_batch_idx,
            alibi_slopes=None,
            softmax_scale=None,
            causal=False,  # Renaming to match function signature
            exact_streaming=exact_streaming,
            window_size_left=-1,  # Using individual parameters instead of tuple
            window_size_right=-1,
            block_window_size=self.window_size // self.block_size,
            rotary_interleaved=False,
            num_splits=16,
            # num_topk=self.topk,
        )
        # raise ValueError('debug')
        # topk_attn_output = block_sparse_attn_func(
        #     query_layer_repeat,
        #     key_layer,
        #     value_layer,
        #     cu_seqlens_q,
        #     cu_seqlens_k,
        #     head_mask_type,
        #     streaming_info,
        #     topk_idx,
        #     max_seqlen_in_batch_q, max_seqlen_in_batch_k,
        #     self.attention_dropout,
        #     deterministic=False,
        #     softmax_scale=None,
        #     is_causal=True,
        #     exact_streaming=False,
        #     return_attn_probs=False,
        #     block_window_size=self.window_size // self.block_size,
        # )
        if repeat_times > 1:
            topk_attn_output = topk_attn_output.view(
                topk_attn_output.shape[0],
                topk_attn_output.shape[1],
                topk_attn_output.shape[2] // repeat_times,
                repeat_times,
                -1,
            ).mean(dim=-2)
        return topk_attn_output
        # raise ValueError('debug')
        if original_hidden_states is not None:
            gate_input = original_hidden_states.squeeze(1).view(
                topk_attn_output.shape[0],
                topk_attn_output.shape[1],
                topk_attn_output.shape[2],
                -1,
            )  # 分头
        else:
            gate_input = new_q

        # breakpoint()
        gate_scores = self.gate_proj(gate_input)
        attn_output = (
            # gate_scores[..., 0:1] * sliding_attn_output   # compressed_attn_output
            gate_scores[..., 0:1] * compressed_attn_output  # topk_attn_output
            + gate_scores[..., 1:2] * topk_attn_output
        )
        del (
            compressed_attn_output,
            query_layer,
            key_layer,
            value_layer,
            original_hidden_states,
        )
        del gate_scores, topk_attn_output
        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
}


class MiniCPMDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](
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
        self.mlp = MiniCPMMLP(
            hidden_size=self.hidden_size,
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
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        return hidden_states, None


class MiniCPMModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiniCPMForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.num_experts = getattr(self.config, "num_experts", 0)
        self.quant_config = quant_config
        self.model = MiniCPMModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if not self.config.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
        else:
            lm_head = self.lm_head
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            )
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = MiniCPMForCausalLM
