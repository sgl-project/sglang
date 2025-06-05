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
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import flash_attn_2_cuda as flash_attn_gpu
import torch
from einops import rearrange
from torch import nn
from transformers import Cache, DynamicCache, PretrainedConfig
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
from sglang.srt.layers.rotary_embedding import MiniCPMScaledRotaryEmbedding, get_rope
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
        **kwargs,
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
        **kwargs,
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
def calc_chunks_with_stride(cu_seqlens, moba_chunk_size, kernel_stride):
    """
    计算需要 MOBA 注意力的 chunks，支持 stride。
    返回:
        - filtered_indices: 用于直接索引 kv 的索引。
        - cu_seqlens_compressed: 压缩后的累积序列长度。
    """
    assert cu_seqlens.dim() == 1
    # 1. 计算每个序列的长度
    batch_sizes = cu_seqlens[1:] - cu_seqlens[:-1]

    # 2. 计算每个序列的 chunk 起始位置 (考虑 stride)
    # print(f"{cu_seqlens=}")
    # print(f"{batch_sizes=}")
    # print(f"{moba_chunk_size=}")
    max_seq_len = torch.max(batch_sizes)
    # print(f"{max_seq_len=}")
    # max_num_chunks_per_seq = (
    #                              max_seq_len - moba_chunk_size
    #                          ) // kernel_stride + 1  # 修正公式
    max_num_chunks_per_seq = torch.clamp_min(
        (max_seq_len - moba_chunk_size) // kernel_stride + 1, 0
    )
    # print(f"{max_num_chunks_per_seq * kernel_stride=}")
    # print(f"{kernel_stride=}")
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlens.device,
    )
    seq_starts = cu_seqlens[:-1]
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
    chunk_indices = torch.arange(0, moba_chunk_size, device=cu_seqlens.device)[
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
        len(cu_seqlens), dtype=torch.int32, device=cu_seqlens.device
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
            self.compress_k_cache: List[torch.Tensor] = []
            self.no_compress_k_cache: List[torch.Tensor] = []
            self.cached_compressed_cu_seqlens: List[torch.Tensor] = []
        else:
            self.key_cache = [[] for _ in range(num_hidden_layers)]
            self.value_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
            self.compress_k_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
            self.no_compress_k_cache: List[torch.Tensor] = [
                [] for _ in range(num_hidden_layers)
            ]
            self.cached_compressed_cu_seqlens: List[torch.Tensor] = [
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
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
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
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

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

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_compressed_k(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        if len(self.compress_k_cache) <= layer_idx:
            return None
            # self.compress_k_cache.append(torch.zeros((0, 0), dtype=torch.bfloat16))

        return self.compress_k_cache[layer_idx]

    def get_compressed_cu_seqlens(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        if len(self.cached_compressed_cu_seqlens) <= layer_idx:
            self.cached_compressed_cu_seqlens.append(torch.zeros(2, dtype=torch.int32))

        return self.cached_compressed_cu_seqlens[layer_idx]

    def update_compress_k(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """

        # Update the cache
        if len(self.compress_k_cache) <= layer_idx:
            self.compress_k_cache.append(key_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.compress_k_cache[layer_idx] == []:
            self.compress_k_cache[layer_idx] = key_states

        else:
            self.compress_k_cache[layer_idx] = torch.cat(
                [self.compress_k_cache[layer_idx], key_states], dim=0
            )
        return self.compress_k_cache[layer_idx]

    def update_no_compress_k(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        kernel_size: int = 32,
        kernel_stride: int = 16,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """

        # Update the cache
        if len(self.no_compress_k_cache) <= layer_idx:
            self.no_compress_k_cache.append(key_states)

        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.no_compress_k_cache[layer_idx] == []:
            self.no_compress_k_cache[layer_idx] = key_states

        else:
            self.no_compress_k_cache[layer_idx] = torch.cat(
                [self.no_compress_k_cache[layer_idx], key_states], dim=0
            )

        current_len = self.no_compress_k_cache[layer_idx].shape[0]

        if current_len >= kernel_size:
            k_chunk = self.no_compress_k_cache[layer_idx][:kernel_size]

            self.no_compress_k_cache[layer_idx] = self.no_compress_k_cache[layer_idx][
                kernel_stride:
            ]

            return k_chunk
        else:
            return None

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

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class CompressK(torch.nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        Module for compressing key (K) representations.

        Args:
            head_num_k (int): Number of key attention heads.
            head_dim (int): Dimension of each attention head.
            kernel_size (int): Size of each chunk used for compression.
            kernel_stride (int, optional): Stride used when dividing input into chunks. Default is 16.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, k: torch.Tensor, cu_seqlens):
        """
        Forward pass for compressing the key (K) tensor.

        Args:
            k (torch.Tensor): Input key tensor of shape (total_seq_len, num_heads, head_dim).
            cu_seqlens (torch.Tensor): Cumulative sequence lengths for each sample in the batch, typically used for handling variable-length sequences.

        Returns:
            compress_k (torch.Tensor): Compressed key tensor.
            cu_seqlens_compressed (torch.Tensor): Updated cumulative sequence lengths after compression.

        """

        # Compute chunk-related metadata, with stride support
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        # Extract filtered key vectors
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))

        # split
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )  # [l, block_size,h,d]
        # print(f"{filtered_k.shape=}")
        # print(f"{k.shape=}")
        # print(f"{cu_seqlens_compressed=}")

        compressed_k = filtered_k.mean(dim=1)

        return compressed_k, cu_seqlens_compressed


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
    total_seq_lens=-1,
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
    from flash_attn import flash_attn_nsa_stage1

    with torch.no_grad():

        num_k_heads, num_q_heads = k.shape[1], q.shape[1]
        num_shared_q_heads = num_q_heads // num_k_heads
        cache_len = 0
        batch_size = cu_seqlens_q.shape[0] - 1
        if total_seq_lens == -1:
            total_seq_lens = max_seqlen_q
            q_idx = torch.cat(
                [
                    torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                    + total_seq_lens
                    - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                    for i in range(batch_size)
                ],
                dim=0,
            )
            q_idx = q_idx // block_size

        else:
            cache_len = total_seq_lens - max_seqlen_q
            assert batch_size == 1, "batch_size must be 1 when total_seq_lens is set"
            q_idx = (
                torch.tensor([total_seq_lens - 1], device=q.device, dtype=torch.int32)
                // block_size
            )

        score = flash_attn_nsa_stage1(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=q_idx.shape[0] > 1,
        )
        score = score[:, : q_idx.shape[0], :]

        from nsa.max_pooling_1d import max_pooling_1d

        # Replace transform_score with max_pooling_1d
        block_score = max_pooling_1d(
            score.contiguous(),
            cache_len=cache_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )
        # get topk
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx >= q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


uint64_memory = None


def topk_to_uint64(
    topk_idx: torch.Tensor, max_seqlen_k: int, block_size: int
) -> Tuple[torch.Tensor, int]:
    """
    Convert topk indices directly to uint64 representation without intermediate bool mask

    Args:
        topk_idx: Tensor of shape [batch, num_heads, total_seqlen, k] or [num_heads, total_seqlen, k]
                 containing block indices
        max_seqlen_k: Maximum sequence length for keys
        block_size: Size of each block

    Returns:
        Tuple of:
            uint64_arrays: Tensor with the same batch dimensions but last dim replaced with uint64 values
            k_blocks: Number of key blocks
    """
    assert topk_idx.dtype == torch.int32
    # Calculate key blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size  # Ceiling division

    # Record original shape
    original_shape = topk_idx.shape

    # Check if we have a batch dimension
    has_batch = len(original_shape) == 4

    if has_batch:
        batch_size, num_heads, total_seqlen, k = original_shape
    else:
        num_heads, total_seqlen, k = original_shape
        batch_size = 1

    # Compute how many uint64 values are needed per row
    n_uint64_per_row = (k_blocks + 63) // 64
    # Flatten batch dimensions
    if has_batch:
        flat_dims = batch_size * num_heads * total_seqlen

        # Create output tensor
        output_shape = (batch_size, num_heads, total_seqlen, n_uint64_per_row)
    else:
        flat_dims = num_heads * total_seqlen

        # Create output tensor
        output_shape = (num_heads, total_seqlen, n_uint64_per_row)

    global uint64_memory
    if uint64_memory is None or uint64_memory.shape != output_shape:
        result = torch.zeros(output_shape, dtype=torch.int64, device=topk_idx.device)
        uint64_memory = result
    else:
        result = uint64_memory

    from sgl_kernel import topk_to_uint64 as topk_to_uint64_sgl

    # Call CUDA kernel
    torch.ops.sgl_kernel.topk_to_uint64(
        topk_idx, result, flat_dims, k, k_blocks, n_uint64_per_row, 0
    )

    return result, k_blocks


def flash_attn_with_kvcache(
    attn: RadixAttention,
    q,
    forward_batch: ForwardBatch,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    topk_idx=None,
    block_window_size=0,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    # assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    # assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # if cache_seqlens is not None and isinstance(cache_seqlens, int):
    # cache_seqlens = torch.full(
    #     (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
    # )
    # cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    if topk_idx is not None:
        assert topk_idx.dtype == torch.int32
        from nsa import (
            topk_to_uint64 as cuda_topk_to_uint64,  # Import the new conversion function
        )

        # blockmask, _ = topk_to_uint64(
        seq_len = forward_batch.seq_lens_sum
        blockmask, _ = cuda_topk_to_uint64(
            topk_idx,
            (seq_len if block_table is None else block_table.shape[1] * seq_len),
            64,
        )  # N_BLOCK_DIM=64
    else:
        blockmask = None

    softmax_lse = None
    # FIXME: If key is supplied, it must have seqlen <= the seqlen of the KV cache
    # out, softmax_lse = flash_attn_gpu.fwd_kvcache(
    #     q,
    #     k_cache,
    #     v_cache,
    #     k,
    #     v,
    #     cache_seqlens,
    #     rotary_cos,
    #     rotary_sin,
    #     cache_batch_idx,
    #     cache_leftpad,
    #     block_table,
    #     alibi_slopes,
    #     None,
    #     softmax_scale,
    #     causal,
    #     window_size[0],
    #     window_size[1],
    #     softcap,
    #     rotary_interleaved,
    #     num_splits,
    #     blockmask,
    #     block_window_size,
    # )
    # print(f"{q.shape=}")
    # print(f"{k.shape=}")
    # print(f"{v.shape=}")
    out = attn(
        q,
        k,
        v,
        forward_batch,
        is_sparse=True,
        blockmask=blockmask,
        block_window_size=block_window_size,
        # k_cache=k_cache,
        # v_cache=v_cache,
    )

    # from sgl_kernel.flash_attn import (
    #     flash_attn_with_kvcache as flash_attn_with_kvcache_kernel,
    # )
    #
    # out = flash_attn_with_kvcache_kernel(
    #     q,
    #     k_cache,
    #     v_cache,
    #     k,
    #     v,
    #     qv=None,
    #     rotary_cos=rotary_cos,
    #     rotary_sin=rotary_sin,
    #     cache_seqlens=cache_seqlens,
    #     cache_batch_idx=cache_batch_idx,
    #     cache_leftpad=cache_leftpad,
    #     page_table=block_table,
    #     # None,
    #     softmax_scale=softmax_scale,
    #     causal=causal,
    #     window_size=window_size,
    #     softcap=softcap,
    #     rotary_interleaved=rotary_interleaved,
    #     num_splits=num_splits,
    #     # blockmask=blockmask,
    #     # block_window_size=block_window_size,
    # )
    return (out, softmax_lse) if return_softmax_lse else out


class MiniCPMSparseFlashAttention2(nn.Module):
    # class MiniCPMSparseFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_id
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
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
        rope_theta = getattr(config, "rope_theta", 10000)
        self.rope_theta = rope_theta
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.max_position_embeddings = max_position_embeddings

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        #  -------sparse-------
        self.kernel_size = self.config.sparse_config.get("kernel_size", 32)
        self.kernel_stride = self.config.sparse_config.get("kernel_stride", 16)

        self.init_blocks = self.config.sparse_config.get("init_blocks", 1)

        self.block_size = self.config.sparse_config.get("block_size", 64)

        self.window_size = self.config.sparse_config.get("window_size", 2048)

        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.topk = self.config.sparse_config.get("topk", 64)
        self.use_nope = self.config.sparse_config.get("use_nope", False)
        self.compress_k = CompressK(
            num_kv_heads,
            self.head_dim,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
        )

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

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.hidden_size = hidden_size
        # set rope as fp32 instead of bf16
        scaling_type = self.config.rope_scaling["rope_type"]
        if scaling_type == "longrope":
            # self.rotary_emb = MiniCPMLongRoPE(
            self.rotary_emb = MiniCPMScaledRotaryEmbedding(
                self.head_dim,
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                short_factor=self.config.rope_scaling["short_factor"],
                long_factor=self.config.rope_scaling["long_factor"],
                base=self.rope_theta,
                original_max_position_embeddings=self.config.rope_scaling[
                    "original_max_position_embeddings"
                ],
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        self.is_causal = True

    def nsa_forward(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        forward_batch: ForwardBatch,
        no_rope_param=None,
        past_key_value=None,
    ):
        # assert forward_batch.forward_mode.is_extend(), forward_batch.forward_mode
        stage1_k = k if no_rope_param is None else no_rope_param["key_states_no_rope"]
        compressed_k, compressed_cu_seqlens = self.compress_k(stage1_k, cu_seqlens_k)
        compressed_v = compressed_k.clone()
        if past_key_value is not None:
            # Compute the start indices of keys (k) that were not compressed, Only batch_size=1 is supported at the moment.
            no_compress_k_start = compressed_k.shape[0] * self.kernel_stride
            past_key_value.update_compress_k(compressed_k, self.layer_idx)
            past_key_value.update_no_compress_k(
                k[no_compress_k_start:], self.layer_idx, no_compress_k_start
            )
            # print(f"{compressed_k.shape=}")
            # print(f"{k.shape=}")
            past_key_value.cached_compressed_cu_seqlens.append(compressed_cu_seqlens)
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        topk_idx = compressed_attention(
            (q if no_rope_param is None else no_rope_param["query_states_no_rope"]),
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

        from flash_attn import (
            flash_attn_func,
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )

        topk_attn_output = self.attn.forward(
            q,
            k,
            v,
            forward_batch,
            is_sparse=True,
            block_window_size=self.window_size // self.block_size,
            topk_idx=topk_idx,
        )
        return topk_attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):

        # def unpad_input(
        #     hidden_states: torch.Tensor,  # Shape (batch_size, seq_len, *other_dims)
        #     attention_mask,
        # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        #     """
        #     处理一个矩形且完全填充的序列批次的 'unpadding'。
        #     这意味着批处理中的每个序列都具有相同的长度 'seq_len'，
        #     并且该长度内的所有token都是有效的（没有padding）。
        #
        #     Args:
        #         hidden_states (torch.Tensor):
        #             输入张量，形状为 (batch_size, seq_len, *other_dims)。
        #
        #     Returns:
        #         一个元组: (unpadded_tensor, indices, cu_seqlens, max_seqlen)
        #             - unpadded_tensor (torch.Tensor):
        #                 重塑后的 hidden_states。形状为 (batch_size * seq_len, *other_dims)。
        #             - indices (torch.Tensor):
        #                 将 unpadded_tensor 中的token映射回概念上的原始扁平化张量的索引。
        #                 形状为 (batch_size * seq_len,)。
        #             - cu_seqlens (torch.Tensor):
        #                 累积序列长度。形状为 (batch_size + 1,)。
        #             - max_seqlen (int):
        #                 序列长度，在此即为最大实际序列长度。
        #     """
        #     batch_size, seq_len, *other_dims = hidden_states.shape
        #     device = hidden_states.device
        #
        #     # 1. 创建 unpadded_tensor (扁平化)
        #     # 这是一个 reshape 操作，通常是 CUDA Graph 友好的。
        #     # 输出形状 (batch_size * seq_len) 仅依赖于输入形状。
        #     unpadded_tensor = hidden_states.reshape(batch_size * seq_len, *other_dims)
        #
        #     # 2. 创建 cu_seqlens
        #     # 对于一个矩形批次，其中每个序列长度为 'seq_len':
        #     # cu_seqlens 将是 [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
        #     # 这可以使用 torch.arange 生成。其形状 (batch_size + 1) 是固定的。
        #     cu_seqlens = torch.arange(
        #         0, (batch_size * seq_len) + 1, seq_len, dtype=torch.int32, device=device
        #     )
        #
        #     # 3. 确定 max_seqlen
        #     # 因为所有序列长度都为 'seq_len' 且完全填充。
        #     max_seqlen = seq_len  # 这是一个 Python int，对于 CUDA Graph 是可以的。
        #
        #     # 4. 创建 indices
        #     # 这些索引将 unpadded_tensor 中的每个token映射回其在概念上的
        #     # 原始扁平化张量 (形状 (batch_size * seq_len_dim, *other_dims)) 中的位置。
        #     # 在这里, seq_len_dim 就是 seq_len。
        #     # 所以, indices 就是 0, 1, ..., (batch_size * seq_len - 1)。
        #     # 其形状 (batch_size * seq_len) 仅依赖于输入形状。
        #     indices = torch.arange(
        #         batch_size * seq_len, dtype=torch.long, device=device
        #     )
        #
        #     return unpadded_tensor, indices, cu_seqlens, max_seqlen, None

        from flash_attn.bert_padding import (
            unpad_input,  # Import moved here for clarity in diff, ideally top-level
        )

        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        # Unpad K and V using unpad_input
        # The unpad_input function expects (batch, seqlen, ...)
        # key_layer and value_layer are (batch_size, kv_seq_len, num_kv_heads, head_dim)
        # attention_mask is (batch_size, kv_seq_len)

        key_layer_unpadded, indices_k, cu_seqlens_k, max_seqlen_in_batch_k, _ = (
            unpad_input(key_layer, attention_mask)
        )
        value_layer_unpadded, _, _, _, _ = unpad_input(
            value_layer, attention_mask
        )  # Indices and seqlens are the same for K and V

        if query_length == kv_seq_len:
            # If query_length is the same as kv_seq_len, Q uses the same mask and hence same unpadding logic
            query_layer_unpadded, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, _ = (
                unpad_input(query_layer, attention_mask)
            )
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[
                :-1
            ]  # For query_length == 1, indices_q are just batch indices 0, 1, ...
            query_layer_unpadded = query_layer.squeeze(
                1
            )  # Query is already effectively unpadded
        else:
            # The -q_len: slice assumes left padding.
            attention_mask_q = attention_mask[:, -query_length:]
            query_layer_unpadded, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, _ = (
                unpad_input(query_layer, attention_mask_q)
            )

        # Ensure cu_seqlens are 1D tensors
        if cu_seqlens_q.dim() != 1:
            cu_seqlens_q = cu_seqlens_q.view(cu_seqlens_q.numel())
        if (
            cu_seqlens_k.dim() != 1
        ):  # This was from the original _get_unpad_data, unpad_input should return 1D
            cu_seqlens_k = cu_seqlens_k.view(cu_seqlens_k.numel())

        return (
            query_layer_unpadded,
            key_layer_unpadded,
            value_layer_unpadded,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        forward_batch: ForwardBatch,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
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

        def pad_input(hidden_states, indices, batch, seqlen):
            """
            Arguments:
                hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
                indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
                batch: int, batch size for the padded sequence.
                seqlen: int, maximum sequence length for the padded sequence.
            Return:
                hidden_states: (batch, seqlen, ...)
            """
            dim = hidden_states.shape[1:]
            output = torch.zeros(
                (batch * seqlen),
                *dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            output[indices] = hidden_states
            return rearrange(output, "(b s) ... -> b s ...", b=batch)

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        # print(f"{attention_mask=}")
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            if no_rope_param is not None:
                # nope unpad
                no_rope_param["query_states_no_rope"] = no_rope_param[
                    "query_states_no_rope"
                ].squeeze(0)
                no_rope_param["key_states_no_rope"] = no_rope_param[
                    "key_states_no_rope"
                ].squeeze(0)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = self.nsa_forward(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                forward_batch=forward_batch,
                no_rope_param=no_rope_param,
                past_key_value=past_key_value,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            raise ValueError("Need attention mask")

        assert isinstance(attn_output, torch.Tensor)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        attention_mask: Optional[torch.LongTensor] = None,
        positions: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCacheQKV] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Only batch_size=1 is supported at the moment."
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ! save no rope
        if self.use_nope:
            query_states_no_rope = q.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states_no_rope = k.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = q_len

        # if past_key_value is not None:
        #     usable_length = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # print(f"{usable_length=}")
        # kv_seq_len += usable_length

        # FIXME: .view() -> .reshape()
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k, forward_batch.seq_lens_cpu.item())

        q = q.view(q_shape)
        k = k.view(k_shape)
        if past_key_value is not None:
            if not forward_batch.forward_mode.is_extend():
                #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    self.layer_idx
                )
                start_out_cache_loc = (
                    forward_batch.out_cache_loc - forward_batch.seq_lens_sum + 1
                )
                # FIXME: transpose
                k = torch.cat(
                    [
                        key_cache[
                            start_out_cache_loc : forward_batch.out_cache_loc,
                            :,
                        ]
                        .view(1, -1, self.num_kv_heads, self.head_dim)
                        .transpose(1, 2),
                        k,
                    ],
                    dim=-2,
                )
                v = torch.cat(
                    [
                        value_cache[
                            start_out_cache_loc : forward_batch.out_cache_loc,
                            :,
                        ]
                        .view(1, -1, self.num_kv_heads, self.head_dim)
                        .transpose(1, 2),
                        v,
                    ],
                    dim=-2,
                )

                # print(f"{k.shape=}")

                # k, v = past_key_value.update(k, v, self.layer_idx)
                # k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_nope:
            no_rope_param = {
                "key_states_no_rope": key_states_no_rope,
                "query_states_no_rope": query_states_no_rope,
            }
        else:
            no_rope_param = None

        dropout_rate = 0.0

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
                target_dtype = next(self.qkv_proj.parameters()).dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)

        # if past_key_value is None or q_len != 1:  # prefilling
        if forward_batch.forward_mode.is_extend():  # prefilling
            attn_output = self._flash_attention_forward(
                q,
                k,
                v,
                attention_mask,
                q_len,
                forward_batch=forward_batch,
                dropout=dropout_rate,
                no_rope_param=no_rope_param,  # if past_key_value is not None else None,
                past_key_value=past_key_value,
            )
        else:
            attn_output = self._flash_attention_forward_with_kv_cache(
                q,
                k,
                v,
                attention_mask,
                q_len,
                forward_batch=forward_batch,
                dropout=dropout_rate,
                no_rope_param=no_rope_param,
                past_key_value=past_key_value,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output, _ = self.o_proj(attn_output)
        return attn_output
        # return attn_output, past_key_valuepip install flash-attn --no-build-isolation

    def _flash_attention_forward_with_kv_cache(
        self,
        q,
        k,
        v,
        attention_mask,
        query_length,
        forward_batch: ForwardBatch,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            q (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            k (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            v (`torch.Tensor`):
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

            batch_size = q.shape[0]

            # query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            #     query_states, key_states, value_states, attention_mask, query_length=query_length
            # )

            assert batch_size == 1, "Only batch_size=1 is supported at the moment."
            # prepare past kv ,new kv
            new_q = q

            new_k = k[:, -1:, :, :].contiguous()
            new_v = v[:, -1:, :, :].contiguous()

            if no_rope_param is not None:
                # nope unpad
                no_rope_param["query_states_no_rope"] = no_rope_param[
                    "query_states_no_rope"
                ].squeeze(0)
                no_rope_param["key_states_no_rope"] = no_rope_param[
                    "key_states_no_rope"
                ].squeeze(0)

            attn_output = self.nsa_forward_with_kv_cache(
                new_k=new_k,
                new_v=new_v,
                new_q=new_q,
                batch_size=batch_size,
                no_rope_param=no_rope_param,
                past_key_value=past_key_value,
                forward_batch=forward_batch,
            )
            # attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            raise ValueError("need attention mask")

        return attn_output

    def nsa_forward_with_kv_cache(
        self,
        forward_batch: ForwardBatch,
        new_k=None,
        new_v=None,
        new_q=None,
        batch_size=None,
        no_rope_param=None,
        past_key_value=None,
    ):
        seq_len = forward_batch.seq_lens_sum
        stage1_k = (
            new_k.squeeze(0)
            if no_rope_param is None
            else no_rope_param["key_states_no_rope"]
        )
        no_compress_k = past_key_value.update_no_compress_k(
            stage1_k,
            self.layer_idx,
            kernel_stride=self.kernel_stride,
            kernel_size=self.kernel_size,
        )

        if no_compress_k is not None:
            compressed_k = no_compress_k.mean(
                dim=0, keepdim=True
            )  # [1, n_heads_k, head_dim]

            compressed_k = past_key_value.update_compress_k(
                compressed_k, self.layer_idx
            )  # [seqlen, nheads_k, head_dim]

            past_key_value.cached_compressed_cu_seqlens[self.layer_idx][
                -1
            ] += 1  # ! Increment the last entry in sequence lengths by 1; currently supports only batch_size = 1
            compressed_cu_seqlens = past_key_value.cached_compressed_cu_seqlens[
                self.layer_idx
            ]
        else:
            # print(f"{self.layer_idx=}")
            # print(f"{len(past_key_value.compress_k_cache)=}")
            # in cuda-graph capturing, this cache might be uniniitalized
            compressed_k = past_key_value.get_compressed_k(
                self.layer_idx
            )  # [seqlen, nheads_k, head_dim]
            if compressed_k is None:
                compressed_k = torch.zeros(
                    (0, self.num_kv_heads, self.head_dim),
                    dtype=torch.bfloat16,
                    device=new_k.device,
                )
            # print(f"111 {compressed_k.shape=}")
            compressed_cu_seqlens = past_key_value.get_compressed_cu_seqlens(
                self.layer_idx
            ).to(device=new_k.device)
            # print(f"111 {compressed_cu_seqlens=}")

        compressed_v = compressed_k.clone()

        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]

        # Manually verify that the lengths match
        assert (
            compressed_k.shape[0] == compressed_seqlens.sum().item()
        ), "The length of compressed_k does not match the sum of compressed_seqlens"
        # if compressed_k.shape[0] == 0:
        #     # If there are no compressed keys (e.g., during CUDA graph capture for an uninitialized layer),
        #     # create a dummy topk_idx indicating no blocks are selected.
        #     # new_q has shape [bsz=1, q_len=1, num_q_heads, head_dim]
        #     # topk_idx is expected to be (num_q_heads, q_len, top_k_value)
        #     # For decode (q_len=1): (self.num_heads, 1, self.topk)
        #     num_q_heads_for_topk_idx = new_q.shape[2]  # This is self.num_heads for the current TP rank
        #     topk_idx = torch.full(
        #         (num_q_heads_for_topk_idx, 1, self.topk),
        #         -1,
        #         dtype=torch.int32,
        #         device=new_q.device
        #     )
        # else:
        topk_idx = compressed_attention(
            (
                new_q.squeeze(0).contiguous()
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"]
            ),
            compressed_k,
            compressed_v,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            torch.tensor([0, 1], device=compressed_k.device, dtype=torch.int32),
            compressed_cu_seqlens,
            1,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            total_seq_lens=seq_len,  # ! Only batch_size=1 is supported at the moment.
        )

        repeat_times = 1
        if repeat_times > 1:
            new_q = new_q.repeat_interleave(repeat_times, dim=-2)
        else:
            new_q = new_q

        cache_batch_idx = torch.arange(
            batch_size, device=new_q.device, dtype=torch.int32
        )

        seqlen_k = (
            seq_len - 1 + new_k.shape[1]
        )  # ! Only batch_size=1 is supported at the moment.
        seqlens_k = torch.full(
            (batch_size,), seqlen_k - 1, dtype=torch.int32, device=new_q.device
        )

        # past_k = torch.cat(
        #     [past_k, torch.zeros_like(new_k, dtype=new_k.dtype)], dim=1
        # ).contiguous()  # Append one zero vector to avoid potential out-of-bounds access
        # past_v = torch.cat(
        #     [past_v, torch.zeros_like(new_v, dtype=new_v.dtype)], dim=1
        # ).contiguous()  # Append one zero vector to avoid potential out-of-bounds access

        topk_attn_output = flash_attn_with_kvcache(
            attn=self.attn,
            forward_batch=forward_batch,
            q=new_q,
            # k_cache=past_k,
            # v_cache=past_v,
            topk_idx=topk_idx,
            block_window_size=self.window_size // self.block_size,
            k=new_k,  # [batch_size, 1, nheads_k, d]
            v=new_v,  # [batch_size, 1, nheads_k, d]
            cache_seqlens=seqlens_k,  # current_seqlens_k-1
            rotary_cos=None,  # No rotary embeddings
            rotary_sin=None,  # No rotary embeddings
            cache_batch_idx=cache_batch_idx,
            causal=False,  # Renaming to match function signature
        )
        return topk_attn_output


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMSparseFlashAttention2,
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
        # FIXME: for testing
        # print(f"{config._attn_implementation=}")
        # config._attn_implementation = "flash_attention_2"
        # if hasattr(config, "sparse_config") and config.sparse_config is not None:
        #     self.self_attn = MiniCPMSparseFlashAttention2(
        #         config=config, layer_id=layer_id
        #     )
        # else:
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
            config=config,
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
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )

        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
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
        self.past_key_values = DynamicCacheQKV()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        # FIXME
        print(f"{forward_batch=}")
        if forward_batch.forward_mode.is_extend():
            self.past_key_values = DynamicCacheQKV()

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb
        else:
            hidden_states = input_embeds
        residual = None

        current_total_tokens = hidden_states.shape[0]
        attention_mask = torch.ones(
            (1, current_total_tokens),
            # (1, 1, 1, current_total_tokens),
            dtype=torch.long,
            device=hidden_states.device,
        )
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                past_key_value=self.past_key_values,
                attention_mask=attention_mask,
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
        # print(f"{positions.shape=}")
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
