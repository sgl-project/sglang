"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.nsa.utils import NSA_KV_CACHE_STORE_FP8
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_bool_env_var, is_cuda, is_npu, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
if _is_npu:
    import torch_npu


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )

        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class MambaPool:
    def __init__(
        self,
        size: int,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        num_mamba_layers: int,
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        device: str,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state = torch.zeros(
            size=(num_mamba_layers, size + 1) + conv_state_shape,
            dtype=conv_dtype,
            device=device,
        )
        temporal_state = torch.zeros(
            size=(num_mamba_layers, size + 1) + temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )
        if speculative_num_draft_tokens is not None:
            # Cache intermediate SSM states per draft token during target verify
            # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
            intermediate_ssm_state_cache = torch.zeros(
                size=(
                    num_mamba_layers,
                    size + 1,
                    speculative_num_draft_tokens,
                    temporal_state_shape[0],
                    temporal_state_shape[1],
                    temporal_state_shape[2],
                ),
                dtype=ssm_dtype,
                device="cuda",
            )
            # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
            # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
            intermediate_conv_window_cache = torch.zeros(
                size=(
                    num_mamba_layers,
                    size + 1,
                    speculative_num_draft_tokens,
                    conv_state_shape[0],
                    conv_state_shape[1],
                ),
                dtype=conv_dtype,
                device="cuda",
            )
            self.mamba_cache = (
                conv_state,
                temporal_state,
                intermediate_ssm_state_cache,
                intermediate_conv_window_cache,
            )
            logger.info(
                f"Mamba Cache is allocated. "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
            )
        else:
            self.mamba_cache = (conv_state, temporal_state)
            logger.info(
                f"Mamba Cache is allocated. "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
            )
        self.size = size
        self.free_slots = list(range(size))
        self.mem_usage = self.get_mamba_size() / GB

    def get_mamba_params_all_layers(self):
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_mamba_params(self, layer_id: int):
        return [self.mamba_cache[i][layer_id] for i in range(len(self.mamba_cache))]

    def get_mamba_size(self):
        return sum(get_tensor_size_bytes(t) for t in self.mamba_cache)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[List[int]]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)
        self.mamba_cache[0][:, free_index] = self.mamba_cache[1][:, free_index] = 0

    def clear(self):
        self.free_slots = list(range(self.size))


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layers: List[int],
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        speculative_num_draft_tokens: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_pool = MambaPool(
            size,
            conv_dtype,
            ssm_dtype,
            len(mamba_layers),
            conv_state_shape,
            temporal_state_shape,
            device,
            speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layers)}

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            size, dtype=torch.int32, device=self.device
        )

        self.rid_to_mamba_index_mapping: Dict[str, int] = {}
        self.mamba_index_to_rid_mapping: Dict[int, str] = {}

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(
        self, need_size: int, reqs: Optional[List["Req"]] = None
    ) -> Optional[List[int]]:
        select_index = super().alloc(need_size)
        if select_index == None:
            return None

        mamba_index = []
        for req in reqs:
            rid = req.rid
            if rid in self.rid_to_mamba_index_mapping:
                mid = self.rid_to_mamba_index_mapping[rid]
            elif (mid := self.mamba_pool.alloc(1)) is not None:
                mid = mid[0]
                self.rid_to_mamba_index_mapping[rid] = mid
                self.mamba_index_to_rid_mapping[mid] = rid
            mamba_index.append(mid)
        assert len(select_index) == len(
            mamba_index
        ), f"Not enough space for mamba cache, try to increase --max-mamba-cache-size."
        self.req_index_to_mamba_index_mapping[select_index] = torch.tensor(
            mamba_index, dtype=torch.int32, device=self.device
        )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def get_mamba_params(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.get_mamba_params(self.mamba_map[layer_id])

    def get_mamba_params_all_layers(self):
        return self.mamba_pool.get_mamba_params_all_layers()

    # For chunk prefill, we can not free mamba cache, we need use it in the future
    def free(self, free_index: Union[int, List[int]], free_mamba_cache: bool = True):
        super().free(free_index)
        if free_mamba_cache:
            mamba_index = self.req_index_to_mamba_index_mapping[free_index]
            mamba_index_list = mamba_index.tolist()
            if isinstance(mamba_index_list, int):
                mamba_index_list = [mamba_index_list]
            self.mamba_pool.free(mamba_index_list)
            for mid in mamba_index_list:
                rid = self.mamba_index_to_rid_mapping[mid]
                self.mamba_index_to_rid_mapping.pop(mid)
                self.rid_to_mamba_index_mapping.pop(rid)

    def clear(self):
        super().clear()
        self.mamba_pool.clear()


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError()


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = get_bool_env_var(
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
        )
        if self.enable_custom_mem_pool:
            # TODO(shangming): abstract custom allocator class for more backends
            from mooncake.allocator import NVLinkAllocator

            allocator = NVLinkAllocator.get_allocator(self.device)
            self.custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
        else:
            self.custom_mem_pool = None

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None
        self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        copy_all_layer_kv_cache[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            tgt_loc,
            src_loc,
            len(tgt_loc),
            next_power_of_2(len(tgt_loc)),
        )


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        if _is_npu:
            TokenToKVPoolClass = AscendTokenToKVPool
        else:
            TokenToKVPoolClass = MHATokenToKVPool
        self.full_kv_pool = TokenToKVPoolClass(
            size=size,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.full_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        self.full_kv_pool.set_kv_buffer(
            None,
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=layer_id,
        )

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        kwargs["page_size"] = 1
        kwargs["enable_memory_saver"] = False
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            layer_num=self.swa_layer_nums,
            **kwargs,
        )
        self.full_kv_pool = token_to_kv_pool_class(
            size=size,
            layer_num=self.full_layer_nums,
            **kwargs,
        )
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        kv_data_ptrs = full_kv_data_ptrs + swa_kv_data_ptrs
        kv_data_lens = full_kv_data_lens + swa_kv_data_lens
        kv_item_lens = full_kv_item_lens + swa_kv_item_lens

        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        layer_id = layer.layer_id
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )


class AscendTokenToKVPool(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        torch_npu._npu_reshape_and_cache(
            key=cache_k,
            value=cache_v,
            key_cache=self.k_buffer[layer_id].view(
                -1, self.page_size, self.head_num, self.head_dim
            ),
            value_cache=self.v_buffer[layer_id].view(
                -1, self.page_size, self.head_num, self.head_dim
            ),
            slot_indices=loc,
        )


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = use_nsa
        # TODO do not hardcode
        self.kv_cache_dim = (
            656
            if use_nsa and NSA_KV_CACHE_STORE_FP8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        # for disagg with nvlink
        self.enable_custom_mem_pool = get_bool_env_var(
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
        )
        if self.enable_custom_mem_pool:
            # TODO(shangming): abstract custom allocator class for more backends
            from mooncake.allocator import NVLinkAllocator

            allocator = NVLinkAllocator.get_allocator(self.device)
            self.custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
        else:
            self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (size + page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if self.use_nsa and NSA_KV_CACHE_STORE_FP8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
        elif cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id - self.start_layer], loc, cache_k_nope, cache_k_rope
        )

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class NSATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_nsa=True,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128

        self.quant_block_size = 128

        assert self.page_size == 64
        self.index_k_with_scale_buffer = [
            torch.zeros(
                # Layout:
                #     ref: test_attention.py :: kv_cache_cast_to_fp8
                #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                #     data: for page i,
                #         * buf[i, :page_size * head_dim] for fp8 data
                #         * buf[i, page_size * head_dim:].view(float32) for scale
                (
                    (size + page_size + 1) // self.page_size,
                    self.page_size
                    * (index_head_dim + index_head_dim // self.quant_block_size * 4),
                ),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(layer_num)
        ]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    # TODO rename later (currently use diff name to avoid confusion)
    def set_index_k_and_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )


class AscendMLAPagedTokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.index_k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.index_k_buffer[layer_id - self.start_layer]

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[layer_id - self.start_layer].view(
                -1, 1, self.qk_rope_head_dim
            ),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id - self.start_layer].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label


@triton.jit
def copy_all_layer_kv_cache(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128

    bid = tl.program_id(0)
    stride = tl.load(strides + bid)

    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.uint8))

    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)
    src_locs = tl.load(src_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)

    # NOTE: we cannot parallelize over the tgt_loc_ptr dim with cuda blocks
    # because this copy is an inplace operation.

    num_loop = tl.cdiv(stride, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = (num_locs_offset < num_locs)[:, None] & (copy_offset < stride)[None, :]
        value = tl.load(
            data_ptr + src_locs[:, None] * stride + copy_offset[None, :], mask=mask
        )
        tl.store(
            data_ptr + tgt_locs[:, None] * stride + copy_offset[None, :],
            value,
            mask=mask,
        )
