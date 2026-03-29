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

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


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

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].is_chunked > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size))


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(cache_params.layers)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

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
                        spec_state_size + 1,
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
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        # clear at alloc time — expand a scalar GPU zero to the right shape, no CPU-GPU sync
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, select_index] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, select_index] = z

        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_index] = self.mamba_cache.conv[i][
                :, src_index
            ]
        self.mamba_cache.temporal[:, dst_index] = self.mamba_cache.temporal[
            :, src_index
        ]
        return

    def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
        dst_index = self.alloc(1)
        if dst_index is None:
            return None
        self.copy_from(src_index, dst_index)
        return dst_index

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        enable_overlap_schedule: bool = True,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # TODO: Support PP
        self.start_layer = 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        self.mamba_pool = MambaPool(
            size=size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(cache_params.layers)}

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            mid = None
            if req.mamba_pool_idx is not None:  # for radix cache
                mid = req.mamba_pool_idx
            else:
                mid = self.mamba_pool.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"
                mid = mid[0]
                req.mamba_pool_idx = mid
            mamba_indices.append(mid)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(
                        self.mamba_ping_pong_track_buffer_size
                    )
                    assert (
                        req.mamba_ping_pong_track_buffer is not None
                    ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
                    req.mamba_next_track_idx = 0
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), f"Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers).to(
                dtype=torch.int32
            )
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def free_mamba_cache(
        self, req: "Req", mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_pool.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


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

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

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

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


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
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
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
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream() if _is_cuda and enable_alt_stream else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

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
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
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

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class MHATokenToKVPoolFP4(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_v_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

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

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


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
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = 0  # TODO: Support PP
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:

            TokenToKVPoolClass = MHATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
        else:

            TokenToKVPoolClass = MLATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

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
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


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
        self.nsa_kv_cache_store_fp8 = (
            use_nsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with nsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_nsa:
            # NSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

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
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
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

        if self.nsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

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


class MLATokenToKVPoolFP4(MLATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
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

        if self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class NSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

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
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
    ):

        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

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
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128

        if _is_hip:
            assert self.page_size == 1
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (index_buf_size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
        self._finalize_allocation_log(size)

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

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
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

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


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
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

                # [size, head_num, heavy_channel_num] for each layer
                self.label_buffer = [
                    torch.zeros(
                        (size + 1, head_num, heavy_channel_num),
                        dtype=dtype,
                        device=device,
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


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)


# ========================================================================
# TurboQuant INT4 KV Cache Pool for MLA
# ========================================================================

class MLATokenToKVPoolTQ(MLATokenToKVPool):
    """MLA KV cache pool with TurboQuant compression (2/3/4-bit).

    Compresses the kv_lora_rank part of the MLA latent, keeps
    qk_rope_head_dim in FP16. Dequantizes on read.

    Bit-width set via SGLANG_KV_CACHE_TURBOQUANT env var:
      "1" or "4" → 4-bit (2.94x compression, CosSim > 0.995)
      "3"        → 3-bit (3.51x compression)
      "2"        → 2-bit (4.36x compression)

    QJL (Stage 2) enabled via SGLANG_KV_CACHE_TURBOQUANT_QJL=1:
      Adds unbiased inner product correction at the cost of extra storage
      (d/8 bytes signs + 2 bytes residual norm per token).
      Only beneficial for GQA models with head_dim <= 64 at bit_width <= 2.
      For MLA models (d=512), QJL is harmful and should be OFF (default).

    Integration with aiter backend: transparent — get_key_buffer() returns
    dequantized FP16, so aiter's mla_decode_fwd works unchanged.
    """

    def __init__(self, *args, tq_bit_width=None, **kwargs):
        self._tq_bit_width_override = tq_bit_width
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        import math, os
        from sglang.srt.layers.quantization.turboquant_engine import (
            get_codebook, generate_rotation_matrix, packed_bytes_per_dim, pad_for_packing,
        )

        # Priority: constructor arg > env var > default 4
        if self._tq_bit_width_override is not None:
            tq_val = str(self._tq_bit_width_override)
        else:
            tq_val = os.environ.get("SGLANG_KV_CACHE_TURBOQUANT", "4")
        try:
            self.tq_effective_bits = float(tq_val) if tq_val not in ("1", "true", "True") else 4.0
        except ValueError:
            self.tq_effective_bits = 4.0
        self.tq_bit_width = int(self.tq_effective_bits)
        self.tq_mixed = self.tq_effective_bits != int(self.tq_effective_bits)
        self.tq_group_size = min(128, self.kv_lora_rank)
        self.tq_n_groups = math.ceil(self.kv_lora_rank / self.tq_group_size)

        if self.tq_mixed:
            from sglang.srt.layers.quantization.turboquant_engine import mixed_bit_config
            self.tq_group_bits = mixed_bit_config(self.tq_effective_bits, self.tq_n_groups)
            logger.info(f"TurboQuant mixed-bit: {self.tq_effective_bits}-bit, per-group={self.tq_group_bits}")
        else:
            if self.tq_bit_width not in (2, 3, 4):
                logger.warning(f"Invalid TQ bit_width {self.tq_bit_width}, defaulting to 4")
                self.tq_bit_width = 4
            self.tq_group_bits = None

        self.tq_use_qjl = os.environ.get("SGLANG_KV_CACHE_TURBOQUANT_QJL", "0") == "1"
        # RoPE quantization: default ON (quantize RoPE too for max compression)
        # Set SGLANG_KV_CACHE_TURBOQUANT_ROPE=0 to keep RoPE in FP16
        self.tq_quant_rope = os.environ.get("SGLANG_KV_CACHE_TURBOQUANT_ROPE", "1") != "0"

        centroids, boundaries = get_codebook(self.tq_bit_width)
        self.tq_centroids = centroids.to(self.device)
        self.tq_boundaries = boundaries.to(self.device)

        self.tq_rotations = {}
        Pi_list = []
        for g in range(self.tq_n_groups):
            g_start = g * self.tq_group_size
            g_end = min(g_start + self.tq_group_size, self.kv_lora_rank)
            g_dim = g_end - g_start
            Pi = generate_rotation_matrix(g_dim, seed=42 + g_start).to(self.device)
            self.tq_rotations[g_start] = Pi
            Pi_list.append(Pi)
        self.tq_Pi_all = torch.stack(Pi_list)

        # RoPE rotation matrix (separate from latent groups)
        if self.tq_quant_rope and self.qk_rope_head_dim > 0:
            rope_dim = self.qk_rope_head_dim
            padded_rope = pad_for_packing(rope_dim, self.tq_bit_width)
            self.tq_rope_Pi = generate_rotation_matrix(rope_dim, seed=42 + 9999).to(self.device)
            self.tq_rope_padded = padded_rope
        else:
            self.tq_rope_Pi = None

        # QJL projection matrix (d x d Gaussian)
        if self.tq_use_qjl:
            gen = torch.Generator().manual_seed(10042)
            self.tq_S = torch.randn(
                self.kv_lora_rank, self.kv_lora_rank,
                generator=gen, dtype=torch.float32,
            ).to(self.device)
            self._tq_qjl_scale = math.sqrt(math.pi / 2) / self.kv_lora_rank
            logger.info("TurboQuant QJL (Stage 2) enabled")
        else:
            self.tq_S = None

        self._tq_gpu_kernel = None
        try:
            self._tq_gpu_kernel = self._load_tq_gpu_kernel()
        except Exception as e:
            logger.warning(f"TurboQuant GPU kernel unavailable, using Python fallback: {e}")

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                if self.tq_mixed:
                    from sglang.srt.layers.quantization.turboquant_engine import mixed_compressed_bytes
                    self.tq_compressed_bytes = mixed_compressed_bytes(
                        self.kv_lora_rank, self.tq_group_size,
                        self.qk_rope_head_dim, self.tq_group_bits, self.tq_use_qjl
                    )
                    # Compute per-group packed offsets for mixed-bit layout
                    self._tq_group_offsets = [0]
                    for g, bw in enumerate(self.tq_group_bits):
                        gs = g * self.tq_group_size
                        ge = min(gs + self.tq_group_size, self.kv_lora_rank)
                        self._tq_group_offsets.append(
                            self._tq_group_offsets[-1] + packed_bytes_per_dim(ge - gs, bw)
                        )
                    self._tq_packed_end = self._tq_group_offsets[-1]
                else:
                    packed_bytes = packed_bytes_per_dim(self.kv_lora_rank, self.tq_bit_width)
                    self._tq_packed_end = packed_bytes

                # RoPE: quantized or FP16
                if self.tq_quant_rope and self.qk_rope_head_dim > 0:
                    rope_packed = packed_bytes_per_dim(self.qk_rope_head_dim, self.tq_bit_width)
                    rope_norm = 2  # FP16 norm for rope group
                    rope_bytes = rope_packed + rope_norm
                else:
                    rope_packed = 0
                    rope_norm = 0
                    rope_bytes = self.qk_rope_head_dim * 2  # FP16

                self._tq_rope_packed = rope_packed
                self._tq_rope_norm = rope_norm

                norms_bytes = self.tq_n_groups * 2  # latent group norms
                qjl_signs_bytes = self.kv_lora_rank // 8 if self.tq_use_qjl else 0
                qjl_rnorm_bytes = 2 if self.tq_use_qjl else 0

                if not self.tq_mixed:
                    self.tq_compressed_bytes = (
                        self._tq_packed_end + norms_bytes + qjl_signs_bytes + qjl_rnorm_bytes + rope_bytes
                    )
                self._tq_norms_end = self._tq_packed_end + norms_bytes
                self._tq_signs_end = self._tq_norms_end + qjl_signs_bytes
                self._tq_rnorm_end = self._tq_signs_end + qjl_rnorm_bytes

                # Store as uint8 for maximum flexibility
                self.store_dtype = torch.uint8
                self.kv_buffer = [
                    torch.zeros(
                        (m, 1, self.tq_compressed_bytes),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self._deq_buffer = torch.zeros(
                    (m, 1, self.kv_cache_dim),
                    dtype=self.dtype,
                    device=self.device,
                )

        # O1: per-layer dirty tracking — skip decompress when nothing changed
        self._deq_dirty = [True] * self.layer_num
        # O2: track max populated slot per layer — decompress only active rows
        self._tq_active = [0] * self.layer_num

        qjl_str = "+QJL" if self.tq_use_qjl else ""
        rope_str = "+ropeQ" if self.tq_quant_rope else ""
        bits_str = f"{self.tq_effective_bits}-bit" if self.tq_mixed else f"{self.tq_bit_width}-bit"
        logger.info(
            f"TurboQuant MLA KV Pool: {bits_str}{rope_str}{qjl_str}, "
            f"{self.tq_compressed_bytes} bytes/token "
            f"(vs {self.kv_cache_dim * 2} FP16), "
            f"{self.kv_cache_dim * 2 / self.tq_compressed_bytes:.2f}x compression"
        )

    @staticmethod
    def _load_tq_gpu_kernel():
        """Try to load pre-compiled .so or JIT-compile from .hip source."""
        import importlib
        import os

        so_candidates = [
            os.path.join(os.path.dirname(__file__), "tq_kv_compress.so"),
        ]
        so_path = next((s for s in so_candidates if os.path.exists(s)), None)
        if so_path:
            spec = importlib.util.spec_from_file_location("tq_kv_compress", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.info(f"TurboQuant GPU kernel loaded from {so_path}")
            return mod

        try:
            import aiter
            aiter_root = os.path.dirname(os.path.dirname(aiter.__file__))
        except ImportError:
            return None

        src = os.path.join(aiter_root, "csrc/turboquant/turboquant_kv_compress.hip")
        if os.path.exists(src):
            from torch.utils.cpp_extension import load
            ck_inc = os.path.join(os.path.dirname(src), "../../3rdparty/composable_kernel/include")
            mod = load(
                name="tq_kv_compress", sources=[src],
                extra_include_paths=[os.path.dirname(src), ck_inc],
                extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
                verbose=False,
            )
            logger.info("TurboQuant GPU kernel JIT-compiled from aiter source")
            return mod

        return None

    def _tq_compress(self, kv_data: torch.Tensor) -> torch.Tensor:
        """Compress KV data to packed format (2/3/4-bit), optionally with QJL or mixed-bit."""
        import math
        from sglang.srt.layers.quantization.turboquant_engine import (
            pack_indices, pad_for_packing,
        )

        T = kv_data.shape[0]
        flat = kv_data.reshape(T, self.kv_cache_dim)

        # Mixed-bit path (2.5/3.5-bit outlier treatment)
        if self.tq_mixed:
            from sglang.srt.layers.quantization.turboquant_engine import mixed_compress_latent
            flat_f = flat.float()
            latent = flat_f[:, :self.kv_lora_rank]
            rope = flat_f[:, self.kv_lora_rank:]

            all_packed, norms_tensor, _ = mixed_compress_latent(
                latent, self.tq_group_bits, self.tq_group_size,
                self.tq_rotations, flat.device,
            )

            result = torch.zeros(T, 1, self.tq_compressed_bytes, dtype=torch.uint8, device=flat.device)
            for g, packed in enumerate(all_packed):
                off_start = self._tq_group_offsets[g]
                off_end = self._tq_group_offsets[g + 1]
                result[:, 0, off_start:off_end] = packed
            result[:, 0, self._tq_packed_end:self._tq_norms_end] = (
                norms_tensor.view(torch.uint8).reshape(T, -1)
            )
            result[:, 0, self._tq_rnorm_end:] = (
                rope.half().contiguous().view(torch.uint8).reshape(T, -1)
            )
            return result

        if self._tq_gpu_kernel is not None and not self.tq_use_qjl:
            flat_bf16 = flat.to(torch.bfloat16) if flat.dtype != torch.bfloat16 else flat
            buf = torch.empty(T, self.tq_compressed_bytes, dtype=torch.uint8, device=flat.device)
            self._tq_gpu_kernel.turboquant_kv_compress_inplace(
                flat_bf16, self.tq_Pi_all, self.tq_boundaries,
                buf, self.tq_n_groups, self.tq_group_size, self.tq_bit_width,
            )
            return buf.unsqueeze(1)

        flat = flat.float()
        latent = flat[:, :self.kv_lora_rank]
        rope = flat[:, self.kv_lora_rank:]
        n_levels = 2 ** self.tq_bit_width

        all_indices = []
        all_norms = []
        latent_mse = torch.zeros_like(latent) if self.tq_use_qjl else None

        for g in range(self.tq_n_groups):
            g_start = g * self.tq_group_size
            g_end = min(g_start + self.tq_group_size, self.kv_lora_rank)
            g_dim = g_end - g_start

            L_g = latent[:, g_start:g_end]
            norms = L_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
            L_norm = L_g / norms
            all_norms.append(norms.squeeze(1))

            Pi = self.tq_rotations[g_start]
            Y = L_norm @ Pi.T * math.sqrt(g_dim)

            indices = torch.searchsorted(self.tq_boundaries, Y.reshape(-1))
            indices = indices.clamp(0, n_levels - 1).reshape(T, g_dim)
            all_indices.append(indices)

            if self.tq_use_qjl:
                Y_hat = self.tq_centroids[indices] / math.sqrt(g_dim)
                latent_mse[:, g_start:g_end] = (Y_hat @ Pi) * norms

        full_indices = torch.cat(all_indices, dim=1)
        norms_tensor = torch.stack(all_norms, dim=1).half()

        padded = pad_for_packing(self.kv_lora_rank, self.tq_bit_width)
        if padded > self.kv_lora_rank:
            full_indices = torch.nn.functional.pad(
                full_indices, (0, padded - self.kv_lora_rank), value=0
            )
        packed = pack_indices(full_indices, self.tq_bit_width)

        result = torch.zeros(T, 1, self.tq_compressed_bytes, dtype=torch.uint8, device=kv_data.device)
        result[:, 0, :self._tq_packed_end] = packed
        result[:, 0, self._tq_packed_end:self._tq_norms_end] = (
            norms_tensor.view(torch.uint8).reshape(T, -1)
        )

        if self.tq_use_qjl:
            residual = latent - latent_mse
            r_norm = residual.norm(dim=1)  # (T,)
            projected = residual @ self.tq_S.T  # (T, d)
            # Bit-pack signs: 1 bit per dim, 8 dims per byte
            sign_bits = (projected >= 0).to(torch.uint8)  # (T, d)
            d = self.kv_lora_rank
            sign_bytes = torch.zeros(T, d // 8, dtype=torch.uint8, device=kv_data.device)
            for bit in range(8):
                sign_bytes |= sign_bits[:, bit::8] << bit
            result[:, 0, self._tq_norms_end:self._tq_signs_end] = sign_bytes
            # Residual norm as FP16
            r_norm_h = r_norm.half()
            result[:, 0, self._tq_signs_end:self._tq_rnorm_end] = (
                r_norm_h.view(torch.uint8).reshape(T, -1)
            )

        # RoPE: quantize or store FP16
        if self.tq_quant_rope and self.tq_rope_Pi is not None:
            rope_f = rope if rope.dtype == torch.float32 else rope.float()
            rope_norms = rope_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
            rope_norm_val = rope_f / rope_norms
            rope_Y = rope_norm_val @ self.tq_rope_Pi.T * math.sqrt(self.qk_rope_head_dim)
            rope_idx = torch.searchsorted(self.tq_boundaries, rope_Y.reshape(-1))
            rope_idx = rope_idx.clamp(0, n_levels - 1).reshape(T, self.qk_rope_head_dim)
            padded_r = pad_for_packing(self.qk_rope_head_dim, self.tq_bit_width)
            if padded_r > self.qk_rope_head_dim:
                rope_idx = torch.nn.functional.pad(rope_idx, (0, padded_r - self.qk_rope_head_dim), value=0)
            rope_packed = pack_indices(rope_idx, self.tq_bit_width)
            rope_norm_h = rope_norms.squeeze(1).half()
            off = self._tq_rnorm_end
            result[:, 0, off:off + self._tq_rope_packed] = rope_packed
            result[:, 0, off + self._tq_rope_packed:off + self._tq_rope_packed + self._tq_rope_norm] = (
                rope_norm_h.view(torch.uint8).reshape(T, -1)
            )
        else:
            result[:, 0, self._tq_rnorm_end:] = (
                rope.half().contiguous().view(torch.uint8).reshape(T, -1)
            )

        return result

    def _tq_decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress packed data back to FP16/BF16 (2/3/4-bit), with optional QJL or mixed-bit."""
        import math
        from sglang.srt.layers.quantization.turboquant_engine import (
            unpack_indices, pad_for_packing,
        )

        T = compressed.shape[0]

        # Mixed-bit path
        if self.tq_mixed:
            from sglang.srt.layers.quantization.turboquant_engine import mixed_decompress_latent
            all_packed = []
            for g in range(self.tq_n_groups):
                off_s = self._tq_group_offsets[g]
                off_e = self._tq_group_offsets[g + 1]
                all_packed.append(compressed[:, 0, off_s:off_e])
            norms_raw = compressed[:, 0, self._tq_packed_end:self._tq_norms_end]
            rope_raw = compressed[:, 0, self._tq_rnorm_end:]
            norms = norms_raw.view(torch.float16).reshape(T, self.tq_n_groups)
            rope = rope_raw.view(torch.float16).reshape(T, self.qk_rope_head_dim)
            latent = mixed_decompress_latent(
                all_packed, norms, self.tq_group_bits, self.tq_group_size,
                self.kv_lora_rank, self.tq_rotations, compressed.device,
            )
            kv_out = torch.cat([latent.to(self.dtype), rope.to(self.dtype)], dim=-1)
            return kv_out.reshape(T, 1, self.kv_cache_dim)

        if self._tq_gpu_kernel is not None and not self.tq_use_qjl:
            flat = compressed[:, 0, :]
            out = self._deq_buffer[:T, 0, :].reshape(T, self.kv_cache_dim)
            self._tq_gpu_kernel.turboquant_kv_decompress_inplace(
                flat, self.tq_Pi_all, self.tq_centroids,
                out, self.tq_n_groups, self.tq_group_size,
                self.kv_lora_rank, self.qk_rope_head_dim, self.tq_bit_width,
            )
            return self._deq_buffer[:T]

        packed = compressed[:, 0, :self._tq_packed_end]
        norms_raw = compressed[:, 0, self._tq_packed_end:self._tq_norms_end]

        norms = norms_raw.view(torch.float16).reshape(T, self.tq_n_groups).float()

        # Decompress RoPE
        rope_start = self._tq_rnorm_end
        if self.tq_quant_rope and self.tq_rope_Pi is not None:
            rope_packed = compressed[:, 0, rope_start:rope_start + self._tq_rope_packed]
            rope_norm_raw = compressed[:, 0, rope_start + self._tq_rope_packed:rope_start + self._tq_rope_packed + self._tq_rope_norm]
            rope_norms = rope_norm_raw.view(torch.float16).reshape(T).float()
            padded_r = pad_for_packing(self.qk_rope_head_dim, self.tq_bit_width)
            rope_idx = unpack_indices(rope_packed, padded_r, self.tq_bit_width)[:, :self.qk_rope_head_dim]
            rope_Y = self.tq_centroids[rope_idx.long()] / math.sqrt(self.qk_rope_head_dim)
            rope = ((rope_Y @ self.tq_rope_Pi) * rope_norms.unsqueeze(1)).to(self.dtype)
        else:
            rope_raw = compressed[:, 0, rope_start:]
            rope = rope_raw.view(torch.float16).reshape(T, self.qk_rope_head_dim)

        padded = pad_for_packing(self.kv_lora_rank, self.tq_bit_width)
        indices = unpack_indices(packed, padded, self.tq_bit_width)[:, :self.kv_lora_rank]

        latent = torch.zeros(T, self.kv_lora_rank, dtype=torch.float32, device=compressed.device)
        for g in range(self.tq_n_groups):
            g_start = g * self.tq_group_size
            g_end = min(g_start + self.tq_group_size, self.kv_lora_rank)
            g_dim = g_end - g_start
            scale = math.sqrt(g_dim)

            Pi = self.tq_rotations[g_start]
            Y_g = self.tq_centroids[indices[:, g_start:g_end].long()] / scale
            L_g = Y_g @ Pi

            if norms.dim() == 1:
                L_g = L_g * norms.unsqueeze(1)
            else:
                L_g = L_g * norms[:, g].unsqueeze(1)

            latent[:, g_start:g_end] = L_g

        # QJL correction: k = k_mse + ‖r‖ · √(π/2)/d · S^T · signs
        if self.tq_use_qjl:
            d = self.kv_lora_rank
            sign_bytes = compressed[:, 0, self._tq_norms_end:self._tq_signs_end]
            rnorm_raw = compressed[:, 0, self._tq_signs_end:self._tq_rnorm_end]

            # Unpack bit-packed signs to {-1, +1} float
            signs = torch.zeros(T, d, dtype=torch.float32, device=compressed.device)
            for bit in range(8):
                bit_vals = ((sign_bytes >> bit) & 1).float() * 2 - 1  # {0,1} → {-1,+1}
                signs[:, bit::8] = bit_vals

            r_norm = rnorm_raw.view(torch.float16).reshape(T).float()

            # QJL dequant: √(π/2)/d · ‖r‖ · S^T · signs
            qjl_correction = self._tq_qjl_scale * (signs @ self.tq_S) * r_norm.unsqueeze(1)
            latent = latent + qjl_correction

        kv_out = torch.cat([latent.to(self.dtype), rope.to(self.dtype)], dim=-1)
        return kv_out.reshape(T, 1, self.kv_cache_dim)

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        layer_idx = layer_id - self.start_layer
        if self._deq_dirty[layer_idx]:
            # O2: only decompress rows 0:active_count, not the full pool
            n = self._tq_active[layer_idx]
            if n > 0:
                compressed = self.kv_buffer[layer_idx][:n]
                decompressed = self._tq_decompress(compressed)
                self._deq_buffer[:n] = decompressed
            self._deq_dirty[layer_idx] = False
        return self._deq_buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        key_buf = self.get_key_buffer(layer_id)
        return key_buf[..., :self.kv_lora_rank]

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        compressed = self._tq_compress(cache_k.unsqueeze(1) if cache_k.dim() == 2 else cache_k)
        self.kv_buffer[layer_idx][loc] = compressed
        self._deq_dirty[layer_idx] = True
        # O2: track max populated slot
        if loc.numel() > 0:
            max_loc = loc.max().item() + 1
            if max_loc > self._tq_active[layer_idx]:
                self._tq_active[layer_idx] = max_loc

    def set_mla_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        if cache_k_nope.dim() == 2:
            cache_k_nope = cache_k_nope.unsqueeze(1)
        if cache_k_rope.dim() == 2:
            cache_k_rope = cache_k_rope.unsqueeze(1)
        kv_full = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
        compressed = self._tq_compress(kv_full)
        self.kv_buffer[layer_idx][loc] = compressed
        self._deq_dirty[layer_idx] = True
        if loc.numel() > 0:
            max_loc = loc.max().item() + 1
            if max_loc > self._tq_active[layer_idx]:
                self._tq_active[layer_idx] = max_loc


# ========================================================================
# TurboQuant KV Cache Pool for GQA/MHA
# ========================================================================

class MHATokenToKVPoolTQ(MHATokenToKVPool):
    """GQA/MHA KV cache pool with TurboQuant compression (2/3/4-bit).

    Compresses K and V per-head independently using TurboQuant.
    Each head's head_dim-dimensional vector is quantized separately.

    Args:
        tq_bit_width: 2, 3, or 4. If None, reads from SGLANG_KV_CACHE_TURBOQUANT env var.

    Supports all standard attention architectures:
      - MHA (Multi-Head Attention): head_num = num_attention_heads
      - GQA (Grouped Query Attention): head_num = num_kv_heads
      - MQA (Multi-Query Attention): head_num = 1

    Activate with: export SGLANG_KV_CACHE_TURBOQUANT=<bit_width>
    where bit_width is 2, 3, or 4.

    Integration: transparent — get_key_buffer/get_value_buffer return
    dequantized FP16, so attention backends work unchanged.
    """

    def __init__(self, *args, tq_bit_width=None, **kwargs):
        self._tq_bit_width_override = tq_bit_width
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        import math, os
        from sglang.srt.layers.quantization.turboquant_engine import (
            get_codebook, generate_rotation_matrix, packed_bytes_per_dim,
        )

        if self._tq_bit_width_override is not None:
            self.tq_bit_width = self._tq_bit_width_override
        else:
            tq_val = os.environ.get("SGLANG_KV_CACHE_TURBOQUANT", "4")
            try:
                self.tq_bit_width = int(float(tq_val)) if tq_val not in ("1", "true", "True") else 4
            except ValueError:
                self.tq_bit_width = 4
        if self.tq_bit_width not in (2, 3, 4):
            self.tq_bit_width = 4

        centroids, boundaries = get_codebook(self.tq_bit_width)
        self.tq_centroids = centroids.to(self.device)
        self.tq_boundaries = boundaries.to(self.device)

        # One rotation matrix per head_dim (shared across all heads and layers)
        self.tq_Pi = generate_rotation_matrix(self.head_dim, seed=42).to(self.device)
        self.tq_scale = math.sqrt(self.head_dim)

        pb = packed_bytes_per_dim(self.head_dim, self.tq_bit_width)
        self.tq_packed_per_head = pb
        # Per head: packed indices + 1 FP16 norm = pb + 2 bytes
        self.tq_bytes_per_head = pb + 2

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                # Store compressed K and V as uint8 flat buffers
                # Each: (m, head_num * tq_bytes_per_head)
                self.k_buffer = [
                    torch.zeros((m, self.head_num * self.tq_bytes_per_head),
                                dtype=torch.uint8, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros((m, self.head_num * self.tq_bytes_per_head),
                                dtype=torch.uint8, device=self.device)
                    for _ in range(self.layer_num)
                ]

                # Decompressed FP16 buffers for attention
                self._k_deq = [
                    torch.zeros((m, self.head_num, self.head_dim),
                                dtype=self.dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self._v_deq = [
                    torch.zeros((m, self.head_num, self.v_head_dim),
                                dtype=self.dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]

        self._deq_dirty_k = [True] * self.layer_num
        self._deq_dirty_v = [True] * self.layer_num
        self._tq_active = [0] * self.layer_num

        # For data_ptrs / data_strides (used by kv copy kernels)
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self._k_deq], dtype=torch.uint64, device=self.device)
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self._v_deq], dtype=torch.uint64, device=self.device)
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs])
        import numpy as np
        self.data_strides = torch.tensor(
            [np.prod(x.shape[1:]) * x.dtype.itemsize for x in self._k_deq + self._v_deq],
            device=self.device,
        )
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

        orig_k = self.head_num * self.head_dim * 2  # FP16
        comp_k = self.head_num * self.tq_bytes_per_head
        logger.info(
            f"TurboQuant MHA KV Pool: {self.tq_bit_width}-bit, "
            f"head_num={self.head_num}, head_dim={self.head_dim}, "
            f"{comp_k} bytes/token K (vs {orig_k} FP16), "
            f"{orig_k / comp_k:.2f}x compression"
        )

    def _compress_heads(self, data: torch.Tensor, dim: int) -> torch.Tensor:
        """Compress (T, head_num, dim) -> (T, head_num * tq_bytes_per_head) uint8."""
        import math
        from sglang.srt.layers.quantization.turboquant_engine import (
            pack_indices, pad_for_packing,
        )

        T = data.shape[0]
        n_levels = 2 ** self.tq_bit_width
        result = torch.zeros(T, self.head_num * self.tq_bytes_per_head,
                             dtype=torch.uint8, device=data.device)

        for h in range(self.head_num):
            head_data = data[:, h, :dim].float()  # (T, dim)
            norms = head_data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            head_norm = head_data / norms

            Y = head_norm @ self.tq_Pi.T * self.tq_scale
            indices = torch.searchsorted(self.tq_boundaries, Y.reshape(-1))
            indices = indices.clamp(0, n_levels - 1).reshape(T, dim)

            padded = pad_for_packing(dim, self.tq_bit_width)
            if padded > dim:
                indices = torch.nn.functional.pad(indices, (0, padded - dim), value=0)
            packed = pack_indices(indices, self.tq_bit_width)

            off = h * self.tq_bytes_per_head
            result[:, off:off + self.tq_packed_per_head] = packed
            norms_h = norms.squeeze(1).half()
            result[:, off + self.tq_packed_per_head:off + self.tq_bytes_per_head] = (
                norms_h.view(torch.uint8).reshape(T, 2)
            )

        return result

    def _decompress_heads(self, compressed: torch.Tensor, dim: int,
                          out: torch.Tensor, n_active: int):
        """Decompress (n_active, head_num * bytes) -> out[:n_active, head_num, dim]."""
        import math
        from sglang.srt.layers.quantization.turboquant_engine import (
            unpack_indices, pad_for_packing,
        )

        if n_active <= 0:
            return

        comp = compressed[:n_active]

        for h in range(self.head_num):
            off = h * self.tq_bytes_per_head
            packed = comp[:, off:off + self.tq_packed_per_head]
            norms_raw = comp[:, off + self.tq_packed_per_head:off + self.tq_bytes_per_head]

            norms = norms_raw.view(torch.float16).reshape(n_active).float()

            padded = pad_for_packing(dim, self.tq_bit_width)
            indices = unpack_indices(packed, padded, self.tq_bit_width)[:, :dim]

            Y_hat = self.tq_centroids[indices.long()] / self.tq_scale
            L = (Y_hat @ self.tq_Pi) * norms.unsqueeze(1)

            out[:n_active, h, :dim] = L.to(out.dtype)

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        layer_idx = layer_id - self.start_layer
        if self._deq_dirty_k[layer_idx]:
            n = self._tq_active[layer_idx]
            self._decompress_heads(
                self.k_buffer[layer_idx], self.head_dim,
                self._k_deq[layer_idx], n)
            self._deq_dirty_k[layer_idx] = False
        return self._k_deq[layer_idx]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        layer_idx = layer_id - self.start_layer
        if self._deq_dirty_v[layer_idx]:
            n = self._tq_active[layer_idx]
            self._decompress_heads(
                self.v_buffer[layer_idx], self.v_head_dim,
                self._v_deq[layer_idx], n)
            self._deq_dirty_v[layer_idx] = False
        return self._v_deq[layer_idx]

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None, v_scale=None, layer_id_override=None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        layer_idx = layer_id - self.start_layer

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        k_comp = self._compress_heads(cache_k, self.head_dim)
        v_comp = self._compress_heads(cache_v, self.v_head_dim)

        self.k_buffer[layer_idx][loc] = k_comp
        self.v_buffer[layer_idx][loc] = v_comp
        self._deq_dirty_k[layer_idx] = True
        self._deq_dirty_v[layer_idx] = True

        if loc.numel() > 0:
            max_loc = loc.max().item() + 1
            if max_loc > self._tq_active[layer_idx]:
                self._tq_active[layer_idx] = max_loc
