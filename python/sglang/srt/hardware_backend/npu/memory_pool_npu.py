import logging
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

logger = logging.getLogger(__name__)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    DSA_KV_QUANT_TILE_SIZE,
    get_dsa_fp8_packed_cache_dim,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu


def _init_npu_conv_state(
    conv_state_in,
    conv_state_shape,
    speculative_num_draft_tokens: Optional[int] = None,
    is_kda: bool = False,
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # Mamba shapes are (channels, window), while KDA shapes are
    # (window, channels). NPU kernels consume KDA state as
    # [layers, pool, channels, window] and other Mamba state as
    # [layers, pool, window, channels]. KDA keeps the base window fixed;
    # speculative per-step windows live in the intermediate cache.
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                conv_shape[1] if is_kda else conv_shape[1] + extra_conv_len,
                conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):

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
        **kwargs,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.use_triton_prefix_kv_cache_store = (
            envs.SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE.get()
        )
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
            **kwargs,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self.v_buffer[i].view(-1, 1, self.head_num, self.v_head_dim)
                    for i in range(self.layer_num)
                ]

    def _init_kv_copy_and_warmup(self):
        # implementation relies on self.data_strides / self.data_ptrs, which the
        # NPU paged buffer layout never builds.
        self._kv_copy_config = None

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
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
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
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
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

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
            num_rows = loc.numel()
            expected_k_numel = num_rows * self.head_num * self.head_dim
            expected_v_numel = num_rows * self.head_num * self.v_head_dim
            if (
                cache_k.numel() != expected_k_numel
                or cache_v.numel() != expected_v_numel
            ):
                raise ValueError(
                    "NPU FIA KV scatter row mismatch: "
                    f"loc_rows={num_rows}, cache_k_shape={tuple(cache_k.shape)}, "
                    f"cache_v_shape={tuple(cache_v.shape)}, "
                    f"head_num={self.head_num}, head_dim={self.head_dim}, "
                    f"v_head_dim={self.v_head_dim}."
                )

            # aclnnScatterNdUpdate on the deployed CANN rejects the otherwise
            # valid 4-D [slot, 1, head, dim] update during tiling. Flatten only
            # the singleton FIA layout axis and scatter through an equivalent
            # 3-D view; the underlying KV storage and attention layout stay
            # unchanged.
            loc_indices = loc.contiguous().view(-1, 1)
            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer.view(-1, self.head_num, self.head_dim),
                loc_indices,
                cache_k.contiguous().view(num_rows, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
                loc_indices,
                cache_v.contiguous().view(num_rows, self.head_num, self.v_head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.v_head_dim
                ),
                slot_indices=loc,
            )

    def set_kv_buffer_prefix_valid(
        self,
        layer: "RadixAttention",
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if not self.use_triton_prefix_kv_cache_store:
            return super().set_kv_buffer_prefix_valid(
                layer,
                loc_2d,
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override,
            )

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got {tuple(loc_2d.shape)}")

        num_rows = loc_2d.numel()
        if (
            cache_k.numel() != num_rows * self.head_num * self.head_dim
            or cache_v.numel() != num_rows * self.head_num * self.v_head_dim
        ):
            raise ValueError(
                "dense NPU KV rows must match loc_2d size: "
                f"cache_k={tuple(cache_k.shape)}, cache_v={tuple(cache_v.shape)}, "
                f"loc_2d={tuple(loc_2d.shape)}"
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
        v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
        if loc_2d.device != k_buffer_layer.device:
            loc_2d = loc_2d.to(device=k_buffer_layer.device, non_blocking=True)
        if commit_lens.device != k_buffer_layer.device:
            commit_lens = commit_lens.to(
                device=k_buffer_layer.device, non_blocking=True
            )
        self._debug_prefix_valid_backend = "npu_triton"
        from sgl_kernel_npu.mem_cache.kv_cache_store import (
            store_kv_cache_prefix_valid_npu_triton,
        )

        store_kv_cache_prefix_valid_npu_triton(
            k_buffer_layer.view(-1, self.head_num, self.head_dim),
            v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
            cache_k.reshape(num_rows, self.head_num, self.head_dim),
            cache_v.reshape(num_rows, self.head_num, self.v_head_dim),
            loc_2d,
            commit_lens,
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices, mamba_indices=None):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMHATokenToKOnlyPool(MHATokenToKOnlyPool):
    """NPU paged K-only cache used by MiniMax sparse index-only layers."""

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
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super(MHATokenToKOnlyPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            if self.use_fia:
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]

        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        k_buffer = self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return k_buffer.view(self.dtype)
        return k_buffer

    def set_k_buffer(
        self,
        layer_id: int,
        loc_info,
        cache_k: torch.Tensor,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer].view(
            -1, self.head_num, self.head_dim
        )
        loc = loc.to(device=cache_k.device, dtype=torch.int32).contiguous()
        torch_npu.npu_scatter_nd_update_(
            k_buffer_layer,
            loc.view(-1, 1),
            cache_k.contiguous().view(-1, self.head_num, self.head_dim),
        )

    def get_contiguous_buf_infos(self):
        data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        return get_tensor_size_bytes(self.k_buffer), 0


class NPUMiniMaxSparseKVPool(MiniMaxSparseKVPool):
    """MiniMax sparse wrapper backed by NPU paged MHA/index pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            main_pool_cls=NPUMHATokenToKVPool,
            index_kv_pool_cls=NPUMHATokenToKVPool,
            index_k_pool_cls=NPUMHATokenToKOnlyPool,
            **kwargs,
        )

    def get_index_k_state_buf_infos(self):
        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.get_key_buffer(i).data_ptr() for i in range(n)]
        data_lens = [pool.get_key_buffer(i).nbytes for i in range(n)]
        if pool.use_fia:
            item_lens = [
                pool.get_key_buffer(i)[0].nbytes * pool.page_size for i in range(n)
            ]
        else:
            item_lens = [pool.get_key_buffer(i)[0].nbytes for i in range(n)]
        return data_ptrs, data_lens, item_lens


class NPUMLATokenToKVPool(MLATokenToKVPool):

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
        index_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        indexer_layer_ids: Optional[Sequence[int]] = None,
        enable_npu_quant_lightning_indexer: bool = False,
        kv_cache_dim: Optional[int] = None,
        selective_host_layer_ids: Optional[Sequence[int]] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        if self.index_head_dim is None:
            if indexer_layer_ids:
                raise ValueError(
                    "indexer_layer_ids must be empty when index_head_dim is None"
                )
            resolved_indexer_layer_ids = ()
        elif indexer_layer_ids is None:
            # Keep the legacy uniform layout for callers and transfer paths
            # that do not provide explicit logical-layer metadata.
            resolved_indexer_layer_ids = tuple(
                range(self.start_layer, self.start_layer + self.layer_num)
            )
        else:
            resolved_indexer_layer_ids = tuple(indexer_layer_ids)
            if len(set(resolved_indexer_layer_ids)) != len(resolved_indexer_layer_ids):
                raise ValueError(
                    "indexer_layer_ids must not contain duplicates: "
                    f"{resolved_indexer_layer_ids}"
                )
            if resolved_indexer_layer_ids != tuple(sorted(resolved_indexer_layer_ids)):
                raise ValueError(
                    "indexer_layer_ids must be in increasing logical-layer order: "
                    f"{resolved_indexer_layer_ids}"
                )
            layer_end = self.start_layer + self.layer_num
            invalid_layer_ids = [
                layer_id
                for layer_id in resolved_indexer_layer_ids
                if layer_id < self.start_layer or layer_id >= layer_end
            ]
            if invalid_layer_ids:
                raise ValueError(
                    "indexer_layer_ids must be absolute layer ids in the local "
                    f"stage range [{self.start_layer}, {layer_end}), got "
                    f"{invalid_layer_ids}"
                )

        if enable_npu_quant_lightning_indexer:
            if self.index_head_dim != 128:
                raise ValueError(
                    "npu_quant_lightning_indexer requires index_head_dim=128"
                )
            if dtype != torch.float8_e4m3fn:
                raise ValueError(
                    "npu_quant_lightning_indexer requires an FP8 E4M3 KV cache"
                )
        self.indexer_layer_ids = resolved_indexer_layer_ids
        self.num_indexer_layers = len(self.indexer_layer_ids)
        self.indexer_layer_id_to_slot = {
            layer_id: slot for slot, layer_id in enumerate(self.indexer_layer_ids)
        }
        self.enable_npu_quant_lightning_indexer = enable_npu_quant_lightning_indexer
        self.dsa_kv_cache_store_fp8 = (
            enable_npu_quant_lightning_indexer
            and kv_cache_dim is not None
            and kv_cache_dim != kv_lora_rank + qk_rope_head_dim
        )
        self.kv_cache_dim = (
            kv_cache_dim if self.dsa_kv_cache_store_fp8 else kv_lora_rank
        )
        self.kr_cache_dim = 0 if self.dsa_kv_cache_store_fp8 else qk_rope_head_dim
        self.k_store_dtype = self.store_dtype
        self.v_store_dtype = self.store_dtype
        if self.dsa_kv_cache_store_fp8:
            expected_cache_dim = get_dsa_fp8_packed_cache_dim(
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
            if self.kv_cache_dim != expected_cache_dim:
                raise ValueError(
                    f"Unexpected packed DSA KV width {self.kv_cache_dim}; "
                    f"expected {expected_cache_dim}."
                )
            self.k_store_dtype = torch.float8_e4m3fn
            self.v_store_dtype = torch.bfloat16

        self.custom_mem_pool = None

        # === Selective HiSparse: resident/selected layer split ===
        local_layer_end = self.start_layer + self.layer_num
        self.selective_host_layer_ids: frozenset[int] = (
            frozenset(
                layer_id
                for layer_id in selective_host_layer_ids
                if self.start_layer <= layer_id < local_layer_end
            )
            if selective_host_layer_ids
            else frozenset()
        )
        self.selective_coordinator = None  # set later by ModelRunner

        if self.selective_host_layer_ids:
            # Build resident layer mapping (only non-selected layers get HBM buffers)
            all_layer_ids = list(range(self.start_layer, self.start_layer + layer_num))
            self.resident_layer_ids = tuple(
                lid for lid in all_layer_ids if lid not in self.selective_host_layer_ids
            )
            self.resident_layer_to_slot = {
                lid: slot for slot, lid in enumerate(self.resident_layer_ids)
            }
            num_main_slots = len(self.resident_layer_ids)
            logger.info(
                f"NPUMLATokenToKVPool selective split: {layer_num} total layers, "
                f"{num_main_slots} resident (HBM), "
                f"{len(self.selective_host_layer_ids)} selected (Host DRAM)"
            )
        else:
            self.resident_layer_ids = tuple(range(self.start_layer, self.start_layer + layer_num))
            self.resident_layer_to_slot = {
                lid: slot for slot, lid in enumerate(self.resident_layer_ids)
            }
            num_main_slots = layer_num

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = torch.zeros(
                (
                    num_main_slots,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_cache_dim,
                ),
                dtype=self.k_store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    num_main_slots,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kr_cache_dim,
                ),
                dtype=self.v_store_dtype,
                device=self.device,
            )
            self.index_k_buffer = None
            self.index_k_scale_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        self.num_indexer_layers,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.k_store_dtype,
                    device=self.device,
                )
                if self.enable_npu_quant_lightning_indexer:
                    self.index_k_scale_buffer = torch.zeros(
                        (
                            self.num_indexer_layers,
                            self.size // self.page_size + 1,
                            self.page_size,
                            1,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )

        self._finalize_allocation_log(size)

    def _resident_slot(self, layer_id: int) -> int:
        """Return the k_buffer/v_buffer slot for a resident layer."""
        try:
            return self.resident_layer_to_slot[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"Layer {layer_id} is not a resident (HBM) layer; "
                f"it may be a selected (Host DRAM) layer. "
                f"Resident layers: {self.resident_layer_ids}"
            ) from exc

    def is_selective_host_layer(self, layer_id: int) -> bool:
        return layer_id in self.selective_host_layer_ids

    def get_pd_target_meta(self):
        """Return metadata for PD transfer descriptor."""
        component_kinds: list[str] = []
        logical_layer_ids: list[int] = []
        memory_kinds: list[str] = []
        for lid in range(self.start_layer, self.start_layer + self.layer_num):
            if lid in self.selective_host_layer_ids:
                component_kinds.append("MAIN_K")
                logical_layer_ids.append(lid)
                memory_kinds.append("DRAM")
            else:
                component_kinds.append("MAIN_K")
                logical_layer_ids.append(lid)
                memory_kinds.append("VRAM")
        if self.index_head_dim is not None:
            for lid in self.indexer_layer_ids:
                component_kinds.append("INDEX_K")
                logical_layer_ids.append(lid)
                memory_kinds.append("VRAM")
                if self.index_k_scale_buffer is not None:
                    component_kinds.append("INDEX_SCALE")
                    logical_layer_ids.append(lid)
                    memory_kinds.append("VRAM")
        return {
            "component_kinds": component_kinds,
            "logical_layer_ids": logical_layer_ids,
            "memory_kinds": memory_kinds,
        }

    def drain_async_writes_before_release(self, req=None):
        """Wait for all pending selective D2H writes before releasing locs."""
        if self.selective_coordinator is not None:
            self.selective_coordinator.drain_all()

    def drain_async_writes_before_retract(self, req=None):
        """Wait for pending selective D2H writes before retraction."""
        if self.selective_coordinator is not None:
            self.selective_coordinator.drain_all()

    def _get_indexer_slot(self, layer_id: int) -> int:
        if self.index_head_dim is None:
            raise RuntimeError("This KV pool does not have an Indexer cache")
        try:
            return self.indexer_layer_id_to_slot[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"Layer {layer_id} is not a physical Indexer layer; configured "
                f"Indexer layers: {self.indexer_layer_ids}"
            ) from exc

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
            if self.index_k_scale_buffer is not None:
                for index_k_scale_cache in self.index_k_scale_buffer:
                    kv_size_bytes += get_tensor_size_bytes(index_k_scale_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.is_selective_host_layer(layer_id):
            # Selected layers: return dummy scratch buffer for npu_kv_rmsnorm_rope_cache
            # write target. Actual KV is handled via selective hisparse path.
            if not hasattr(self, "_selective_scratch_k"):
                self._selective_scratch_k = torch.zeros(
                    1, self.size // self.page_size + 1,
                    self.page_size, 1, self.kv_cache_dim,
                    dtype=self.k_store_dtype, device=self.device,
                )
                self._selective_scratch_v = torch.zeros(
                    1, self.size // self.page_size + 1,
                    self.page_size, 1, self.kr_cache_dim,
                    dtype=self.v_store_dtype, device=self.device,
                )
            return self._selective_scratch_k[0], self._selective_scratch_v[0]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        slot = self._resident_slot(layer_id)
        return (
            self.k_buffer[slot],
            self.v_buffer[slot],
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        data_ptrs = [
            self.index_k_buffer[i].data_ptr() for i in range(self.num_indexer_layers)
        ]
        data_lens = [
            self.index_k_buffer[i].nbytes for i in range(self.num_indexer_layers)
        ]
        item_lens = [
            self.index_k_buffer[i][0].nbytes for i in range(self.num_indexer_layers)
        ]
        if self.index_k_scale_buffer is not None:
            data_ptrs += [
                self.index_k_scale_buffer[i].data_ptr()
                for i in range(self.num_indexer_layers)
            ]
            data_lens += [
                self.index_k_scale_buffer[i].nbytes
                for i in range(self.num_indexer_layers)
            ]
            item_lens += [
                self.index_k_scale_buffer[i][0].nbytes
                for i in range(self.num_indexer_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.is_selective_host_layer(layer_id):
            raise RuntimeError(
                f"get_key_buffer: layer {layer_id} is a selected (Host DRAM) layer"
            )
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        slot = self._resident_slot(layer_id)
        if self.k_store_dtype != self.dtype:
            return self.k_buffer[slot].view(self.dtype)
        return self.k_buffer[slot]

    def get_value_buffer(self, layer_id: int):
        if self.is_selective_host_layer(layer_id):
            raise RuntimeError(
                f"get_value_buffer: layer {layer_id} is a selected (Host DRAM) layer"
            )
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        slot = self._resident_slot(layer_id)
        if self.v_store_dtype == self.store_dtype and self.store_dtype != self.dtype:
            return self.v_buffer[slot].view(self.dtype)
        return self.v_buffer[slot]

    def get_index_k_buffer(self, layer_id: int):
        indexer_slot = self._get_indexer_slot(layer_id)
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.k_store_dtype != self.dtype:
            return self.index_k_buffer[indexer_slot].view(self.dtype)
        return self.index_k_buffer[indexer_slot]

    def get_index_k_scale_buffer(self, layer_id: int):
        indexer_slot = self._get_indexer_slot(layer_id)
        if self.index_k_scale_buffer is None:
            raise RuntimeError(
                "Indexer scale cache is unavailable because the quantized "
                "lightning Indexer is disabled"
            )
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_scale_buffer[indexer_slot]

    # for disagg
    def get_contiguous_buf_infos(self):
        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        kv_item_lens: list[int] = []

        # MAIN_K: iterate over all logical layers
        # NOTE: kv_item_lens must be PER-PAGE bytes (page_size * record_bytes)
        # because the PD transfer path is page-indexed (kv_to_page_indices).
        # Resident layers get this naturally from k_buffer[slot][0].nbytes;
        # selected layers must match the same per-page convention.
        per_page_bytes = self.kv_cache_dim * self.page_size
        for lid in range(self.start_layer, self.start_layer + self.layer_num):
            if self.is_selective_host_layer(lid):
                # Selected layer: publish Host HVA
                host_pool = getattr(self, "selective_host_pool", None)
                if host_pool is not None:
                    kv_data_ptrs.append(host_pool.layer_hva(lid))
                    kv_data_lens.append(host_pool.layer_bytes(lid))
                    kv_item_lens.append(per_page_bytes)
                else:
                    # Fallback: publish zero-size placeholder
                    kv_data_ptrs.append(0)
                    kv_data_lens.append(0)
                    kv_item_lens.append(per_page_bytes)
            else:
                # Resident layer: publish NPU pointer
                slot = self._resident_slot(lid)
                kv_data_ptrs.append(self.k_buffer[slot].data_ptr())
                kv_data_lens.append(self.k_buffer[slot].nbytes)
                kv_item_lens.append(self.k_buffer[slot][0].nbytes)

        # MAIN_V: zero-size in packed FP8 mode, skip
        if not self.dsa_kv_cache_store_fp8:
            for lid in range(self.start_layer, self.start_layer + self.layer_num):
                if not self.is_selective_host_layer(lid):
                    slot = self._resident_slot(lid)
                    kv_data_ptrs.append(self.v_buffer[slot].data_ptr())
                    kv_data_lens.append(self.v_buffer[slot].nbytes)
                    kv_item_lens.append(self.v_buffer[slot][0].nbytes)

        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr()
                for i in range(self.num_indexer_layers)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.num_indexer_layers)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.num_indexer_layers)
            ]
            if self.index_k_scale_buffer is not None:
                kv_data_ptrs += [
                    self.index_k_scale_buffer[i].data_ptr()
                    for i in range(self.num_indexer_layers)
                ]
                kv_data_lens += [
                    self.index_k_scale_buffer[i].nbytes
                    for i in range(self.num_indexer_layers)
                ]
                kv_item_lens += [
                    self.index_k_scale_buffer[i][0].nbytes
                    for i in range(self.num_indexer_layers)
                ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_kv_layer_ids(self):
        """Logical layer ids aligned with ``get_contiguous_buf_infos``."""
        local_layer_ids = list(
            range(self.start_layer, self.start_layer + self.layer_num)
        )
        layer_ids = (
            local_layer_ids
            if self.dsa_kv_cache_store_fp8
            else local_layer_ids * 2
        )
        if self.index_head_dim is not None:
            layer_ids += list(self.indexer_layer_ids)
            if self.index_k_scale_buffer is not None:
                layer_ids += list(self.indexer_layer_ids)
        return layer_ids

    def get_state_layer_ids(self):
        layer_ids = list(self.indexer_layer_ids)
        if self.index_k_scale_buffer is not None:
            layer_ids += list(self.indexer_layer_ids)
        return layer_ids

    def _pack_dsa_fp8_kv_cache(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        expected_cache_dim = get_dsa_fp8_packed_cache_dim(
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        if self.kv_cache_dim != expected_cache_dim:
            raise RuntimeError(
                f"Unexpected packed DSA KV width {self.kv_cache_dim}; "
                f"expected {expected_cache_dim}."
            )
        if cache_k.numel() != num_tokens * self.kv_lora_rank:
            raise RuntimeError(
                f"Unexpected DSA k_nope shape {tuple(cache_k.shape)} for "
                f"{num_tokens} cache locations."
            )
        if cache_v.numel() != num_tokens * self.qk_rope_head_dim:
            raise RuntimeError(
                f"Unexpected DSA k_rope shape {tuple(cache_v.shape)} for "
                f"{num_tokens} cache locations."
            )

        num_tiles = self.kv_lora_rank // DSA_KV_QUANT_TILE_SIZE
        k_nope_tiles = (
            cache_k.to(torch.bfloat16)
            .reshape(num_tokens * num_tiles, DSA_KV_QUANT_TILE_SIZE)
            .contiguous()
        )
        k_nope_fp8, k_nope_scale = torch_npu.npu_dynamic_quant(
            k_nope_tiles, dst_type=torch.float8_e4m3fn
        )
        k_nope_fp8 = k_nope_fp8.reshape(num_tokens, 1, self.kv_lora_rank).contiguous()
        k_rope_bf16 = (
            cache_v.to(torch.bfloat16)
            .reshape(num_tokens, 1, self.qk_rope_head_dim)
            .contiguous()
        )
        k_nope_scale = (
            k_nope_scale.to(torch.float32)
            .reshape(num_tokens, 1, num_tiles)
            .contiguous()
        )

        packed = torch.empty(
            (num_tokens, 1, self.kv_cache_dim),
            dtype=torch.float8_e4m3fn,
            device=cache_k.device,
        )
        packed_bytes = packed.view(torch.uint8)
        rope_begin = self.kv_lora_rank
        rope_end = rope_begin + self.qk_rope_head_dim * torch.bfloat16.itemsize
        packed_bytes[..., :rope_begin].copy_(k_nope_fp8.view(torch.uint8))
        packed_bytes[..., rope_begin:rope_end].copy_(k_rope_bf16.view(torch.uint8))
        packed_bytes[..., rope_end:].copy_(k_nope_scale.view(torch.uint8))
        return packed

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id

        # Selective HiSparse: selected layers pack to scratch + D2H
        if self.is_selective_host_layer(layer_id):
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            if self.selective_coordinator is not None:
                packed = self._pack_dsa_fp8_kv_cache(
                    cache_k, cache_v, loc.numel()
                )
            if self.selective_coordinator is not None:
                self.selective_coordinator.publish_new_packed_kv(
                    layer_id=layer_id,
                    logical_locs=loc,
                    packed_kv=packed.view(-1, self.kv_cache_dim),
                )
            return

        if self.dsa_kv_cache_store_fp8:
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            packed_cache = self._pack_dsa_fp8_kv_cache(cache_k, cache_v, loc.numel())
            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[self._resident_slot(layer_id)].view(
                    -1, 1, self.kv_cache_dim
                ),
                loc.view(-1, 1),
                packed_cache,
            )
            return

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[self._resident_slot(layer_id)].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[self._resident_slot(layer_id)].view(
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
        indexer_slot = self._get_indexer_slot(layer_id)
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.k_store_dtype != self.dtype:
            index_k = index_k.view(self.k_store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[indexer_slot].view(-1, 1, self.index_head_dim),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k_scale: torch.Tensor,
    ):
        indexer_slot = self._get_indexer_slot(layer_id)
        if self.index_k_scale_buffer is None:
            raise RuntimeError(
                "Indexer scale cache is unavailable because the quantized "
                "lightning Indexer is disabled"
            )
        torch_npu.npu_scatter_nd_update_(
            self.index_k_scale_buffer[indexer_slot].view(-1, 1),
            loc.view(-1, 1),
            index_k_scale.to(torch.float32).view(-1, 1),
        )

    def _get_cpu_offload_layer_buffers(self, local_layer_id: int):
        """Return the physical buffers owned by one logical transformer layer."""
        num_slots = (
            self.k_buffer[local_layer_id].shape[0]
            * self.k_buffer[local_layer_id].shape[1]
        )
        buffers = [
            self.k_buffer[local_layer_id].reshape(num_slots, 1, self.kv_cache_dim),
            self.v_buffer[local_layer_id].reshape(num_slots, 1, self.kr_cache_dim),
        ]
        layer_id = self.start_layer + local_layer_id
        indexer_slot = self.indexer_layer_id_to_slot.get(layer_id)
        if indexer_slot is not None:
            buffers.append(
                self.index_k_buffer[indexer_slot].view(
                    num_slots, 1, self.index_head_dim
                )
            )
            if self.index_k_scale_buffer is not None:
                buffers.append(
                    self.index_k_scale_buffer[indexer_slot].view(num_slots, 1)
                )
        return buffers

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def get_cpu_copy(self, indices, mamba_indices=None):
        if self.selective_host_layer_ids:
            raise RuntimeError(
                "get_cpu_copy is not supported when selective HiSparse is "
                "enabled. Use PD rebootstrap for retraction recovery."
            )
        torch.npu.synchronize()
        buf_of_layers = [
            self._get_cpu_offload_layer_buffers(local_layer_id)
            for local_layer_id in range(self.layer_num)
        ]

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        if self.selective_host_layer_ids:
            raise RuntimeError(
                "load_cpu_copy is not supported when selective HiSparse is "
                "enabled. Use PD rebootstrap for retraction recovery."
            )
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            layer_buffers = self._get_cpu_offload_layer_buffers(local_layer_id)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                assert len(chunk) == len(layer_buffers)
                for layer_buffer, cpu_buffer in zip(layer_buffers, chunk):
                    assert cpu_buffer.shape[0] == len(chunk_indices)
                    layer_buffer[chunk_indices] = cpu_buffer.to(
                        layer_buffer.device, non_blocking=True
                    )
        torch.npu.synchronize()
