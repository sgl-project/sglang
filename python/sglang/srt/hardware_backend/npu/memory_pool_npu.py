from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.dcp.layout import localize_dcp_indices
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu


def _mla_fia_nz_scatter_indices(
    loc: torch.Tensor, head_dim: int, page_size: int
) -> torch.Tensor:
    """Return physical rows for token-wise writes into an MLA NZ cache.

    The storage allocation remains page-major ``[page, slot, 1, D]`` for
    transfer and bookkeeping compatibility. FIA reads that storage as
    ``[page, 1, D / 16, page_size, 16]``. A token-major scatter would therefore
    write the wrong physical rows, so every logical token expands to its
    ``D / 16`` NZ tiles.
    """
    if head_dim % 16:
        raise ValueError(
            "FIA NZ MLA cache requires a head dimension divisible by 16, "
            f"got {head_dim}."
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")

    num_tiles = head_dim // 16
    page = torch.div(loc, page_size, rounding_mode="floor")
    slot = torch.remainder(loc, page_size)
    tiles = torch.arange(num_tiles, dtype=loc.dtype, device=loc.device)
    # Flatten [token, tile] in the same order as source.view(T, tiles, 16).
    rows = ((page[:, None] * num_tiles + tiles) * page_size) + slot[:, None]
    return rows.reshape(-1, 1)


def _init_npu_conv_state(
    conv_state_in,
    conv_state_shape,
    speculative_num_draft_tokens: Optional[int] = None,
    is_kda: bool = False,
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # Both KDA and Mamba/GDN NPU conv states use the unified
    # [layers, pool, window, channels] layout. KDA shapes arrive as
    # (window, channels) while Mamba/GDN shapes arrive as (channels, window);
    # resolve the correct axis ordering and extend the window by
    # speculative_num_draft_tokens - 1 so that verify can write all draft
    # token conv states directly into conv_states (GDN rollback scheme).
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                (conv_shape[0] if is_kda else conv_shape[1]) + extra_conv_len,
                conv_shape[1] if is_kda else conv_shape[0],
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
            # FIA exposes the KV cache as per-layer Python views so graph
            # capture does not retain the full multi-layer tensor. HiCache's
            # NPU exchange operator still requires the original contiguous
            # [layer, page, token, head, dim] allocation.
            self._hicache_k_buffer = torch.zeros(
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
            self._hicache_v_buffer = torch.zeros(
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
            # Keep a reference to the contiguous tensor for HiCache
            # D2H/H2D transfers (transfer_kv_dim_exchange expects a
            # tensor, not the per-layer list used in FIA mode below).
            self.k_buffer_tensor = self._hicache_k_buffer
            self.v_buffer_tensor = self._hicache_v_buffer

            self.k_buffer = self._hicache_k_buffer
            self.v_buffer = self._hicache_v_buffer

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self._hicache_k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self._hicache_v_buffer[i].view(
                        -1, 1, self.head_num, self.v_head_dim
                    )
                    for i in range(self.layer_num)
                ]

    def get_hicache_transfer_buffers(self):
        """Return contiguous all-layer KV tensors for NPU HiCache IO."""
        return self._hicache_k_buffer, self._hicache_v_buffer

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
    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
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

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
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
            # Keep a reference to the contiguous tensor for HiCache
            # D2H/H2D transfers (transfer_kv_dim_exchange expects a
            # tensor, not the per-layer list used in FIA mode below).
            self.k_buffer_tensor = self.k_buffer
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
        index_size: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_page_size: Optional[int] = None,
        indexer_layer_ids: Optional[Sequence[int]] = None,
        kv_cache_dim: Optional[int] = None,
        is_draft_worker: bool = False,
        kv_int8_layout: bool = False,
    ):
        # MLAPO historically owned NZ writes. Keep the allocation unchanged and
        # write into the NZ-addressed view below so ordinary MLA (including
        # Kimi-K3 MTP) can use FIA NZ without MLAPO.
        self.use_fia_nz = get_bool_env_var("SGLANG_USE_FIA_NZ")
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
        self.enable_sparsity_driven_kv_offload = (
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get()
        )
        if self.enable_sparsity_driven_kv_offload and self.index_head_dim is None:
            raise ValueError("Sparsity-driven KV offload requires an index KV cache.")
        if index_head_dim is None:
            self.indexer_layer_ids = ()
        elif indexer_layer_ids is None:
            self.indexer_layer_ids = tuple(
                range(self.start_layer, self.start_layer + self.layer_num)
            )
        else:
            self.indexer_layer_ids = tuple(indexer_layer_ids)
        self.num_indexer_layers = len(self.indexer_layer_ids)
        self.indexer_layer_id_to_slot = {
            layer_id: slot for slot, layer_id in enumerate(self.indexer_layer_ids)
        }
        assert len(self.indexer_layer_id_to_slot) == self.num_indexer_layers
        assert all(
            self.start_layer <= i < self.start_layer + self.layer_num
            for i in self.indexer_layer_ids
        )
        requested_kv_cache_dim = kv_cache_dim
        self.dsa_kv_cache_store_fp8 = (
            index_head_dim is not None
            and dtype == torch.float8_e4m3fn
            and requested_kv_cache_dim is not None
        )
        if self.dsa_kv_cache_store_fp8:
            assert index_head_dim == 128 and kv_lora_rank % 128 == 0
            assert requested_kv_cache_dim == (
                kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2
            )
            self.store_dtype = dtype
        self.kv_cache_dim = (
            requested_kv_cache_dim if self.dsa_kv_cache_store_fp8 else kv_lora_rank
        )
        self.kr_cache_dim = 0 if self.dsa_kv_cache_store_fp8 else qk_rope_head_dim
        self.index_k_scale_buffer = None
        self.indexer_hadamard_128 = None
        self.index_page_size = page_size if index_page_size is None else index_page_size
        self.index_size = size if index_size is None else index_size
        parallel = get_parallel()
        self.dcp_size = parallel.attn_dcp_size
        self.dcp_rank = parallel.attn_dcp_rank
        global_page_padding = self.dcp_size if self.dcp_size > 1 else 1
        kv_page_padding = global_page_padding if is_draft_worker else 1
        index_page_padding = global_page_padding if index_head_dim is not None else 1
        if kv_page_padding < 1 or index_page_padding < 1:
            raise ValueError("NPU MLA page padding must be positive")
        self.kv_page_padding = kv_page_padding
        self.index_page_padding = index_page_padding
        self.is_draft_worker = is_draft_worker

        # int8 COMBINE layout for DSA models on Ascend NPU, consumed by
        # npu_kv_quant_sparse_flash_attention:
        #   K row (656B for kv_lora_rank=512): int8 nope [0:512) | bf16 rope
        #     bytes [512:640) (the operator tiling pins rope_head_dim=64,
        #     i.e. 128 bf16 bytes; NoPE models zero-fill) | fp32 scales
        #     [640:656) (per-128-dim-tile amax/127, kv_lora_rank//128 tiles)
        # Single-pool layout: the operator's `value` argument is a dead
        # parameter (the kernel binds valueGm but never dereferences it;
        # single 656B pool and dual 656+528 pools give bit-identical
        # outputs), so under the int8 layout we allocate ONLY the 656B
        # k_buffer; v_buffer is None and all reads/writes go through
        # k_buffer. Buffers use torch.int8 (NOT uint8):
        # npu_scatter_nd_update_ (aclnn) has no uint8 instance in its
        # dtype whitelist.
        # Enabled via SGLANG_DSA_KV_INT8=1 (model_runner overrides
        # kv_cache_dtype to torch.int8, the pool builder passes
        # kv_int8_layout=True).
        # Dtype self-check fallback: some assembly points (e.g. the draft
        # worker path) build the pool with kv_cache_dtype already
        # overridden to torch.int8 but do not forward kv_int8_layout.
        # An int8 dtype with the bf16 layout would produce a wrong-shaped
        # k_buffer and an int8 index_k_buffer, which crashes the
        # bf16-only npu_lightning_indexer -- so dtype==torch.int8 forces
        # the int8 COMBINE layout. Belt-and-braces with the explicit
        # kv_int8_layout kwarg above.
        self.kv_int8_layout = kv_int8_layout or (dtype == torch.int8)
        if self.kv_int8_layout and self.enable_sparsity_driven_kv_offload:
            raise ValueError(
                "SGLANG_DSA_KV_INT8 is not supported together with "
                "sparsity-driven KV offload (SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD)."
            )
        self.int8_rope_bytes = 2 * 64  # rope_head_dim pinned to 64 by tiling
        self.int8_scale_bytes = 4 * (kv_lora_rank // 128)  # fp32 per-128-tile
        self.k_buffer_width = (
            kv_lora_rank + self.int8_rope_bytes + self.int8_scale_bytes
            if self.kv_int8_layout
            else kv_lora_rank
        )
        self.v_buffer_width = qk_rope_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            if self.enable_sparsity_driven_kv_offload:
                self.k_buffer = None
                self.v_buffer = None
            elif self.kv_int8_layout:
                # int8 layout stores the COMBINE rows as int8 rows with
                # the width above. No V pool: the quantized-sparse
                # operator never dereferences the value pointer, so a
                # single 656B K pool serves both K and V.
                self.k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + self.kv_page_padding,
                        self.page_size,
                        1,
                        self.k_buffer_width,
                    ),
                    dtype=torch.int8,
                    device=self.device,
                )
                self.v_buffer = None
            else:
                self.k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + self.kv_page_padding,
                        self.page_size,
                        1,
                        self.kv_cache_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                self.v_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + self.kv_page_padding,
                        self.page_size,
                        1,
                        self.kr_cache_dim,
                    ),
                    dtype=(
                        torch.bfloat16
                        if self.dsa_kv_cache_store_fp8
                        else self.store_dtype
                    ),
                    device=self.device,
                )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        self.num_indexer_layers,
                        self.index_size // self.index_page_size
                        + self.index_page_padding,
                        self.index_page_size,
                        1,
                        self.index_head_dim,
                    ),
                    # The indexer (npu_lightning_indexer) keeps consuming
                    # bf16 index K; the int8 layout applies only to the
                    # latent K/V rows, so pin the index pool to bf16 even
                    # though kv_cache_dtype was overridden to int8.
                    dtype=torch.bfloat16 if self.kv_int8_layout else self.store_dtype,
                    device=self.device,
                )
                if self.dsa_kv_cache_store_fp8 and self.num_indexer_layers > 0:
                    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
                        create_npu_hadamard_128,
                    )

                    self.index_k_scale_buffer = torch.zeros(
                        (*self.index_k_buffer.shape[:-2], 1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.indexer_hadamard_128 = create_npu_hadamard_128(
                        self.index_head_dim, self.device
                    )

        self._finalize_allocation_log(size)

    def _copy_indices_for_buffer(self, indices, uses_global_slots):
        if uses_global_slots or self.dcp_size <= 1:
            return indices
        local_indices = localize_dcp_indices(
            indices,
            self.dcp_size,
            self.dcp_rank,
            self.page_size,
        )
        return local_indices[local_indices >= 0]

    def get_kv_size_bytes(self):
        kv_size_bytes = 0
        if getattr(self, "k_buffer", None) is not None:
            for k_cache in self.k_buffer:
                kv_size_bytes += get_tensor_size_bytes(k_cache)
        # Single pool: v_buffer is None under the int8 layout (do not
        # double-count the shared k_buffer).
        if getattr(self, "v_buffer", None) is not None:
            for v_cache in self.v_buffer:
                kv_size_bytes += get_tensor_size_bytes(v_cache)
        if getattr(self, "index_k_buffer", None) is not None:
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        if self.index_k_scale_buffer is not None:
            kv_size_bytes += get_tensor_size_bytes(self.index_k_scale_buffer)
        return kv_size_bytes

    def _raise_if_native_kv_cache_disabled(self):
        # Single pool: v_buffer is None by design under the int8
        # layout; the k_buffer alone is the native device KV cache.
        if self.kv_int8_layout:
            if getattr(self, "k_buffer", None) is None:
                raise RuntimeError(
                    "Native NPU MLA device KV cache is disabled; "
                    "k_buffer is not available. Use the sparse KV manager "
                    "path, or re-enable native device KV cache for this "
                    "code path."
                )
            return
        if (
            getattr(self, "k_buffer", None) is None
            or getattr(self, "v_buffer", None) is None
        ):
            raise RuntimeError(
                "Native NPU MLA device KV cache is disabled; "
                "k_buffer/v_buffer are not available. Use the sparse KV manager "
                "path, or re-enable native device KV cache for this code path."
            )

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        # Single pool: both K and V views are the same k_buffer tensor
        # under the int8 layout.
        if self.kv_int8_layout:
            return (
                self.k_buffer[layer_id - self.start_layer],
                self.k_buffer[layer_id - self.start_layer],
            )
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        buffers = list(self.index_k_buffer)
        if self.index_k_scale_buffer is not None:
            buffers += list(self.index_k_scale_buffer)
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [buf[0].nbytes for buf in buffers]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()

        # Return the raw int8 COMBINE rows; the int8 reader
        # (npu_kv_quant_sparse_flash_attention) consumes them directly.
        # The bf16 view below is only meaningful for the non-int8 layout.
        if self.kv_int8_layout:
            return self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()

        # Single pool: the value argument of
        # npu_kv_quant_sparse_flash_attention is fed the same k_buffer
        # rows (dead parameter at the kernel level).
        if self.kv_int8_layout:
            return self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if getattr(self, "index_k_buffer", None) is None:
            raise RuntimeError("NPU MLA index KV cache is not allocated.")

        # The index pool is pinned to bf16 (see the allocation above);
        # never reinterpret it through the int8 self.dtype.
        if self.kv_int8_layout:
            return self.index_k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.index_k_buffer[self._get_indexer_slot(layer_id)].view(
                self.dtype
            )
        return self.index_k_buffer[self._get_indexer_slot(layer_id)]

    def _get_indexer_slot(self, layer_id: int) -> int:
        return self.indexer_layer_id_to_slot[layer_id]

    def get_index_k_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_scale_buffer[self._get_indexer_slot(layer_id)]

    def set_index_k_scale_buffer(self, layer_id: int, loc, scale):
        torch_npu.npu_scatter_nd_update_(
            self.index_k_scale_buffer[self._get_indexer_slot(layer_id)].view(-1, 1),
            loc.view(-1, 1),
            scale.view(-1, 1),
        )

    def _get_disagg_buffer_entries(self):
        """Return (buffer, uses_global_slots) entries in PD transfer order."""
        self._raise_if_native_kv_cache_disabled()
        global_kv = self.is_draft_worker
        entries = [(buffer, global_kv) for buffer in self.k_buffer]
        if not getattr(self, "dsa_kv_cache_store_fp8", False):
            entries += [(buffer, global_kv) for buffer in self.v_buffer]
        if self.index_head_dim is not None:
            entries += [(buffer, True) for buffer in self.index_k_buffer]
            if self.index_k_scale_buffer is not None:
                entries += [(buffer, True) for buffer in self.index_k_scale_buffer]
        return entries

    # for disagg
    def get_contiguous_buf_infos(self):
        # Under the int8 layout v_buffer is None and the PD-disagg peer
        # would have to interpret the single 656B pool as the int8 COMBINE
        # layout -- fail loud if anyone tries to take this path with the
        # int8 layout on.
        if self.kv_int8_layout:
            raise NotImplementedError(
                "get_contiguous_buf_infos (PD disaggregation) is not "
                "supported with the int8 KV single-pool layout "
                "(SGLANG_DSA_KV_INT8=1); peer-side int8 interpretation is "
                "not synchronized."
            )
        entries = self._get_disagg_buffer_entries()
        return (
            [buffer.data_ptr() for buffer, _ in entries],
            [buffer.nbytes for buffer, _ in entries],
            [
                buffer[0].nbytes * (self.dcp_size if uses_global_slots else 1)
                for buffer, uses_global_slots in entries
            ],
        )

    def get_dcp_remote_decode_layout(self) -> list[bool]:
        """Whether each PD entry uses allocator-global slots on decode."""
        return [
            uses_global_slots
            for _, uses_global_slots in self._get_disagg_buffer_entries()
        ]

    def get_kv_layer_ids(self):
        return (
            list(range(self.start_layer, self.start_layer + self.layer_num)) * 2
            + self.get_state_layer_ids()
        )

    def get_state_layer_ids(self):
        return list(self.indexer_layer_ids) * (
            2 if self.index_k_scale_buffer is not None else 1
        )

    def _pack_dsa_fp8_kv_cache(self, cache_k, cache_v):
        latent = cache_k.reshape(-1, self.kv_lora_rank)
        quantized, scale = torch_npu.npu_dynamic_quant(
            latent.reshape(-1, 128), dst_type=self.dtype
        )
        rows = latent.shape[0]
        # Opaque record: latent FP8 | rope BF16 bytes | per-tile FP32 scales.
        packed = torch.cat(
            (
                quantized.reshape(rows, self.kv_lora_rank).view(torch.uint8),
                cache_v.to(torch.bfloat16)
                .reshape(rows, self.qk_rope_head_dim)
                .contiguous()
                .view(torch.uint8),
                scale.to(torch.float32)
                .reshape(rows, self.kv_lora_rank // 128)
                .contiguous()
                .view(torch.uint8),
            ),
            dim=-1,
        )
        return packed.view(self.dtype)

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        self._raise_if_native_kv_cache_disabled()
        layer_id = layer.layer_id
        # Single-pool write: per-128-tile int8 quantization into the
        # 656B COMBINE row (int8 nope | rope bytes (NoPE zero-fill) |
        # fp32 scales), one scatter, no V row at all (the operator
        # never reads value). Measured dequantized error against bf16
        # latents: ~3.3%.
        if self.kv_int8_layout:
            k_row = self._quantize_kv_int8_combine(cache_k, cache_v)
            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.k_buffer_width
                ),
                loc.view(-1, 1),
                k_row.view(-1, 1, self.k_buffer_width),
            )
            return

        if self.dsa_kv_cache_store_fp8:
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            packed = self._pack_dsa_fp8_kv_cache(cache_k, cache_v)
            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.kv_cache_dim
                ),
                loc.view(-1, 1),
                packed.view(-1, 1, self.kv_cache_dim),
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

        if self.use_fia_nz:
            self._set_fia_nz_kv_buffer(layer_id, loc, cache_k, cache_v)
            return

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.k_buffer_width),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.k_buffer_width),
        )
        if self.qk_rope_head_dim > 0:
            # Models with qk_rope_head_dim=0 (no rope segment) have a
            # 0-width v_buffer; an empty scatter would raise on view(-1, 1, 0).
            torch_npu.npu_scatter_nd_update_(
                self.v_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.v_buffer_width
                ),
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.v_buffer_width),
            )

    def _quantize_kv_int8_combine(
        self, cache_k: torch.Tensor, cache_v: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Quantize bf16 MLA latents into the single int8 COMBINE row
        consumed by npu_kv_quant_sparse_flash_attention.

        cache_k arrives as (num_tokens, tp_k_head_num, kv_lora_rank) bf16
        latents; cache_v is the (num_tokens, 1, qk_rope_head_dim) post-rotary
        k_rope (its bf16 bytes are spliced into the row's [512:640) rope
        segment; NoPE models zero-fill it). cache_v doubles as the dead
        value-row placeholder: the K row is the single latent (the operator
        never reads the value argument).
        Quantization: symmetric per-128-dim-tile,
        scale = amax/127 (fp32), via torch_npu.npu_dynamic_block_quant(
        row_block_size=1, col_block_size=128) -- bit-identical to the
        reference amax/127 formula and 5.6x faster. All ops here are
        NPU-side with static shapes => NPUGraph-capturable (no host
        sync, no data-dependent branching).
        """
        num_tokens = cache_k.shape[0]
        k_latent = cache_k.reshape(num_tokens, self.kv_lora_rank)

        q_k, s_k = self._int8_tile_quant(k_latent)

        # K row: int8 nope | bf16 rope bytes | fp32 scale bytes.
        # The rope bytes are already post-rotary (forward_dsa_prepare_npu
        # applies m.rotary_emb before set_kv_buffer), so a raw byte splice
        # is exactly what the bf16 pool stores in v_buffer -- no rotary is
        # (re)applied anywhere on the read path, kernel included (zero-rope
        # rows were verified mathematically equivalent, i.e. the kernel
        # never applies rope itself). Move the bytes as int8
        # (bit-identical move; scatter's aclnn whitelist has no
        # uint8/bf16 instance). NoPE models (qk_rope_head_dim==0) keep
        # the zero fill.
        if self.qk_rope_head_dim > 0:
            if cache_v is None or cache_v.numel() == 0:
                raise RuntimeError(
                    "int8 KV cache: rope model (qk_rope_head_dim="
                    f"{self.qk_rope_head_dim}) but set_kv_buffer got no "
                    "k_rope; the rope segment cannot be synthesized."
                )
            rope_flat = cache_v.reshape(num_tokens, -1)
            if rope_flat.shape[1] != self.int8_rope_bytes // 2:
                raise RuntimeError(
                    "int8 KV cache: k_rope width "
                    f"{rope_flat.shape[1]} != the COMBINE rope segment "
                    f"{self.int8_rope_bytes // 2} (tiling pins "
                    "rope_head_dim=64)."
                )
            rope_bytes = rope_flat.contiguous().view(torch.int8)
        else:
            rope_bytes = torch.zeros(
                (num_tokens, self.int8_rope_bytes),
                dtype=torch.int8,
                device=cache_k.device,
            )
        k_row = torch.cat(
            [q_k.view(torch.int8), rope_bytes, s_k.view(torch.int8)],
            dim=-1,
        )
        return k_row

    def _int8_tile_quant(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-128-tile symmetric int8 quantization of a bf16
        (num_tokens, kv_lora_rank) latent. Returns (int8 q [N, kv_lora_rank],
        fp32 scale bytes [N, int8_scale_bytes]) -- the scale is returned as a
        raw-byte view so callers can splice it into the COMBINE row.
        Fallback pure-torch path mirrors npu_dynamic_block_quant exactly
        (amax/127, round, clamp +-127) for environments without the op.
        """
        num_tokens = latent.shape[0]
        if hasattr(torch_npu, "npu_dynamic_block_quant"):
            # dst_type=1 -> uint8 payload, row=1/col=128 -> per-tile scale
            # (amax/127, verified bit-identical against the reference
            # formula).
            q, s = torch_npu.npu_dynamic_block_quant(
                latent,
                min_scale=0.0,
                dst_type=1,
                row_block_size=1,
                col_block_size=128,
            )
            q = q.view(torch.int8)
            s = s.float()
            while s.dim() > 2:
                s = s.squeeze(-1)
            # (num_tokens, num_tiles) fp32 -> raw bytes (num_tokens, 16)
            s = s.reshape(num_tokens, self.int8_scale_bytes // 4).contiguous()
            return q, s.view(torch.int8)

        num_tiles = self.kv_lora_rank // 128
        latent = latent.to(torch.float32)
        scales = (
            torch.amax(latent.view(num_tokens, num_tiles, 128).abs(), dim=-1) / 127.0
        )
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        q = torch.clamp(
            torch.round(
                latent.view(num_tokens, num_tiles, 128)
                / scales.unsqueeze(-1).clamp(min=1e-30)
            ),
            -127,
            127,
        ).to(torch.int8)
        return (
            q.reshape(num_tokens, self.kv_lora_rank),
            scales.contiguous().view(torch.int8),
        )

    def _set_fia_nz_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Store MLA latent and RoPE KV tensors in FIA's NZ tile order."""

        def scatter(cache: torch.Tensor, values: torch.Tensor, head_dim: int):
            num_tiles = head_dim // 16
            indices = _mla_fia_nz_scatter_indices(loc, head_dim, self.page_size)
            # Destination rows are ordered [page, tile, slot]. Source rows use
            # the matching [token, tile] order after this reshape.
            dst = cache.view(-1, 1, num_tiles, self.page_size, 16).view(-1, 16)
            src = values.contiguous().view(-1, num_tiles, 16).view(-1, 16)
            torch_npu.npu_scatter_nd_update_(dst, indices, src)

        offset = layer_id - self.start_layer
        scatter(self.k_buffer[offset], cache_k, self.kv_lora_rank)
        scatter(self.v_buffer[offset], cache_v, self.qk_rope_head_dim)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        # Index pool is pinned to bf16 under the int8 layout (see the
        # allocation above); cast against the buffer dtype, not self.dtype.
        index_dtype = torch.bfloat16 if self.kv_int8_layout else self.dtype
        if index_k.dtype != index_dtype:
            index_k = index_k.to(index_dtype)

        if not self.kv_int8_layout and self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[self._get_indexer_slot(layer_id)].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def _chunk_copy_npu_to_cpu(
        self, buf_of_layers, indices, uses_global_slots_per_layer
    ):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer, uses_global_slots in zip(
            buf_of_layers, uses_global_slots_per_layer, strict=True
        ):  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[self._copy_indices_for_buffer(ci, uses_global)].to(
                            "cpu", non_blocking=True
                        )
                        for t, uses_global in zip(
                            tensors_per_layer, uses_global_slots, strict=True
                        )
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def _get_cpu_offload_layer_buffers(self, local_layer_id):
        # flatten(page, slot) also works for the zero-width packed V placeholder.
        buffers = [
            self.k_buffer[local_layer_id].flatten(0, 1),
            self.v_buffer[local_layer_id].flatten(0, 1),
        ]
        slot = self.indexer_layer_id_to_slot.get(local_layer_id + self.start_layer)
        if slot is not None:
            buffers.append(self.index_k_buffer[slot].flatten(0, 1))
            if self.index_k_scale_buffer is not None:
                buffers.append(self.index_k_scale_buffer[slot].flatten(0, 1))
        if self.dsa_kv_cache_store_fp8:
            # Retraction copies opaque records; byte views also avoid FP8
            # advanced-indexing restrictions, without decoding/requantizing.
            buffers = [
                buf.view(torch.uint8) if buf.dtype == torch.float8_e4m3fn else buf
                for buf in buffers
            ]
        return buffers

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # CPU offloading of the int8 COMBINE rows is not wired: the
        # offload path assumes the two-buffer bf16 layout.
        if self.kv_int8_layout:
            raise NotImplementedError(
                "CPU KV offloading is not supported with the int8 KV "
                "single-pool layout (SGLANG_DSA_KV_INT8=1)."
            )
        torch.npu.synchronize()
        buf_of_layers = [
            self._get_cpu_offload_layer_buffers(i) for i in range(self.layer_num)
        ]
        uses_global_slots_per_layer = []
        for buffers in buf_of_layers:
            # MLA K/V is rank-local under DCP. The replicated
            # indexer buffers retain allocator-global slot identities.
            uses_global_slots_per_layer.append(
                [self.is_draft_worker, self.is_draft_worker]
                + [True] * (len(buffers) - 2)
            )

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(
            buf_of_layers, indices, uses_global_slots_per_layer
        )
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        # See get_cpu_copy: the offload path assumes the two-buffer
        # bf16 layout.
        if self.kv_int8_layout:
            raise NotImplementedError(
                "CPU KV offloading is not supported with the int8 KV "
                "single-pool layout (SGLANG_DSA_KV_INT8=1)."
            )
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            buffers = self._get_cpu_offload_layer_buffers(local_layer_id)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                cpu_index = 0
                for buffer, uses_global_slots in zip(
                    buffers,
                    [self.is_draft_worker, self.is_draft_worker]
                    + [True] * (len(buffers) - 2),
                    strict=True,
                ):
                    if buffer is None:
                        continue
                    cpu = chunk[cpu_index]
                    cpu_index += 1
                    target_indices = self._copy_indices_for_buffer(
                        chunk_indices, uses_global_slots
                    )
                    assert cpu.shape[0] == len(target_indices)
                    buffer[target_indices] = cpu.to(buffer.device, non_blocking=True)
        torch.npu.synchronize()
