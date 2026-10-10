"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-request pending
index-key/RoPE ring and the paged compressed-K cache on top of the hybrid
full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.cache_sharding import (
    QSACacheShardingRuntime,
    assert_qsa_cache_sharding_runtime_match,
    get_qsa_cache_sharding_runtime,
)
from sglang.srt.mem_cache.allocator.page_interleave import (
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)

logger = logging.getLogger(__name__)

# State layer IDs are serialized as uint32 by the disaggregation protocols.
# Reserve the value below PLE's request-wide sentinel for QSA's request-wide
# RoPE ring, which is shared by all full-attention layers.
QSA_ROPE_STATE_LAYER_ID = (1 << 32) - 2


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


# ``--qsa-indexer-dtype`` choices. The pending key ring stays bf16 regardless.
QSA_INDEXER_DTYPE_CHOICES = ("auto", "bfloat16", "fp8_e4m3")


def resolve_qsa_indexer_dtype(name: str) -> torch.dtype:
    """Storage dtype of the compressed QSA indexer cache for a CLI value."""
    if name in ("auto", "bfloat16"):
        return torch.bfloat16
    if name == "fp8_e4m3":
        return torch.float8_e4m3fn
    raise ValueError(
        f"Unsupported --qsa-indexer-dtype {name!r}; expected one of "
        f"{QSA_INDEXER_DTYPE_CHOICES}"
    )


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def assert_qsa_indices_in_bounds(
    indices: torch.Tensor,
    upper_bound: int,
    *,
    valid_mask: torch.Tensor | None = None,
    label: str,
) -> None:
    """Validate QSA kernel indices without synchronizing CUDA tensors to host."""

    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if indices.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{label} must have integer dtype")
    if upper_bound < 0:
        raise ValueError(f"{label} upper bound must be non-negative")
    if valid_mask is None:
        valid_mask = indices >= 0
    elif valid_mask.shape != indices.shape:
        raise ValueError(f"{label} valid mask must match index shape")
    else:
        valid_mask = valid_mask.to(device=indices.device, dtype=torch.bool)
    condition = torch.all(
        (~valid_mask) | ((indices >= 0) & (indices < upper_bound))
    )
    message = f"{label} must be in [0, {upper_bound})"
    if indices.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise IndexError(message)


def _qsa_placement(*, rank: int, world_size: int, page_size: int):
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return PageInterleavePlacement(
        PageShardSpec(
            shard_rank=rank,
            shard_size=world_size,
            page_size=page_size,
            max_prefix_tokens=0,
            chunk_tokens=0,
        )
    )


def qsa_global_compressed_capacity(
    local_token_capacity: int,
    *,
    compress_ratio: int,
    world_size: int | None = None,
    allocator=None,
) -> int:
    if allocator is not None:
        allocator_world_size = page_interleave_shard_size(allocator)
        if world_size is not None and world_size != allocator_world_size:
            raise ValueError("allocator and explicit QSA shard widths disagree")
        world_size = allocator_world_size
    world_size = 1 if world_size is None else world_size
    if min(local_token_capacity, compress_ratio, world_size) <= 0:
        raise ValueError("QSA compressed-capacity inputs must be positive")
    return -((local_token_capacity * world_size) // -compress_ratio)


@dataclass(frozen=True)
class QSARawKVSharding:
    """Interleaved owner mapping for the rank-local raw QSA K/V cache."""

    local_capacity: int
    page_size: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.local_capacity < 0:
            raise ValueError("local_capacity must be non-negative")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        object.__setattr__(
            self,
            "placement",
            _qsa_placement(rank=self.rank, world_size=self.world_size, page_size=1),
        )

    @property
    def global_capacity(self) -> int:
        return self.local_capacity * self.world_size

    def global_to_local_tokens(self, global_slots: torch.Tensor) -> torch.Tensor:
        global_slots = global_slots.long()
        valid = (global_slots >= 0) & (global_slots < self.global_capacity)
        owned = self.placement.local_mask(global_slots, self.rank)
        local = self.placement.local_index(global_slots.clamp_min(0))
        return torch.where(valid & owned, local, -1)

    def local_write_targets(
        self, global_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert_qsa_indices_in_bounds(
            global_slots,
            self.global_capacity,
            valid_mask=global_slots >= 0,
            label="raw KV write locations",
        )
        local = self.global_to_local_tokens(global_slots)
        owner = local >= 0
        safe_local = torch.where(owner, local, 0)
        assert_qsa_indices_in_bounds(
            safe_local,
            self.local_capacity,
            valid_mask=owner,
            label="raw KV local write locations",
        )
        return owner, safe_local

    def local_copy_targets(
        self, global_src: torch.Tensor, global_dst: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_src.shape != global_dst.shape:
            raise ValueError("source and destination slots must have matching shapes")
        src = self.global_to_local_tokens(global_src)
        dst = self.global_to_local_tokens(global_dst)
        src_owned = src >= 0
        dst_owned = dst >= 0
        if bool(torch.any(src_owned != dst_owned).item()):
            raise ValueError("cross-owner raw K/V copy requires communication")
        return src[src_owned], dst[dst_owned]

@dataclass(frozen=True)
class QSACompressedBlockSharding:
    """Interleaved full-page ownership for QSA compressed blocks.

    Every compressed block in a physical page has the same owner. The small
    per-request pending ring remains replicated because an incomplete
    compression group can be completed by the next token before a
    compressed-page owner is selected.
    """

    global_blocks: int
    compressed_page_size: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.global_blocks < 0:
            raise ValueError("global_blocks must be non-negative")
        if self.compressed_page_size <= 0:
            raise ValueError(
                "compressed_page_size must be positive, got "
                f"{self.compressed_page_size}"
            )
        object.__setattr__(
            self,
            "placement",
            _qsa_placement(
                rank=self.rank,
                world_size=self.world_size,
                page_size=self.compressed_page_size,
            ),
        )

    @property
    def global_pages(self) -> int:
        return -(self.global_blocks // -self.compressed_page_size)

    @property
    def local_pages(self) -> int:
        if self.rank >= self.global_pages:
            return 0
        return (self.global_pages - 1 - self.rank) // self.world_size + 1

    @property
    def local_blocks(self) -> int:
        return self.local_pages * self.compressed_page_size

    def global_to_local_pages(self, global_pages: torch.Tensor) -> torch.Tensor:
        global_pages = global_pages.long()
        valid = (global_pages >= 0) & (global_pages < self.global_pages)
        owned = global_pages.remainder(self.world_size) == self.rank
        return torch.where(
            valid & owned,
            torch.div(global_pages, self.world_size, rounding_mode="floor"),
            -1,
        )

    def global_to_local_blocks(self, global_blocks: torch.Tensor) -> torch.Tensor:
        global_blocks = global_blocks.long()
        valid = (global_blocks >= 0) & (global_blocks < self.global_blocks)
        safe_blocks = global_blocks.clamp_min(0)
        owned = self.placement.local_mask(safe_blocks, self.rank)
        local_blocks = self.placement.local_index(safe_blocks)
        return torch.where(valid & owned, local_blocks, -1)

    def local_write_targets(
        self,
        global_blocks: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense owner mask and safe rank-local destinations.

        Eager fused compression keeps its static, shape-derived group count.
        Non-owner and padded rows therefore stay in the launch but are masked
        before the kernel reads group members or writes compressed state.
        """
        if global_blocks.ndim != 1:
            raise ValueError("global_blocks must be one-dimensional")
        checked_mask = (
            global_blocks >= 0
            if valid_mask is None
            else valid_mask.to(device=global_blocks.device, dtype=torch.bool)
        )
        assert_qsa_indices_in_bounds(
            global_blocks,
            self.global_blocks,
            valid_mask=checked_mask,
            label="compressed KV write locations",
        )
        local_blocks = self.global_to_local_blocks(global_blocks)
        owner_mask = local_blocks >= 0
        if valid_mask is not None:
            if valid_mask.shape != global_blocks.shape:
                raise ValueError("valid_mask must match global_blocks")
            owner_mask &= valid_mask.to(device=global_blocks.device, dtype=torch.bool)
        safe_local_blocks = torch.where(owner_mask, local_blocks, 0)
        assert_qsa_indices_in_bounds(
            safe_local_blocks,
            self.local_blocks,
            valid_mask=owner_mask,
            label="compressed KV local write locations",
        )
        return owner_mask, safe_local_blocks


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # Full-KV pages are a multiple of the compress ratio, so no group straddles pages;
    # ``compressed_slot = full_slot // ratio`` needs no ownership bookkeeping;
    # lifecycle rides the full-KV allocator and radix tree.
    # Full slot 0 is the reserved padding slot; compressed slot 0 is the inert dump.
    # Pending-ring dtype: the raw keys are averaged from here, so it stays bf16.
    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls,
        *,
        kv_heads: int,
        head_dim: int,
        compress_ratio: int,
        num_layers: int,
        compressed_dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Per-token QSA index-cache cost: compressed keys only;
        the per-request pending ring is budgeted with the other per-request buffers."""
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=compressed_dtype
        )
        return index_k_bytes // compress_ratio * num_layers

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: list[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        cache_sharding_runtime: QSACacheShardingRuntime | None = None,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: int | None = None,
        full_kv_pool_class: type | None = None,
        quant_method=None,
        post_capture_active: bool = False,
        qsa_indexer_dtype: torch.dtype = torch.bfloat16,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. This needs the mamba "
                "extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # super().__init__ computes mem_usage via the overridden get_kv_size_bytes,
        # so the QSA buffers get placeholders first; mem_usage is recomputed last.
        self.qsa_key_state_buffer_pool = []
        self.qsa_compressed_k_buffer_pool = []
        self.qsa_rope_position_buffer = torch.empty(0)
        topology_runtime = get_qsa_cache_sharding_runtime()
        if cache_sharding_runtime is None:
            cache_sharding_runtime = topology_runtime
        assert_qsa_cache_sharding_runtime_match(
            cache_sharding_runtime,
            topology_runtime,
            component="pool",
        )
        self.cache_sharding_runtime = cache_sharding_runtime
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if (
            min(
                qsa_index_kv_heads,
                qsa_index_head_dim,
                qsa_compress_ratio,
                qsa_token_topk,
            )
            <= 0
        ):
            raise ValueError("QSA cache configuration values must be positive")
        if qsa_token_topk % qsa_compress_ratio != 0:
            raise ValueError("qsa_token_topk must be divisible by qsa_compress_ratio")
        self.qsa_compress_ratio = int(qsa_compress_ratio)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_token_topk = int(qsa_token_topk)
        self.qsa_block_topk = self.qsa_token_topk // self.qsa_compress_ratio
        if qsa_indexer_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "QSA compressed indexer cache dtype must be bfloat16 or "
                f"float8_e4m3fn, got {qsa_indexer_dtype}"
            )
        # Storage dtype of the compressed keys and the index Q (the GEMM operands).
        self.qsa_compressed_dtype = qsa_indexer_dtype
        logger.info(
            "QSA compressed indexer cache dtype: %s (pending ring %s)",
            self.qsa_compressed_dtype,
            self.index_state_dtype,
        )
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_global_compressed_capacity = qsa_global_compressed_capacity(
            state_size,
            compress_ratio=self.qsa_compress_ratio,
            world_size=1,
        )
        self.qsa_compressed_capacity = self.qsa_global_compressed_capacity
        self.qsa_raw_kv_sharding = None
        self.qsa_owner_group = None
        # Pre-compression index-K state is a per-request ring, not a per-token cache:
        # only the pending group's ``ratio`` members must survive a forward,
        # addressed as ``req_pool_idx * ratio + position % ratio``.
        # Request slot 0 is never allocated, so rows [0, ratio) are the inert dump.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        self.qsa_num_request_slots = int(num_request_slots)
        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio
        self.qsa_compressed_sharding = None
        if cache_sharding_runtime.enabled:
            self.qsa_global_compressed_capacity = qsa_global_compressed_capacity(
                state_size,
                compress_ratio=self.qsa_compress_ratio,
                world_size=cache_sharding_runtime.size,
            )
            self.qsa_owner_group = cache_sharding_runtime.group
            self.qsa_raw_kv_sharding = QSARawKVSharding(
                local_capacity=size + page_size,
                page_size=page_size,
                world_size=cache_sharding_runtime.size,
                rank=cache_sharding_runtime.rank,
            )
            self.qsa_compressed_sharding = QSACompressedBlockSharding(
                global_blocks=self.qsa_global_compressed_capacity,
                compressed_page_size=self.qsa_compressed_page_size,
                world_size=cache_sharding_runtime.size,
                rank=cache_sharding_runtime.rank,
            )
            self.qsa_compressed_capacity = self.qsa_compressed_sharding.local_blocks
        # These buffers participate in Mooncake PD transfer just like the base
        # KV and Mamba pools.  Keep their allocation in the same memory-saver
        # and Mooncake custom-pool regions; otherwise MNNVL cannot resolve the
        # ordinary CUDA allocation when the first QSA state page is sent.
        allocation_pool = self.full_kv_pool
        with (
            allocation_pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(allocation_pool.custom_mem_pool)
                if allocation_pool.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            self.qsa_key_state_buffer_pool = [
                torch.zeros(
                    (
                        ring_slots,
                        self.qsa_index_kv_heads,
                        self.qsa_index_head_dim,
                    ),
                    dtype=self.index_state_dtype,
                    device=device,
                )
                for _ in full_attention_layer_ids
            ]
            # RoPE coordinates are layer-independent. Keep the exact Qwen4-Exp
            # MRoPE position of every incomplete key so compression can rotate
            # the pooled key with the group's real starting coordinate.
            self.qsa_rope_position_buffer = torch.zeros(
                (ring_slots, 3), dtype=torch.int64, device=device
            )
            # One contiguous allocation behind per-layer views: every layer's
            # compressed pages are addressable from a single base pointer.
            self.qsa_compressed_flat = torch.zeros(
                (
                    len(full_attention_layer_ids),
                    self.qsa_compressed_capacity
                    * self.qsa_index_kv_heads
                    * self.qsa_index_head_dim,
                ),
                dtype=self.qsa_compressed_dtype,
                device=device,
            )
        self.qsa_compressed_k_buffer_pool = [
            self.qsa_compressed_flat[layer_offset].view(
                self.qsa_compressed_capacity,
                self.qsa_index_kv_heads,
                self.qsa_index_head_dim,
            )
            for layer_offset in range(len(full_attention_layer_ids))
        ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_qsa_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        dcp_kv_mask: torch.Tensor | None = None,
    ) -> None:
        sharding = self.qsa_raw_kv_sharding
        if sharding is not None:
            local_layer_id = self._transfer_full_attention_id(layer.layer_id)
            key_buffer = self.full_kv_pool.get_key_buffer(local_layer_id)
            value_buffer = self.full_kv_pool.get_value_buffer(local_layer_id)
            buffer_heads = key_buffer.shape[1]
            buffer_dim = key_buffer.shape[-1]
            if (
                cache_k.ndim != 3
                or cache_v.shape != cache_k.shape
                or value_buffer.shape != key_buffer.shape
                or cache_k.shape[1] != buffer_heads
                or cache_k.shape[2] != buffer_dim
            ):
                raise ValueError(
                    "raw K/V write layout must match the owner-local cache: "
                    f"cache={tuple(cache_k.shape)}, buffer={tuple(key_buffer.shape)}"
                )
            owner_mask, loc = sharding.local_write_targets(loc)
            if dcp_kv_mask is not None:
                owner_mask &= dcp_kv_mask.to(device=owner_mask.device, dtype=torch.bool)
            dcp_kv_mask = owner_mask
        super().set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            dcp_kv_mask=dcp_kv_mask,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        sharding = self.qsa_raw_kv_sharding
        if sharding is not None:
            src_loc, tgt_loc = sharding.local_copy_targets(src_loc, tgt_loc)
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def clear_raw_kv_slots(self, global_slots: torch.Tensor) -> None:
        sharding = self.qsa_raw_kv_sharding
        if sharding is None:
            local_slots = global_slots
        else:
            local_slots = sharding.global_to_local_tokens(global_slots)
            local_slots = local_slots[local_slots >= 0]
        if local_slots.numel() == 0:
            return
        layer_ids = self.full_attention_layer_id_mapping.values()
        for layer_id in layer_ids:
            key = self.full_kv_pool.get_key_buffer(layer_id)
            value = self.full_kv_pool.get_value_buffer(layer_id)
            key[local_slots] = 0
            value[local_slots] = 0

    def set_qsa_key_state_buffer(
        self, layer_id: int, loc: torch.Tensor, token_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_key_state_buffer(layer_id)
        buffer[loc.long()] = token_k.to(buffer.dtype)

    def set_qsa_rope_position_buffer(
        self, loc: torch.Tensor, positions: torch.Tensor
    ) -> None:
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"QSA RoPE positions must be [tokens] or [3, tokens], got {positions.shape}"
            )
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc: torch.Tensor) -> torch.Tensor:
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id: int) -> torch.Tensor:
        # The indexer reads compressed keys before attention reads the full KV.
        self._wait_for_layer(layer_id)
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        if self.qsa_compressed_sharding is not None:
            local_blocks = self.qsa_compressed_sharding.global_to_local_blocks(loc)
            source_rows = torch.nonzero(
                local_blocks >= 0, as_tuple=False
            ).flatten()
            loc = local_blocks.index_select(0, source_rows)
            compressed_k = compressed_k.index_select(0, source_rows)
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

    @staticmethod
    def _get_paged_state_buf_infos(tensors, page_size: int):
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [tensor[0].nbytes * page_size for tensor in tensors],
        )

    def get_qsa_pending_state_buf_infos(self):
        """Per-request pending key-state and RoPE ring transfer buffers."""
        # A PP stage without a local QSA layer never writes the shared RoPE
        # ring.  Do not register it as a transfer source: otherwise that stage
        # can race with a QSA-owning stage and overwrite valid positions with
        # its zero-initialized or stale contents.
        if not self.full_attention_layer_id_mapping:
            return [], [], []
        tensors = [*self.qsa_key_state_buffer_pool, self.qsa_rope_position_buffer]
        return self._get_paged_state_buf_infos(
            tensors,
            self.qsa_compress_ratio,
        )

    def get_qsa_pending_state_layer_ids(self):
        """Global layer metadata for the compact QSA pending-state list."""
        if not self.full_attention_layer_id_mapping:
            return []
        return [
            *self.full_attention_layer_id_mapping.keys(),
            QSA_ROPE_STATE_LAYER_ID,
        ]

    def get_qsa_compressed_state_layer_ids(self):
        """Global layer metadata for the compact compressed-K list."""
        return list(self.full_attention_layer_id_mapping.keys())

    def get_qsa_compressed_state_buf_infos(self):
        """Per-full-page compressed-K transfer buffers.

        One full KV page maps to one compressed page because the full page size
        is an integer multiple of the compression ratio.
        """
        return self._get_paged_state_buf_infos(
            self.qsa_compressed_k_buffer_pool,
            self.qsa_compressed_page_size,
        )

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        qsa_k_size = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_key_state_buffer_pool
            )
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_k_buffer_pool
            )
            + self.qsa_rope_position_buffer.numel() * 8
        )
        return k_size + qsa_k_size, v_size
