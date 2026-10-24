"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-request pending
index-key/RoPE ring and the paged compressed-K cache on top of the hybrid
full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool

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
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
        qsa_indexer_dtype: torch.dtype = torch.bfloat16,
        logical_size: Optional[int] = None,
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
        # ``size`` is the number of device slots.  A host-tier pool (HiSparse)
        # holds ``logical_size`` slots in host memory and keeps the indexer
        # cache for all of them on device; ``logical_size`` defaults to
        # ``size`` for every other pool.
        state_size = (logical_size or size) + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
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


@triton.jit
def _qsa_host_kv_scatter_kernel(
    k_ptr,
    v_ptr,
    hk_ptr,
    hv_ptr,
    dk_ptr,
    dv_ptr,
    idx_ptr,
    n_rows,
    row_dim: tl.constexpr,
    RESIDENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write one step's K/V through to both KV tiers, capture-safe.

    Row ``r`` lands in host row ``dst = idx[r]`` and, when that slot is inside
    the static resident window (``0 < dst < RESIDENT``), also in the device row
    of the same index.  Both decisions are per-program scalars and the launch
    geometry is just ``n_rows``: no data-dependent shape (the old
    ``logical[resident]`` gather changed length with the batch), no D2H copy and
    no host sync, so the launch stays legal inside a decode CUDA graph.  The
    pinned host mirror is plain UVA memory, which device code addresses
    directly.
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    dst = tl.load(idx_ptr + row)
    if dst <= 0:
        return
    dst_off = dst.to(tl.int64) * row_dim
    for c in range(0, row_dim, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        m = cols < row_dim
        off_src = row * row_dim + cols
        k = tl.load(k_ptr + off_src, mask=m, other=0.0)
        v = tl.load(v_ptr + off_src, mask=m, other=0.0)
        tl.store(hk_ptr + dst_off + cols, k, mask=m)
        tl.store(hv_ptr + dst_off + cols, v, mask=m)
        if dst < RESIDENT:
            tl.store(dk_ptr + dst_off + cols, k, mask=m)
            tl.store(dv_ptr + dst_off + cols, v, mask=m)


class HiSparseQSATokenToKVPool(QSATokenToKVPool):
    """QSA pool with a host tier for the full-attention K/V.

    The logical (host) slot space is ``host_to_device_ratio`` times the device
    slot space, exactly like the other HiSparse pools.  QSA stores plain MHA
    K/V (``kv_heads x head_dim`` bf16 per token per layer), so the paged device
    tier is an :class:`HiSparseMHAMainPool`; the indexer caches
    (``qsa_key_state_buffer_pool`` / ``qsa_compressed_k_buffer_pool``) stay
    device-resident and only their slot space grows with the ratio.

    P0 residency policy: a static prefix window.  Logical slot ids below
    ``resident_tokens`` are read from the device tier at the *same* row index;
    every other slot id is read through the pinned host copy (the kernels get
    ``resident_tokens`` and treat ``dslot <= 0`` as "read host").  This is a
    pure address-translation mask: no device page is swapped, and
    ``HiSparseTokenToKVPoolAllocator`` never has to allocate device pages for
    the window.  ``resident_tokens=0`` keeps every K/V row on the host.
    """

    is_hisparse_qsa = True

    def __init__(
        self,
        *,
        host_to_device_ratio: int = 2,
        resident_tokens: Optional[int] = None,
        **kwargs,
    ):
        if host_to_device_ratio < 1:
            raise ValueError(
                f"host_to_device_ratio must be >= 1, got {host_to_device_ratio}"
            )
        self.host_to_device_ratio = int(host_to_device_ratio)
        self._logical_size = kwargs["size"] * self.host_to_device_ratio
        super().__init__(**kwargs, logical_size=self._logical_size)
        full = self.full_kv_pool
        if not getattr(full, "register_mapping", None):
            raise ValueError(
                "HiSparseQSATokenToKVPool requires a HiSparse full-KV pool; "
                f"got {type(full).__name__}"
            )
        if resident_tokens is None:
            resident_tokens = self.size
        # The identity window can only cover rows the device tier really has
        # (the full KV pool was sized for ``size`` device slots).
        self.hisparse_resident_tokens = min(int(resident_tokens), self.size)
        # Static (allocation-time) answer to "can any logical slot ever be read
        # from the host tier?".  When the resident window covers the whole
        # logical space the host mirror is unreachable on the read path, so the
        # write path can skip it entirely -- which also keeps the write legal
        # inside a decode CUDA graph (capture forbids D2H and host syncs).
        self._host_tier_in_use = self.hisparse_resident_tokens < self._logical_size
        # Host rows are addressed by logical slot id, so there is one row per
        # logical slot plus the base pool's padding page.  Pinned memory keeps
        # the zero-copy device reads cheap; rows are bf16 like the device tier.
        rows = self._logical_size + self.page_size
        kv_heads = full.head_num
        head_dim = full.head_dim
        host_kwargs = dict(
            dtype=full.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.host_key_buffer_pool = [
            torch.zeros((rows, kv_heads, head_dim), **host_kwargs)
            for _ in self.full_attention_layer_id_mapping
        ]
        self.host_value_buffer_pool = [
            torch.zeros((rows, kv_heads, head_dim), **host_kwargs)
            for _ in self.full_attention_layer_id_mapping
        ]

    # ---- address translation -------------------------------------------------
    def _translate_loc_to_hisparse_device(self, loc: torch.Tensor) -> torch.Tensor:
        """Logical slot id -> device row for the resident window, else -1."""
        keep = (loc > 0) & (loc < self.hisparse_resident_tokens)
        return torch.where(keep, loc, torch.full_like(loc, -1))

    @property
    def resident_device_bytes(self) -> int:
        if not self.host_key_buffer_pool:
            return 0
        per_row = self.host_key_buffer_pool[0][0].numel() * 2
        return (
            self.hisparse_resident_tokens * per_row * 2 * len(self.host_key_buffer_pool)
        )

    def register_mapping(self, mapping) -> None:
        """Receive the logical->device slot table from the HiSparse allocator.

        P0 maps slots by a static window (see ``hisparse_resident_tokens``), so
        the table itself is only kept for bookkeeping and forwarded to the
        device tier, whose write path needs it.
        """
        self.full_to_hisparse_device_index_mapping = mapping
        self.full_kv_pool.register_mapping(mapping)

    def get_host_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.host_key_buffer_pool[self._transfer_full_attention_id(layer_id)]

    def get_host_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.host_value_buffer_pool[self._transfer_full_attention_id(layer_id)]

    # ---- write path (write-through) -----------------------------------------
    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, *args, **kwargs) -> None:
        if args or kwargs:
            raise NotImplementedError(
                "HiSparseQSATokenToKVPool.set_kv_buffer does not take scales"
            )
        from sglang.srt.mem_cache.memory_pool import (
            unwrap_write_loc,
            write_loc_is_physical,
        )

        if write_loc_is_physical(loc_info):
            raise NotImplementedError("HiSparse QSA writes must carry logical slot ids")
        loc, _, full_loc = unwrap_write_loc(loc_info)
        logical = loc if full_loc is None else full_loc
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        k_buf = self.full_kv_pool.k_buffer[layer_id]
        v_buf = self.full_kv_pool.v_buffer[layer_id]
        if not self._host_tier_in_use:
            # Every logical slot is read back from the device tier, so this is a
            # plain fixed-shape scatter: no host round trip, no device->host
            # sync, hence legal inside a decode CUDA graph.
            k_buf[logical] = cache_k.to(k_buf.dtype)
            v_buf[logical] = cache_v.to(v_buf.dtype)
            return
        # Mixed tier: one fixed-shape launch writes the pinned host mirror and,
        # only for slots inside the resident window, the device tier.  Nothing
        # here branches on tensor contents, so this is legal inside a decode
        # CUDA graph (see ``_qsa_host_kv_scatter_kernel``).
        n_rows = int(logical.numel())
        if n_rows == 0:
            return
        row_dim = int(k_buf.shape[1] * k_buf.shape[2])
        _qsa_host_kv_scatter_kernel[(n_rows,)](
            cache_k.to(k_buf.dtype).reshape(-1),
            cache_v.to(v_buf.dtype).reshape(-1),
            self.host_key_buffer_pool[layer_id].reshape(-1),
            self.host_value_buffer_pool[layer_id].reshape(-1),
            k_buf.reshape(-1),
            v_buf.reshape(-1),
            logical.reshape(-1),
            n_rows,
            row_dim=row_dim,
            RESIDENT=self.hisparse_resident_tokens,
            BLOCK=min(1024, triton.next_power_of_2(row_dim)),
            num_warps=4,
        )
