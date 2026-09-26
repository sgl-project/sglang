"""Compressed INT8 HiCache host (L2) pool for dense MHA models.

Experimental. Activated by ``SGLANG_EXPERIMENTAL_HICACHE_INT8=1``; see
``get_mha_host_pool_cls`` in ``pool_host/mha.py`` for the dispatch.

Thesis
------
Keep active L1 KV in BF16 for attention, compress KV when it is evicted to the
CPU L2 cache, and restore it to BF16 on an L2 hit. SGLang's caching policy,
radix tree, scheduler and L2 transfer engine are all unchanged -- only the L2
*representation* differs. That is the whole point: the experimental surface is
one new pool class plus a one-branch dispatch.

Storage
-------
One 1152-byte aligned record per ``(layer, K|V, token)``::

    0     .. 1023   1024 INT8 payload   (head_num x head_dim bytes)
    1024  .. 1039   8 BF16 per-head scales
    1040  .. 1151   112 bytes padding

``1152 = 9 x 128`` keeps the row on SGLang's CUDA JIT HiCache mover, which
requires ``element_size % 128 == 0``. For Qwen3-8B at TP=1 that is
82,944 B/token against a 147,456 B baseline: 43.75% fewer host bytes, 1.78x the
L2 token capacity for the same ``--hicache-size``.

Data movement
-------------
The mover cannot convert widths: ``transfer_hicache_*`` takes separate source and
destination strides but a single ``element_size``, so a 2048-byte BF16 device row
can never be written straight into a 1152-byte encoded row. Both directions
therefore stage through a same-width device buffer:

    D2H   device BF16 -> encode -> staging [layer_num, N, 1152] -> mover -> host arena
    H2D   host arena -> mover -> staging [layer_num, N, 1152] -> decode -> device BF16

Both directions then move bytes with ``element_size = 1152``, which is exactly
the width on both sides of the mover.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import int8_codec as codec
from sglang.srt.mem_cache.pool_host import int8_staging as staging
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

logger = logging.getLogger(__name__)

#: Layouts this pool can address. Only ``layer_first`` gives the per-layer
#: contiguous arena the staging design needs.
SUPPORTED_LAYOUTS = ("layer_first",)

#: io_backend values the pool can use. ``direct`` and ``kernel_ascend`` both
#: assume raw BF16 rows in the host buffer.
SUPPORTED_IO_BACKENDS = ("kernel",)


def _retire(buffers: staging.StagingBuffers, device: torch.device | str) -> None:
    """Make a replaced staging allocation safe to free.

    Dropping the last Python reference to a CUDA tensor returns its block to the
    caching allocator, which may then hand the same memory to a new allocation.
    If a transfer stream still has work queued against the old block, that work
    would read memory that has since been reused -- corruption that surfaces as
    wrong KV rather than as an error.

    ``record_stream`` defers reuse until every stream recorded on the tensor has
    passed the point where it was used. The transfer streams are named explicitly
    rather than taken from ``torch.cuda.current_stream()``: the transfer engine
    wraps each submission in ``device_module.stream(...)``, so the work was
    enqueued on one of those two, and "current" here means whatever stream the
    caller happened to be on.

    Both directions are recorded because either may hold work against staging.
    Called on growth only; the steady-state path allocates nothing.
    """
    if torch.device(device).type != "cuda":
        return
    # Imported lazily: l2_transfer is imported by the controller that builds this
    # pool, so a module-level import here would be circular.
    try:
        from sglang.srt.mem_cache import l2_transfer
    except ImportError:  # pragma: no cover - only during partial installs
        return
    for stream in (
        getattr(l2_transfer, "device_to_host_stream", None),
        getattr(l2_transfer, "host_to_device_stream", None),
    ):
        if stream is None:
            continue
        for tensor in (buffers.k, buffers.v):
            try:
                tensor.record_stream(stream)
            except Exception:  # noqa: BLE001 - cleanup must never break a transfer
                pass


class MHATokenToKVPoolHostINT8(MHATokenToKVPoolHost):
    """MHA host pool that stores INT8 payload plus BF16 per-head scales.

    Host layout (``layer_first``)::

        kv_buffer: [2, layer_num, size, 1152] uint8

    so ``kv_buffer[0, layer, slot]`` is one K record and ``kv_buffer[1, layer,
    slot]`` one V record. ``get_size_per_token()`` returns the *encoded*
    ``2 * layer_num * 1152``, which is what makes ``--hicache-size`` buy 1.78x
    more L2 tokens than the BF16 pool for the same host memory.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
    ):
        self._validate_configuration(
            device_pool, page_size, layout, mtp_draft_device_pools
        )

        # Geometry must be set before super().__init__, because HostKVCache
        # calls get_size_per_token() *before* init_kv_buffer() to size the arena.
        # get_size_per_token() derives these from device_pool, and the codec
        # geometry check below needs them too.
        codec.check_layout(
            device_pool.row_dim // device_pool.head_dim,
            device_pool.head_dim,
            device_pool.store_dtype.itemsize,
        )

        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            mtp_draft_device_pools=mtp_draft_device_pools,
            pool_label=pool_label,
        )

        self._init_staging_buffers()
        self._log_configuration()

    # ------------------------------------------------------------------
    # Configuration validation (fail fast)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_configuration(
        device_pool, page_size: int, layout: str, mtp_draft_device_pools: Sequence
    ) -> None:
        """Reject unsupported configurations at construction time.

        A silently-wrong compressed KV cache corrupts generations without
        crashing, so every unsupported combination raises here rather than
        producing plausible-looking output later.
        """
        if layout != "layer_first":
            raise NotImplementedError(
                f"INT8 HiCache host pool requires --hicache-mem-layout layer_first, "
                f"got {layout!r}. The staging design addresses the arena one "
                f"contiguous layer at a time."
            )
        if page_size != 1:
            raise NotImplementedError(
                f"INT8 HiCache host pool requires --page-size 1, got {page_size}. "
                f"Records are per token and the codec is position independent, so "
                f"multi-token pages are not representable in v1."
            )
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "INT8 HiCache host pool does not support packed MTP/draft KV layers."
            )
        if getattr(device_pool, "is_quantized_kv_cache", False):
            raise NotImplementedError(
                "INT8 HiCache host pool requires a BF16/FP16 device KV pool; the "
                "device pool already uses a quantized KV cache dtype."
            )
        if device_pool.store_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(
                f"INT8 HiCache host pool requires a BF16 or FP16 device pool, got "
                f"{device_pool.store_dtype}."
            )
        if device_pool.head_dim != device_pool.v_head_dim:
            raise NotImplementedError(
                f"INT8 HiCache host pool requires symmetric K/V head dims, got "
                f"head_dim={device_pool.head_dim} v_head_dim={device_pool.v_head_dim}."
            )
        if getattr(device_pool, "use_hnd", False):
            raise NotImplementedError(
                "INT8 HiCache host pool requires the NHD device KV layout."
            )
        # TP=1 is a hard restriction, not an accident of the payload width. The
        # record is fixed at 8 heads x 128 dims == 1024 payload bytes plus 16
        # scale bytes; at TP=2 there are 4 local KV heads (512 + 8 bytes), so the
        # padding, the alignment proof, the capacity maths, the JIT geometry and
        # every benchmark change with it. Reject it by name so the failure is
        # legible instead of surfacing as a payload/geometry mismatch inside
        # codec.check_layout.
        local_kv_heads = device_pool.row_dim // device_pool.head_dim
        if local_kv_heads != codec.EXPECTED_KV_HEADS:
            raise NotImplementedError(
                f"INT8 HiCache L2 records support {codec.EXPECTED_KV_HEADS} "
                f"local KV heads only ({codec.EXPECTED_KV_HEADS} x "
                f"{device_pool.head_dim} dims = {codec.PAYLOAD_BYTES} payload "
                f"bytes). This device pool exposes {local_kv_heads} local KV heads, "
                f"so the model or TP size differs. TP-sharded record formats are "
                f"future work."
            )
        if getattr(device_pool, "layer_shard_enabled", False):
            raise NotImplementedError(
                "INT8 HiCache host pool does not support layer-sharded device pools."
            )
        # backup_from_device_all_layer walks device_pool.k_buffer[0 .. layer_num)
        # and relies on those being the whole picture. Assert that here rather
        # than discovering a partial transfer later as wrong output.
        #
        # ``end_layer`` is INCLUSIVE -- the last valid layer index, set by
        # KVCache as ``end_layer or layer_num - 1`` (memory_pool.py:1857) -- so a
        # full pool reports layer_num - 1, not layer_num. MHATokenToKVPool does
        # accept start_layer/end_layer (memory_pool.py:1988), so a pool built
        # with an explicit sub-range really does reach this check.
        start_layer = getattr(device_pool, "start_layer", 0)
        end_layer = getattr(device_pool, "end_layer", device_pool.layer_num - 1)
        if start_layer != 0 or end_layer < device_pool.layer_num - 1:
            raise NotImplementedError(
                f"INT8 HiCache host pool expects a device pool covering every layer "
                f"from 0, got start_layer={start_layer} end_layer={end_layer} "
                f"(inclusive) for layer_num={device_pool.layer_num}."
            )
        # ``k_buffer`` is None until the device pool has created its buffers, so
        # only assert the count when there is one to count.
        kb = getattr(device_pool, "k_buffer", None)
        if kb is not None and len(kb) != device_pool.layer_num:
            raise NotImplementedError(
                f"INT8 HiCache host pool expects one device K buffer per layer, got "
                f"{len(kb)} buffers for layer_num={device_pool.layer_num}."
            )
        if device_pool.page_size != page_size:
            raise NotImplementedError(
                f"INT8 HiCache host pool requires host page_size to match the device "
                f"pool page_size; got host {page_size} vs device "
                f"{device_pool.page_size}."
            )
        # Only the mover this pool actually calls. The staged write-back kernel
        # (can_use_write_back_jit_kernel) is used by the inherited page_first
        # path, which this pool rejects outright, and its module is never
        # imported here -- requiring it would reject a working system because an
        # unused kernel failed to build.
        if not can_use_hicache_jit_kernel(
            element_size=codec.ROW_BYTES,
            page_size=page_size,
        ):
            raise NotImplementedError(
                f"INT8 HiCache host pool needs the JIT HiCache kernel for "
                f"{codec.ROW_BYTES}-byte rows (CUDA or HIP)."
            )

    def _log_configuration(self) -> None:
        baseline = (
            2
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.device_pool.store_dtype.itemsize
        )
        logger.info(
            "HiCache INT8 host pool [%s]: %d tokens, %.2f GB host memory, "
            "size_per_token=%d B (baseline %d B, reduction %.2f%%, capacity %.3fx).",
            self.pool_label,
            self.size,
            self.size * self.size_per_token / 1e9,
            self.size_per_token,
            baseline,
            100.0 * (1 - self.size_per_token / baseline),
            baseline / self.size_per_token,
        )

    # ------------------------------------------------------------------
    # Host arena
    # ------------------------------------------------------------------

    def get_size_per_token(self):
        """Encoded host bytes per token -- NOT the inherited BF16 figure.

        ``HostKVCache.__init__`` uses this to convert ``--hicache-size`` into a
        token capacity, so returning the encoded size here is precisely what
        turns a fixed host-memory budget into 1.78x more cached tokens.

        Called before ``init_kv_buffer``, hence the explicit geometry setup.
        """
        self.head_num = self.device_pool.row_dim // self.device_pool.head_dim
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        return codec.bytes_per_token(self.layer_num)

    def init_kv_buffer(self):
        """Allocate the encoded arena: ``[2, layer_num, size, ROW_BYTES]`` uint8.

        Deliberately does not allocate the inherited BF16 ``kv_buffer``. The
        shape mirrors the parent's ``layer_first`` layout so that
        ``k_data_refs`` / ``v_data_refs`` and every downstream index domain stay
        identical to the BF16 pool.
        """
        if self.layout != "layer_first":
            raise NotImplementedError(
                f"INT8 HiCache host pool supports only layer_first, got {self.layout!r}."
            )
        dims = (2, self.layer_num, self.size, codec.ROW_BYTES)
        alloc_func = self._alloc_func()
        return alloc_func(
            dims,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            # Matches the parent's layer_first behaviour: the inherited pool
            # passes a custom granularity only for the page-oriented layouts.
            # None keeps the single-call registration, which is correct here
            # because the whole arena is one contiguous token-major region.
            registration_granularity_bytes=None,
        )

    def _alloc_func(self):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        return ALLOC_MEMORY_FUNCS[self.device_pool.device]

    def get_ksize_per_token(self):
        return codec.ROW_BYTES * self.layer_num

    # ------------------------------------------------------------------
    # Device staging
    # ------------------------------------------------------------------

    def _init_staging_buffers(self) -> None:
        """Allocate the directional staging buffers and the D2H pointer tables.

        D2H and H2D get separate buffers because the two transfer streams can be
        in flight together. The D2H host pointer tables are built eagerly (they
        point at the arena, which never moves); the D2H staging tables are
        rebuilt only when the buffers grow.
        """
        initial = max(envs.SGLANG_HICACHE_INT8_STAGING_TOKENS.get(), 1)
        self._d2h = staging.allocate_staging(
            self.layer_num,
            codec.ROW_BYTES,
            device=self.device_pool.device,
            capacity=initial,
        )
        self._h2d = staging.allocate_staging(
            self.layer_num,
            codec.ROW_BYTES,
            device=self.device_pool.device,
            capacity=initial,
        )
        self._rebuild_d2h_staging_tables()
        # Staging row indices, one list per direction. They are deliberately not
        # shared: the two streams are independent, so a single list that one
        # direction extends while the other still has queued work against it
        # would be the same lifetime hazard the buffers themselves had.
        self._d2h_indices = torch.arange(
            initial, dtype=torch.int64, device=self.device_pool.device
        )
        self._h2d_indices = torch.arange(
            initial, dtype=torch.int64, device=self.device_pool.device
        )
        # Host arena destination tables: one entry per layer, fixed for the
        # lifetime of the pool.
        self._d2h_k_dst_ptrs = staging.pointer_table(
            self.k_data_refs, device=self.device_pool.device
        )
        self._d2h_v_dst_ptrs = staging.pointer_table(
            self.v_data_refs, device=self.device_pool.device
        )

    def _rebuild_d2h_staging_tables(self) -> None:
        self._d2h_k_src_ptrs = staging.pointer_table(
            self._d2h.k_layer_views(self._d2h.capacity),
            device=self.device_pool.device,
        )
        self._d2h_v_src_ptrs = staging.pointer_table(
            self._d2h.v_layer_views(self._d2h.capacity),
            device=self.device_pool.device,
        )

    def _ensure_d2h_capacity(self, num_tokens: int) -> None:
        """Grow the D2H staging buffer so it can hold ``num_tokens`` rows.

        Grows **only** the D2H pair. A D2H backup must never reallocate the H2D
        buffer: the two directions run on independent streams and can be in
        flight simultaneously, so freeing H2D storage here would pull it out from
        under a load that is still reading it. Each direction grows only for its
        own transfers, and always before it enqueues its own copy.
        """
        capacity = staging.next_staging_capacity(num_tokens, self._d2h.capacity)
        if capacity == self._d2h.capacity:
            return
        logger.info(
            "HiCache INT8 D2H staging grew from %d to %d tokens (%d layers, %d B/row).",
            self._d2h.capacity,
            capacity,
            self.layer_num,
            codec.ROW_BYTES,
        )
        old = self._d2h
        self._d2h = staging.allocate_staging(
            self.layer_num,
            codec.ROW_BYTES,
            device=self.device_pool.device,
            capacity=capacity,
        )
        _retire(old, self.device_pool.device)
        # The staging pointer tables embed data_ptr() values, so they are only
        # valid for the allocation they were built from.
        self._rebuild_d2h_staging_tables()
        self._d2h_indices = torch.arange(
            capacity, dtype=torch.int64, device=self.device_pool.device
        )

    def _ensure_h2d_capacity(self, num_tokens: int) -> None:
        """Grow the H2D staging buffer. Deliberately never touches the D2H pair."""
        capacity = staging.next_staging_capacity(num_tokens, self._h2d.capacity)
        if capacity == self._h2d.capacity:
            return
        logger.info(
            "HiCache INT8 H2D staging grew from %d to %d tokens (%d layers, %d B/row).",
            self._h2d.capacity,
            capacity,
            self.layer_num,
            codec.ROW_BYTES,
        )
        old = self._h2d
        self._h2d = staging.allocate_staging(
            self.layer_num,
            codec.ROW_BYTES,
            device=self.device_pool.device,
            capacity=capacity,
        )
        _retire(old, self.device_pool.device)
        self._h2d_indices = torch.arange(
            capacity, dtype=torch.int64, device=self.device_pool.device
        )

    def _d2h_index_view(self, num_tokens: int) -> torch.Tensor:
        """First ``num_tokens`` staging rows, for the D2H mover's source indices."""
        return self._d2h_indices[:num_tokens]

    def _h2d_index_view(self, num_tokens: int) -> torch.Tensor:
        """First ``num_tokens`` staging rows, for the H2D mover's dest indices."""
        return self._h2d_indices[:num_tokens]

    # ------------------------------------------------------------------
    # D2H: encode, then move
    # ------------------------------------------------------------------

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """Encode every layer's KV for ``device_indices`` into the host arena.

        Runs entirely on the caller's current stream, which ``L2TransferEngine``
        has already set to ``device_to_host_stream``. That is what makes the
        per-layer sequence below safe without any extra synchronisation:

            encode layer 0 .. encode layer N-1 .. one all-layer move

        and it is why the single copy cannot observe a half-encoded staging
        buffer.
        """
        if io_backend not in SUPPORTED_IO_BACKENDS:
            raise NotImplementedError(
                f"INT8 HiCache host pool supports only io_backend='kernel', got "
                f"{io_backend!r}."
            )

        num_tokens = int(device_indices.numel())
        if num_tokens == 0:
            return
        self._ensure_d2h_capacity(num_tokens)

        device_indices = device_indices.to(torch.int64)
        head_num, head_dim = self.head_num, self.head_dim
        # Encode time: index_select + amax/round/clamp/cast + record packing.
        # Measured separately from the mover below so the codec's share of the
        # L2 path is attributable.
        # k_buffer / v_buffer are indexed by local layer, and validation above
        # guarantees the device pool covers every layer with start_layer == 0.
        for layer in range(device_pool.layer_num):
            rows = device_pool.k_buffer[layer].index_select(0, device_indices)
            codec.write_record(
                rows.reshape(num_tokens, head_num, head_dim),
                self._d2h.layer_k(layer, num_tokens),
            )
            rows = device_pool.v_buffer[layer].index_select(0, device_indices)
            codec.write_record(
                rows.reshape(num_tokens, head_num, head_dim),
                self._d2h.layer_v(layer, num_tokens),
            )

        # One byte-for-byte move of every layer. Both sides are ROW_BYTES wide,
        # so a single element_size is correct on both.
        jit_transfer_hicache_all_layer(
            page_size=1,
            k_ptr_dst=self._d2h_k_dst_ptrs,
            v_ptr_dst=self._d2h_v_dst_ptrs,
            indices_dst=host_indices,
            k_ptr_src=self._d2h_k_src_ptrs,
            v_ptr_src=self._d2h_v_src_ptrs,
            indices_src=self._d2h_index_view(num_tokens),
            kv_cache_src_stride_bytes=codec.ROW_BYTES,
            kv_cache_dst_stride_bytes=codec.ROW_BYTES,
            element_size=codec.ROW_BYTES,
        )

    # ------------------------------------------------------------------
    # H2D: move, then decode
    # ------------------------------------------------------------------

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ) -> None:
        """Decode one layer's records from the host arena into device BF16.

        Everything is enqueued on ``host_to_device_stream`` before this returns.
        That is a hard requirement, not a style choice: ``L2TransferEngine``
        calls ``on_layer_done(layer_id)`` immediately after this function
        returns, and the model's forward stream waits on exactly that per-layer
        event. Returning before the dequantise is enqueued would let attention
        read partially-written BF16 KV -- a data race that shows up as
        intermittently wrong tokens rather than a crash.
        """
        if io_backend not in SUPPORTED_IO_BACKENDS:
            raise NotImplementedError(
                f"INT8 HiCache host pool supports only io_backend='kernel', got "
                f"{io_backend!r}."
            )
        if is_draft:
            raise NotImplementedError("INT8 HiCache host pool has no draft layers.")

        # Inherited guard: resolves host/device layer ids and returns early for
        # device layers this rank does not own.
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer_id = self._host_layer_index(layer_id)

        num_tokens = int(device_indices.numel())
        if num_tokens == 0:
            return
        self._ensure_h2d_capacity(num_tokens)

        # Move the encoded records. Both sides are already ROW_BYTES-wide byte
        # buffers, so the transfer is a straight byte copy at element_dim =
        # ROW_BYTES with uint8 on both sides:
        #
        #     dtype uint8 <-> uint8, element_dim 1152, itemsize 1
        #     => element_size 1152 bytes, matching kElementSize
        #
        # Do NOT reinterpret one side only. run_one() binds a single
        # SymbolicDType to all four cache tensors, so a uint8 source against a
        # bf16 destination is rejected outright.
        k_records = self._h2d.layer_k(layer_id, num_tokens)
        v_records = self._h2d.layer_v(layer_id, num_tokens)
        jit_transfer_hicache_one_layer(
            page_size=1,
            k_cache_dst=k_records,
            v_cache_dst=v_records,
            k_cache_src=self.k_data_refs[host_layer_id],
            v_cache_src=self.v_data_refs[host_layer_id],
            indices_dst=self._h2d_index_view(num_tokens),
            indices_src=host_indices,
            element_dim=codec.ROW_BYTES,
        )

        # Decode and scatter into the device pool. Enqueued here, before return.
        scatter = device_indices.to(torch.int64)
        dtype = device_pool.store_dtype
        k_bf16 = codec.decode_records(
            k_records, head_num=self.head_num, head_dim=self.head_dim, dtype=dtype
        )
        device_pool.k_buffer[layer_id][scatter] = k_bf16.reshape(
            num_tokens, self.head_num, self.head_dim
        )
        v_bf16 = codec.decode_records(
            v_records, head_num=self.head_num, head_dim=self.head_dim, dtype=dtype
        )
        device_pool.v_buffer[layer_id][scatter] = v_bf16.reshape(
            num_tokens, self.head_num, self.head_dim
        )

    # ------------------------------------------------------------------
    # Storage (L3) page interface
    # ------------------------------------------------------------------

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "INT8 HiCache host pool does not implement flat storage (L3) pages. "
            "An encoded arena is not a BF16 kv_buffer, so zero-copy backends "
            "cannot derive per-page pointers from it. Run without "
            "--hicache-storage-backend."
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = self.kv_buffer[:, :, index : index + self.page_size, :]
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, codec.ROW_BYTES),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.kv_buffer[:, :, index : index + self.page_size, :] = data_page.reshape(
            2, self.layer_num, self.page_size, codec.ROW_BYTES
        )

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(self, indices, split_factor: int):
        raise self._storage_pages_unsupported()

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """O_DIRECT alignment for the encoded stride.

        One page is ``page_size * layer_num * ROW_BYTES`` bytes of K followed by
        the same of V, so the per-page stride is
        ``page_size * layer_num * 1152``.
        """
        stride = self.page_size * self.layer_num * codec.ROW_BYTES
        base_aligned = (
            self.k_buffer.data_ptr() % page_size_bytes == 0
            and self.v_buffer.data_ptr() % page_size_bytes == 0
        )
        return base_aligned and stride % page_size_bytes == 0
