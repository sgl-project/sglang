"""
Unified HP + int2 KV cache pool.

Quant arena: paged with ``N_Q`` slots per page. HP arena: shared HP-prefix
pool (paged) followed by per-request HP-recent ring slabs. Slot id namespace
is flat (``[0, num_quant_pages*N_Q)`` quant, ``[HP_OFFSET, ...)`` HP), and
kernels dispatch by ``slot >= HP_OFFSET``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.fused_hadamard_int2_kv import (
    quantized_set_kv_int2_pretransformed_triton,
)
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_oscar_rotate_k_clip_triton,
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_quant_kernels import _get_num_scale_groups
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    OscarRotationConfig,
    _set_kv_buffer_impl,
    get_tensor_size_bytes,
    load_oscar_rotation_config,
    load_oscar_rotations,
    unwrap_write_loc,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@triton.jit
def _set_mixed_hp_buffer_kernel(
    src_ptr,
    dst_ptr,
    loc_ptr,
    num_tokens,
    row_dim: tl.constexpr,
    src_stride_token: tl.constexpr,
    src_stride_dim: tl.constexpr,
    dst_stride_loc: tl.constexpr,
    dst_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    loc = tl.load(loc_ptr + token_idx)
    is_hp = loc >= HP_OFFSET
    hp_loc = loc - HP_OFFSET
    mask = is_hp & (token_idx < num_tokens) & (offs < row_dim)
    vals = tl.load(
        src_ptr + token_idx * src_stride_token + offs * src_stride_dim,
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + hp_loc * dst_stride_loc + offs * dst_stride_dim,
        vals,
        mask=mask,
    )


def _resolve_torch_dtype(name: str, *, kind: str) -> torch.dtype:
    """Map a friendly dtype name (``bf16``/``bfloat16``/``fp16``/``half``/``fp32``)
    to the corresponding ``torch.dtype``. ``kind`` is only used in the error
    message ("scale" / "HP" / etc.) so the caller's intent surfaces in the
    failure mode.
    """
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported {kind} dtype: {name}. Expected bf16/fp16/fp32.")


def resolve_scale_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="scale")


def resolve_hp_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="HP")


def compute_page_geometry(hp_dtype: torch.dtype) -> Tuple[int, int]:
    """Return ``(N_H, N_Q)`` for int2 + ``hp_dtype``.

    ``N_Q`` is the int2 page size used by the paged quant allocator (and
    ``--page-size``). ``N_H`` is retained as ``1`` for documentation and for
    legacy callers, but no longer carries the LCM byte-equivalence invariant —
    HP and quant arenas are decoupled allocations under the slab design.
    """
    hp_itemsize = torch.empty(0, dtype=hp_dtype).element_size()
    return 1, 4 * hp_itemsize


def compute_recent_ring_size(hp_recent_tokens: int, n_q: int) -> int:
    # Max HP-recent occupancy between flushes is hp_recent + (N_Q - 1); the
    # ring reuses slots after the oldest N_Q have been demoted to quant.
    return int(hp_recent_tokens) + int(n_q) - 1


class UnifiedInt2HPKVPool(KVCache):
    """Unified HP + int2 MHA KV cache.

    The pool exposes:
      * ``k_buffer[l]``, ``v_buffer[l]``           – quant (int2 packed uint8) views
      * ``hp_k_buffer[l]``, ``hp_v_buffer[l]``     – HP (``hp_dtype``) views
      * ``k_scales_zeros[l]``, ``v_scales_zeros[l]`` – per-group scales+zeros in
        ``scale_dtype`` (bf16/fp16/fp32)

    The quant and HP views alias the same byte arena. Callers must treat a
    physical page as homogeneous (either tier) at any given time; this invariant
    is enforced by the ``UnifiedInt2HPKVAllocator`` that hands out slot ids into
    these views.
    """

    def __init__(
        self,
        num_quant_pages: int,
        hp_dtype: torch.dtype,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        dtype: str,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_req_slots: int,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        model_dtype: Optional[torch.dtype] = None,
        kv_cache_quant_group_size: Optional[int] = None,
        scale_dtype: torch.dtype = torch.bfloat16,
        num_hp_prefix_slots: int = 0,
    ):
        assert dtype == "int2", (
            "UnifiedInt2HPKVPool supports only int2 quant tier; got %s" % dtype
        )
        # Work around KVCache.__init__ dtype validation: it stores ``dtype`` as
        # a string and sets ``store_dtype=torch.uint8`` for int2.
        super().__init__(
            size=num_quant_pages,  # used by base class for sizing heuristics only
            page_size=1,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            model_dtype=model_dtype,
        )

        self.num_quant_pages = int(num_quant_pages)
        self.hp_dtype = hp_dtype
        self.scale_dtype = scale_dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.kv_cache_quant_group_size = kv_cache_quant_group_size

        self.N_H, self.N_Q = compute_page_geometry(hp_dtype)
        self.hp_recent_ring_size = compute_recent_ring_size(
            self.hp_recent_tokens, self.N_Q
        )
        if num_hp_prefix_slots > 0 and num_hp_prefix_slots % self.N_Q != 0:
            num_hp_prefix_slots = (
                (num_hp_prefix_slots + self.N_Q - 1) // self.N_Q * self.N_Q
            )
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        self.slab_size = self.hp_recent_ring_size  # back-compat alias
        self._hp_offset = self.num_quant_pages * self.N_Q
        self._hp_recent_base = self.num_hp_prefix_slots

        # Window sizes must be N_Q-aligned so radix tree (page_size=N_Q) and
        # the flush kernel land on page boundaries.
        if self.hp_prefix_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_PREFIX_TOKENS ({self.hp_prefix_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        if self.hp_recent_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_RECENT_TOKENS ({self.hp_recent_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        # Flush every N_Q decode steps demotes exactly N_Q HP-recent slots
        # into one quant page. Per-request counter (initialized at admission
        # to (hp_recent+N_Q-1)-H_0) keeps every flush whole-page.
        self.flush_interval = self.N_Q
        self.max_req_slots = int(max_req_slots)
        self._flush_counter = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )
        self._next_slab_offset = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )

        # Forward-done event stashed each iteration; consumed by
        # ``wait_pending_forward`` inside the flush apply phase.
        self._pending_forward_done = None

        # Grouping for quantization.
        self.k_quant_group_size, self.k_num_scale_groups = self._resolve_quant_grouping(
            self.head_dim, "K"
        )
        self.v_quant_group_size, self.v_num_scale_groups = self._resolve_quant_grouping(
            self.v_head_dim, "V"
        )
        assert (
            self.head_dim % 4 == 0
        ), f"head_dim={self.head_dim} must be divisible by 4 for int2 packing"
        assert (
            self.v_head_dim % 4 == 0
        ), f"v_head_dim={self.v_head_dim} must be divisible by 4 for int2 packing"

        self._create_arenas()

        # Cached attributes used by the rest of the stack.
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self.row_dim = self.head_num * self.head_dim  # for store_cache helpers
        self.same_kv_dim = self.head_dim == self.v_head_dim

        # Oscar rotation + clip. Per-layer orthogonal matrices [head_dim,
        # head_dim] / [v_head_dim, v_head_dim] are loaded in ``hp_dtype`` so
        # the ``rows @ R`` pre-pass and ``result @ R.T`` inverse are plain
        # bf16 GEMMs.
        self._oscar_cfg: OscarRotationConfig = load_oscar_rotation_config()
        self._k_clip_ratio: float = self._oscar_cfg.k_clip_ratio
        self._v_clip_ratio: float = self._oscar_cfg.v_clip_ratio
        self._lloyd_max: bool = envs.SGLANG_LLOYD_MAX.get()
        self._R_k: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        self._R_v: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.v_head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        logger.info(
            "UnifiedInt2HPKVPool: Oscar rotation enabled (k_clip=%.4f v_clip=%.4f lloyd_max=%s)",
            self._k_clip_ratio,
            self._v_clip_ratio,
            self._lloyd_max,
        )

        hp_total_slots = (
            self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size
        )
        self._finalize_allocation_log(hp_total_slots)
        hp_itemsize = torch.empty(0, dtype=self.hp_dtype).element_size()
        hp_bytes = (
            hp_total_slots
            * self.layer_num
            * self.head_num
            * (self.head_dim + self.v_head_dim)
            * hp_itemsize
        )
        logger.info(
            "UnifiedInt2HPKVPool: HP arena reserves %.2f GB "
            "(hp_prefix_pool_slots=%d, max_req_slots=%d, recent_ring=%d "
            "= R=%d + N_Q-1=%d, P=%d, layers=%d, head_num=%d, "
            "head_dim+v_head_dim=%d, hp_dtype=%s)",
            hp_bytes / GB,
            self.num_hp_prefix_slots,
            self.max_req_slots,
            self.hp_recent_ring_size,
            self.hp_recent_tokens,
            self.N_Q - 1,
            self.hp_prefix_tokens,
            self.layer_num,
            self.head_num,
            self.head_dim + self.v_head_dim,
            str(self.hp_dtype),
        )

    # -- Configuration accessors -------------------------------------------

    def mixed_kv_enabled(self) -> bool:
        return True

    def stash_pending_forward(self, event) -> None:
        """Record the most recent forward-stream completion event.

        Called once per iteration from the scheduler. The event is consumed
        by :meth:`wait_pending_forward` at the apply boundary inside
        ``_alloc_for_decode_mixed``.
        """
        self._pending_forward_done = event

    def wait_pending_forward(self) -> None:
        """Order the current stream after the stashed forward-done event.

        Must be issued *before* the apply phase of the flush
        (``gpu_flush_int2_apply``), since the remap kernel writes
        ``req_to_token`` at positions the previous forward's attention is
        concurrently reading. Pre-apply work (allocator free, plan kernel)
        runs ahead of this wait so its host syncs don't block on the
        previous forward.
        """
        if self._pending_forward_done is None:
            return
        torch.cuda.current_stream().wait_event(self._pending_forward_done)
        self._pending_forward_done = None

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    @property
    def hp_size(self) -> int:
        return self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size

    @property
    def quant_size(self) -> int:
        return self.num_quant_pages * self.N_Q

    @property
    def hp_prefix_pool_slots(self) -> int:
        return self.num_hp_prefix_slots

    @property
    def hp_recent_base(self) -> int:
        """First HP-buffer index reserved for per-req recent slabs."""
        return self._hp_recent_base

    def release_req_slab(self, req_pool_idx) -> None:
        # Reset the per-req HP-recent cursor and flush counter so the next
        # request taking over ``req_pool_idx`` starts clean.
        if isinstance(req_pool_idx, torch.Tensor):
            idx = req_pool_idx.to(self._next_slab_offset.device).to(torch.int64)
            if idx.numel() == 0:
                return
            self._next_slab_offset[idx] = 0
            self._flush_counter[idx] = 0
        else:
            i = int(req_pool_idx)
            self._next_slab_offset[i] = 0
            self._flush_counter[i] = 0

    def _resolve_quant_grouping(
        self, head_dim: int, tensor_name: str
    ) -> tuple[int, int]:
        group_size = (
            head_dim
            if self.kv_cache_quant_group_size is None
            else self.kv_cache_quant_group_size
        )
        if group_size <= 0:
            raise ValueError(
                f"{tensor_name} kv_cache_quant_group_size must be positive, got {group_size}"
            )
        if head_dim % group_size != 0:
            raise ValueError(
                f"{tensor_name} head_dim ({head_dim}) must be divisible by "
                f"kv_cache_quant_group_size ({group_size})"
            )
        return group_size, head_dim // group_size

    # -- Arena construction ------------------------------------------------

    def _create_arenas(self):
        # HP arena layout: [shared prefix pool] [per-req recent slab 0]
        # [per-req recent slab 1] ... Quant arena is paged with N_Q slots
        # per page; scales/zeros are quant-only.
        hp_total_slots = (
            self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            self.head_dim // 4,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            self.v_head_dim // 4,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.k_scales_zeros = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            2 * self.k_num_scale_groups,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scales_zeros = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            2 * self.v_num_scale_groups,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_k_buffer = [
                    torch.zeros(
                        (hp_total_slots, self.head_num, self.head_dim),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_v_buffer = [
                    torch.zeros(
                        (hp_total_slots, self.head_num, self.v_head_dim),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        # Cached device pointer arrays for the fused decode-flush kernel. The
        # flush kernel loops over layers inside the kernel, so it needs
        # per-layer base pointers as an int64 GPU tensor. Strides are identical
        # across layers (we enforce that below); the flush kernel reads the
        # single set at launch time via tl.constexpr.
        def _base_ptrs(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors],
                dtype=torch.int64,
                device=self.device,
            )

        self._flush_hp_k_ptrs = _base_ptrs(self.hp_k_buffer)
        self._flush_hp_v_ptrs = _base_ptrs(self.hp_v_buffer)
        self._flush_quant_k_ptrs = _base_ptrs(self.k_buffer)
        self._flush_quant_v_ptrs = _base_ptrs(self.v_buffer)
        self._flush_k_sz_ptrs = _base_ptrs(self.k_scales_zeros)
        self._flush_v_sz_ptrs = _base_ptrs(self.v_scales_zeros)

        # Strides (elements, not bytes) for each kind of buffer. The arenas are
        # all contiguous, so every layer shares the same strides; assert to be
        # safe.
        def _strides(t: torch.Tensor) -> tuple:
            return (int(t.stride(0)), int(t.stride(1)), int(t.stride(2)))

        hp_k_stride = _strides(self.hp_k_buffer[0])
        hp_v_stride = _strides(self.hp_v_buffer[0])
        q_k_stride = _strides(self.k_buffer[0])
        q_v_stride = _strides(self.v_buffer[0])
        k_sz_stride = _strides(self.k_scales_zeros[0])
        v_sz_stride = _strides(self.v_scales_zeros[0])
        for l in range(self.layer_num):
            assert _strides(self.hp_k_buffer[l]) == hp_k_stride
            assert _strides(self.hp_v_buffer[l]) == hp_v_stride
            assert _strides(self.k_buffer[l]) == q_k_stride
            assert _strides(self.v_buffer[l]) == q_v_stride
            assert _strides(self.k_scales_zeros[l]) == k_sz_stride
            assert _strides(self.v_scales_zeros[l]) == v_sz_stride

        self._flush_hp_k_stride = hp_k_stride
        self._flush_hp_v_stride = hp_v_stride
        self._flush_quant_k_stride = q_k_stride
        self._flush_quant_v_stride = q_v_stride
        self._flush_k_sz_stride = k_sz_stride
        self._flush_v_sz_stride = v_sz_stride

    # -- KVCache interface -------------------------------------------------

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer)
        k += sum(get_tensor_size_bytes(s) for s in self.k_scales_zeros)
        k += sum(get_tensor_size_bytes(t) for t in self.hp_k_buffer)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer)
        v += sum(get_tensor_size_bytes(s) for s in self.v_scales_zeros)
        v += sum(get_tensor_size_bytes(t) for t in self.hp_v_buffer)
        return k, v

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # Triton backend asks for the quant view in the mixed path; HP view is
        # accessed via ``get_hp_key_buffer``.
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_raw_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._layer_index(layer_id)]

    def get_raw_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_key_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.k_scales_zeros[self._layer_index(layer_id)]

    def get_value_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.v_scales_zeros[self._layer_index(layer_id)]

    def get_hp_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_k_buffer[self._layer_index(layer_id)]

    def get_hp_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_v_buffer[self._layer_index(layer_id)]

    def get_raw_kv_buffer(self, layer_id: int):
        idx = self._layer_index(layer_id)
        return {
            "k_buffer": self.k_buffer[idx],
            "v_buffer": self.v_buffer[idx],
            "k_scales_zeros": self.k_scales_zeros[idx],
            "v_scales_zeros": self.v_scales_zeros[idx],
            "dtype": "int2",
        }

    def _split_global_locs(self, loc: torch.Tensor):
        loc64 = loc.to(torch.int64)
        hp_mask = loc64 >= self._hp_offset
        quant_loc = loc64[~hp_mask]
        hp_loc_global = loc64[hp_mask] - self._hp_offset
        return quant_loc, hp_loc_global, hp_mask

    def _rotate_kv_inplace(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the per-layer Oscar rotation ``rows @ R`` to HP K/V tiles.

        Returns tensors in ``self.hp_dtype`` ready to be stored or packed.
        ``R_k`` / ``R_v`` are ``[head_dim, head_dim]`` bf16 on the KV device,
        loaded in ``__init__``.
        """
        idx = self._layer_index(layer_id)
        k_hp = cache_k.to(self.hp_dtype) @ self._R_k[idx]
        if v_rotation_absorbed:
            v_hp = cache_v.to(self.hp_dtype)
        else:
            v_hp = cache_v.to(self.hp_dtype) @ self._R_v[idx]
        return k_hp, v_hp

    def _prepare_hp_kv_tensors(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_rotated: bool,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Oscar rotation to HP K/V and cast to ``hp_dtype``.
        ``already_rotated`` skips the rotation pre-pass.
        """
        if already_rotated:
            return cache_k.to(self.hp_dtype), cache_v.to(self.hp_dtype)
        return self._rotate_kv_inplace(layer_id, cache_k, cache_v, v_rotation_absorbed)

    def _set_hp_kv_buffer(
        self,
        layer_id: int,
        hp_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.hp_k_buffer[idx],
            self.hp_v_buffer[idx],
            hp_loc,
            row_dim=self.row_dim,
            store_dtype=self.hp_dtype,
            device_module=self.device_module,
            size_limit=self.hp_k_buffer[idx].shape[0],
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def _set_quant_kv_buffer_extend(
        self,
        layer_id: int,
        quant_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_hadamard_transformed: bool,
        mixed_hp_offset: Optional[int] = None,
        v_rotation_absorbed: bool = False,
    ):
        """Prefill/extend-only: rotate (oscar R) + optional per-row clip +
        int2-pack + write quant slots.

        Decode-time flushes go through the dedicated GPU flush kernel
        (see ``gpu_flush_int2``); this method is *not* used for those.
        """
        idx = self._layer_index(layer_id)
        clip_on = self._k_clip_ratio > 0.0 or self._v_clip_ratio > 0.0

        # Fused rotate(K) + clip(KV) + quantize(KV) + set(KV). Skips the
        # standalone ``K @ R_k`` GEMM and its bf16 staging tensor by doing
        # the rotation inside the int2 pack kernel via ``tl.dot``. V must
        # already be in R_v space (rotation absorbed) — the kernel does
        # not rotate V. Requires single-scale layout (num_groups == 1) for
        # both K and V scales/zeros.
        if envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get():
            assert (
                v_rotation_absorbed
            ), "V rotation must be absorbed for fused oscar K-rotation + clip + quant + set"

        use_fused_rotate = (
            envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get()
            and not already_hadamard_transformed
            and v_rotation_absorbed
            and clip_on
            and _get_num_scale_groups(self.k_scales_zeros[idx]) == 1
            and _get_num_scale_groups(self.v_scales_zeros[idx]) == 1
        )
        if use_fused_rotate:
            quantized_set_kv_int2_oscar_rotate_k_clip_triton(
                cache_k.to(self.hp_dtype),
                cache_v.to(self.hp_dtype),
                self._R_k[idx],
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                self._k_clip_ratio,
                self._v_clip_ratio,
                hp_global_offset=mixed_hp_offset,
            )
            return

        if not already_hadamard_transformed:
            cache_k, cache_v = self._rotate_kv_inplace(
                layer_id, cache_k, cache_v, v_rotation_absorbed
            )
        else:
            cache_k = cache_k.to(self.hp_dtype)
            cache_v = cache_v.to(self.hp_dtype)

        if not clip_on:
            quantized_set_kv_int2_pretransformed_triton(
                cache_k,
                cache_v,
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                hp_global_offset=mixed_hp_offset,
            )
            return

        quantized_set_kv_int2_pretransformed_clip_triton(
            cache_k,
            cache_v,
            quant_loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scales_zeros[idx],
            self.v_scales_zeros[idx],
            self._k_clip_ratio,
            self._v_clip_ratio,
            hp_global_offset=mixed_hp_offset,
            lloyd_max=self._lloyd_max,
        )

    def _set_mixed_hp_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)

        def _launch(src: torch.Tensor, dst: torch.Tensor):
            src2 = src.reshape(src.shape[0], -1)
            dst2 = dst.reshape(dst.shape[0], -1)
            row_dim = src2.shape[1]
            if row_dim == 0 or src2.shape[0] == 0:
                return
            block_row = min(1024, triton.next_power_of_2(row_dim))
            grid = (src2.shape[0], triton.cdiv(row_dim, block_row))
            _set_mixed_hp_buffer_kernel[grid](
                src2,
                dst2,
                loc,
                src2.shape[0],
                row_dim,
                src2.stride(0),
                src2.stride(1),
                dst2.stride(0),
                dst2.stride(1),
                HP_OFFSET=int(self._hp_offset),
                BLOCK_ROW=block_row,
                num_warps=4,
                num_stages=1,
            )

        _launch(cache_k, self.hp_k_buffer[idx])
        _launch(cache_v, self.hp_v_buffer[idx])

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
    ):
        """Write K/V to the unified pool.

        ``is_decode`` selects the path:
            * ``True`` -- single-token decode write. Caller fills ``loc`` with
              valid HP slot ids (from ``allocator.alloc_hp_recent``); we
              write only the HP buffer, with no boolean masking (safe under
              CUDA-graph capture).
            * ``False`` (extend / prefill) -- mixed write: int2 quant slots get
              the rotated+clipped pack, HP slots get the bf16 row.

        Callers must pass ``is_decode`` based on
        ``forward_batch.forward_mode.is_decode_or_idle()``. Capture state is
        *not* a reliable proxy: piecewise CUDA graph captures parts of prefill
        (would mis-route to the HP-only branch) and ``--disable-cuda-graph``
        runs decode eagerly (would mis-route to the quant+HP branch).
        """
        loc, _, _ = unwrap_write_loc(loc)
        if loc.numel() == 0:
            return

        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))

        if is_decode:
            hp_local = loc.to(torch.int64) - self._hp_offset
            cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
                layer_id,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            )
            self._set_hp_kv_buffer(layer_id, hp_local, cache_k_hp, cache_v_hp)
            return

        self._set_quant_kv_buffer_extend(
            layer_id,
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            mixed_hp_offset=int(self._hp_offset),
            v_rotation_absorbed=v_rotation_absorbed,
        )
        cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
            layer_id,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            v_rotation_absorbed,
        )
        self._set_mixed_hp_kv_buffer(layer_id, loc, cache_k_hp, cache_v_hp)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        # Moves only make sense within the same tier. The allocator ensures
        # that; callers should split by tier before calling here.
        tgt_q, tgt_hp, tgt_mask = self._split_global_locs(tgt_loc)
        src_q, src_hp, src_mask = self._split_global_locs(src_loc)
        assert torch.equal(
            tgt_mask, src_mask
        ), "move_kv_cache requires src/tgt tiers to match"
        for l in range(self.layer_num):
            if tgt_q.numel() > 0:
                self.k_buffer[l][tgt_q] = self.k_buffer[l][src_q]
                self.v_buffer[l][tgt_q] = self.v_buffer[l][src_q]
                self.k_scales_zeros[l][tgt_q] = self.k_scales_zeros[l][src_q]
                self.v_scales_zeros[l][tgt_q] = self.v_scales_zeros[l][src_q]
            if tgt_hp.numel() > 0:
                self.hp_k_buffer[l][tgt_hp] = self.hp_k_buffer[l][src_hp]
                self.hp_v_buffer[l][tgt_hp] = self.hp_v_buffer[l][src_hp]

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")
