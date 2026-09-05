"""Tiered KV pool: int8-g64 ring over the int4-g32 full-context pool.

Layout per local layer l: the int4 pool's k/v_buffer + k/v_scale_buffer (rows = size + page_size, on the lazy
VMM owner, untouched) PLUS a fixed ring of R = SGLANG_KV_TIERS_W slots outside the owner:
  ring_k[l], ring_v[l]     int8 [R, H, D]                  ring row r = slot & (R - 1)
  ring_ks[l], ring_vs[l]   fp16 [R, H, D // 64]            absmax/127 per (row, head, 64-channel group)
  ring_owner               int32 [R]                       slot that owns ring row r (-1 = nobody)
set_kv_buffer stamps the owner (one launch) and then writes both tiers (dual-write launch; the ring row
only by the stamped owner, so same-write ring-row collisions degrade to cold instead of mixing bytes);
readers test owner[slot & mask] == slot on the device and take the int8 ring row (hot) or the int4 row (cold).
The ring is allocated once at construction (before graph capture; its pointers are baked into the decode
graph) and never reallocated; it is NOT a KvBufferDesc, so bytes_per_token()/lazy_ensure/kv_lazy.py stay
exact and PD registration (desc-aligned) is unchanged.  `kv_tiered = True` is the backend's dispatch key
(kv_bits stays 4: scratch shape, prefix pack shape and the int4 cell size are inherited).
"""

import logging
import os
from typing import Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.sparse_attn import (
    KV_INT8_GROUP,
    quant_store_kv_tiered,
)
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4, _is_unit_scale
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
from sglang.srt.utils.async_probe import maybe_detect_oob

logger = logging.getLogger(__name__)


def ring_slots_from_env() -> int:
    """SGLANG_KV_TIERS_W: ring slots R (default 8192); must be a positive power of two (ring index = slot & (R - 1))."""
    raw = os.environ.get("SGLANG_KV_TIERS_W", "8192")
    try:
        r = int(raw)
    except ValueError:
        raise ValueError(f"SGLANG_KV_TIERS_W={raw!r} is not an integer") from None
    if r <= 0 or (r & (r - 1)):
        raise ValueError(f"SGLANG_KV_TIERS_W must be a positive power of two, got {r}")
    return r


class MHATokenToKVPoolTiered(MHATokenToKVPoolInt4):
    """int4-g32 full-context rows (inherited) + an int8-g64 ring of the last R written slots + owner table."""

    RING_GROUP = KV_INT8_GROUP
    kv_tiered = True  # QSA backend dispatch key (kv_bits == 4 is inherited)

    def __init__(self, *args, **kwargs):
        self.ring_k = None
        self.ring_v = None
        self.ring_ks = None
        self.ring_vs = None
        self.ring_owner = None
        super().__init__(*args, **kwargs)
        R = ring_slots_from_env()
        if R > self.size:
            raise ValueError(
                f"SGLANG_KV_TIERS_W={R} exceeds the KV pool size {self.size}"
            )
        if self.head_dim % self.RING_GROUP or self.v_head_dim % self.RING_GROUP:
            raise ValueError(
                f"head dims {self.head_dim}/{self.v_head_dim} not multiples of RING_GROUP {self.RING_GROUP}"
            )
        self.ring_slots = R
        L, H = self.layer_num, self.head_num
        ks_shape = (R, H, self.head_dim // self.RING_GROUP)
        vs_shape = (R, H, self.v_head_dim // self.RING_GROUP)
        # Plain torch tensors (torch allocator, no 2 MiB granule rounding), allocated once before capture.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.ring_k = [
                torch.zeros((R, H, self.head_dim), dtype=torch.int8, device=self.device)
                for _ in range(L)
            ]
            self.ring_v = [
                torch.zeros(
                    (R, H, self.v_head_dim), dtype=torch.int8, device=self.device
                )
                for _ in range(L)
            ]
            self.ring_ks = [
                torch.zeros(ks_shape, dtype=torch.float16, device=self.device)
                for _ in range(L)
            ]
            self.ring_vs = [
                torch.zeros(vs_shape, dtype=torch.float16, device=self.device)
                for _ in range(L)
            ]
            self.ring_owner = torch.full(
                (R,), -1, dtype=torch.int32, device=self.device
            )
        ring_bytes = sum(
            t.numel() * t.element_size()
            for t in self.ring_k + self.ring_v + self.ring_ks + self.ring_vs
        )
        logger.info(
            "KV tiers: ring R=%d slots (int8_g%d over int4_g%d), %.0f MB, owner %d KB",
            R,
            self.RING_GROUP,
            self.GROUP,
            ring_bytes / 2**20,
            self.ring_owner.numel() * 4 // 1024,
        )

    # -- layout ---------------------------------------------------------------------------------

    @property
    def ring_mask(self) -> int:
        return self.ring_slots - 1

    def _ring_bytes(self) -> Tuple[int, int]:
        kb = sum(t.numel() * t.element_size() for t in self.ring_k + self.ring_ks)
        vb = sum(t.numel() * t.element_size() for t in self.ring_v + self.ring_vs)
        return kb + self.ring_owner.numel() * self.ring_owner.element_size(), vb

    def get_kv_size_bytes(self):
        k_size_bytes, v_size_bytes = super().get_kv_size_bytes()
        if self.ring_k is not None:
            rk, rv = self._ring_bytes()
            k_size_bytes += rk
            v_size_bytes += rv
        return k_size_bytes, v_size_bytes

    # _pd_registerable_tensors is inherited on purpose: get_contiguous_buf_infos zips it with
    # _kv_buffer_descs, and the ring has no desc (it is not on the owner).  PD transfer of a tiered pool
    # would move the int4 tier only; PD is unsupported here anyway (CPU offload raises in the int4 pool).

    def _clear_buffers(self):
        super()._clear_buffers()
        self.ring_k = self.ring_v = self.ring_ks = self.ring_vs = self.ring_owner = None

    # -- accessors ------------------------------------------------------------------------------

    def get_kv_ring_buffer(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.ring_k[idx], self.ring_v[idx], self.ring_ks[idx], self.ring_vs[idx]

    def get_kv_ring_owner(self) -> torch.Tensor:
        return self.ring_owner

    # -- write path -----------------------------------------------------------------------------

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError(
                "int8ring_int4 KV cache does not support DCP KV masks"
            )
        if not (_is_unit_scale(k_scale) and _is_unit_scale(v_scale)):
            raise ValueError(
                "int8ring_int4 KV cache computes its own scales; got k_scale/v_scale"
            )
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-TIERED)"
        )
        if loc.numel() > self.ring_slots:
            # more tokens than ring rows would make every write collide (chunked prefill 1024 <= R); a
            # collision is safe (the stamp launch picks one owner per ring row, the others are cold) but
            # this can only be a misconfiguration
            raise ValueError(
                f"int8ring_int4: {loc.numel()} tokens in one write exceed the ring of {self.ring_slots} slots"
            )
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        idx = layer_id - self.start_layer
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.v_head_dim)
        quant_store_kv_tiered(
            cache_k,
            cache_v,
            loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scale_buffer[idx],
            self.v_scale_buffer[idx],
            self.ring_k[idx],
            self.ring_v[idx],
            self.ring_ks[idx],
            self.ring_vs[idx],
            self.ring_owner,
            self.sm_k_inv[idx],
            self.sm_v_inv[idx],
            self.ring_mask,
        )

    # -- lazy VMM hooks -------------------------------------------------------------------------

    def lazy_release(self) -> None:
        super().lazy_release()
        # Hygiene only: every slot is re-stamped by its own write before any read, so correctness never
        # depends on this; it just keeps a stale owner from pointing a reader at an old ring row.
        if self.ring_owner is not None:
            self.ring_owner.fill_(-1)
