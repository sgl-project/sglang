"""INT4-G32 KV pool for the Qwen sparse attention layers.

Layout per local layer l (rows = size + page_size, NHD):
  k_buffer[l], v_buffer[l]             uint8 [rows, H, D // 2]     two 4-bit channels per byte
                                       (low nibble = even channel, high = odd, stored q + 8, q in [-7, 7])
  k_scale_buffer[l], v_scale_buffer[l] fp16 [rows, H, D // GROUP]  absmax/7 per (token, head, group)
The scale buffers are extra KvBufferDescs (int8 [rows, H * D // GROUP * 2]) on the lazy VMM owner and
viewed as fp16; on the eager path they are plain zero tensors.  `kv_bits = 4` is the QSA backend's
dispatch key (the int8 pool carries 8; the fp8 pool also stores uint8, so dtype alone is ambiguous).
Smoothing constants (sm_k / sm_v and their inverses, fp16 [L, H, D]) are identity, as in the int8 pool.
"""

from typing import Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.sparse_attn import (
    KV_INT4_GROUP,
    quant_store_kv_int4,
)
from sglang.srt.mem_cache.memory_pool import (
    KvBufferDesc,
    MHATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.utils.async_probe import maybe_detect_oob


def _is_unit_scale(s) -> bool:
    return s is None or (isinstance(s, (int, float)) and s == 1)


class MHATokenToKVPoolInt4(MHATokenToKVPool):
    """MHA KV pool storing nibble-packed int4 K/V with fp16 per-(token, head, GROUP-channel) scales."""

    GROUP = KV_INT4_GROUP
    kv_bits = 4  # QSA backend dispatch key (the int8_g64 pool carries 8)

    def __init__(self, *args, **kwargs):
        self.k_scale_buffer = None
        self.v_scale_buffer = None
        super().__init__(*args, **kwargs)
        if self.dtype != torch.uint8 or self.store_dtype != torch.uint8:
            raise ValueError(
                f"MHATokenToKVPoolInt4 needs dtype uint8, got {self.dtype}/{self.store_dtype}"
            )
        if self.use_hnd or self.kv_cache_layout != "nhd":
            raise ValueError("MHATokenToKVPoolInt4 supports the NHD layout only")
        # Static per-channel smoothing constants; identity.
        L, H = self.layer_num, self.head_num
        self.sm_k = torch.ones(
            (L, H, self.head_dim), dtype=torch.float16, device=self.device
        )
        self.sm_v = torch.ones(
            (L, H, self.v_head_dim), dtype=torch.float16, device=self.device
        )
        self.sm_k_inv = torch.ones_like(self.sm_k)
        self.sm_v_inv = torch.ones_like(self.sm_v)

    # -- layout ---------------------------------------------------------------------------------

    @property
    def k_groups(self) -> int:
        if self.head_dim % self.GROUP:
            raise ValueError(
                f"head_dim {self.head_dim} not a multiple of GROUP {self.GROUP}"
            )
        return self.head_dim // self.GROUP

    @property
    def v_groups(self) -> int:
        if self.v_head_dim % self.GROUP:
            raise ValueError(
                f"v_head_dim {self.v_head_dim} not a multiple of GROUP {self.GROUP}"
            )
        return self.v_head_dim // self.GROUP

    def _kv_buffer_shapes(self):
        """Payload rows hold D // 2 bytes per head (two channels per byte)."""
        if self.use_hnd:
            raise ValueError("MHATokenToKVPoolInt4 supports the NHD layout only")
        if self.head_dim % 2 or self.v_head_dim % 2:
            raise ValueError("MHATokenToKVPoolInt4 needs even head dims")
        rows = self.size + self.page_size
        return (
            (rows, self.head_num, self.head_dim // 2),
            (rows, self.head_num, self.v_head_dim // 2),
        )

    def _scale_shapes(self):
        rows = self.size + self.page_size
        return (rows, self.head_num, self.k_groups), (
            rows,
            self.head_num,
            self.v_groups,
        )

    def _create_buffers_normal(self):
        super()._create_buffers_normal()
        ks_shape, vs_shape = self._scale_shapes()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_scale_buffer = [
                torch.zeros(ks_shape, dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.v_scale_buffer = [
                torch.zeros(vs_shape, dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]

    def _build_kv_buffer_descs(self):
        descs = super()._build_kv_buffer_descs()
        if any(d.tokens_per_row != 1 for d in descs):
            raise ValueError(
                "MHATokenToKVPoolInt4 scale descs assume one token per row (NHD)"
            )
        ks_shape, vs_shape = self._scale_shapes()
        for prefix, shape in (("ks", ks_shape), ("vs", vs_shape)):
            row_bytes = shape[1] * shape[2] * 2  # fp16 scales as int8 bytes
            for layer in range(self.layer_num):
                descs.append(
                    KvBufferDesc(
                        f"{prefix}{layer}",
                        (shape[0], row_bytes),
                        row_bytes=row_bytes,
                        tokens_per_row=1,
                    )
                )
        return descs

    def _assign_post_capture_tensors(self, tensors):
        L = self.layer_num
        ks_shape, vs_shape = self._scale_shapes()
        self.k_buffer = tensors[:L]
        self.v_buffer = tensors[L : 2 * L]
        self.k_scale_buffer = [
            t.view(torch.float16).view(ks_shape) for t in tensors[2 * L : 3 * L]
        ]
        self.v_scale_buffer = [
            t.view(torch.float16).view(vs_shape) for t in tensors[3 * L : 4 * L]
        ]
        # The owner backs exactly one page at construction; zero only those rows (never touch
        # unbacked VA).  Every other slot is written before it is read.
        for t in self.k_scale_buffer + self.v_scale_buffer:
            t[: self.page_size].zero_()

    def _pd_registerable_tensors(self):
        return self.k_buffer + self.v_buffer + self.k_scale_buffer + self.v_scale_buffer

    # -- accessors ------------------------------------------------------------------------------

    def get_kv_scale_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.k_scale_buffer[idx], self.v_scale_buffer[idx]

    def get_kv_smooth_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.sm_k[idx], self.sm_v[idx]

    # -- write path -----------------------------------------------------------------------------

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError("int4_g32 KV cache does not support DCP KV masks")
        if not (_is_unit_scale(k_scale) and _is_unit_scale(v_scale)):
            raise ValueError(
                "int4_g32 KV cache computes its own scales; got k_scale/v_scale"
            )
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-INT4)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        idx = layer_id - self.start_layer
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.v_head_dim)
        quant_store_kv_int4(
            cache_k,
            cache_v,
            loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scale_buffer[idx],
            self.v_scale_buffer[idx],
            self.sm_k_inv[idx],
            self.sm_v_inv[idx],
        )

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "int4_g32 KV cache: prefix-valid commit is not supported"
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("int4_g32 KV cache: CPU offload is not supported")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("int4_g32 KV cache: CPU offload is not supported")
