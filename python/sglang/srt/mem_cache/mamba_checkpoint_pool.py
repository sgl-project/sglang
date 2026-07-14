"""
Copyright 2023-2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MambaCheckpointPool — the radix prefix cache's int8-compressed store for cached
linear-attention (KDA / GDN / Mamba2 gated-delta-rule) recurrent states.

It decouples the *cached* states (radix-owned, idle, compressed) from the *active*
``MambaPool`` (running requests, full precision, kernel-facing). The radix stores
one cached state per node HERE; on a prefix-cache hit it is dequantized back into
a fresh active slot (copy-on-write).

Per cached slot it holds:
  * the SSM temporal state in **int8** (per-(head,k-channel) symmetric), via the
    embedded ``Int8CheckpointStore`` — ~2x more cached states than bf16,
    quality-safe (quantized once on store, dequantized once on a hit; never
    re-enters the recurrence as a quant->dequant loop).
  * the conv1d window state at its native dtype (tiny, W-1 tokens; not worth
    quantizing).

Why int8 (not fp8): a cached checkpoint is loaded ONCE on a cache hit, then
decoding continues at full precision, so the only error is a single rounding of
S. The temporal state is roughly uniformly distributed, so int8-per-(head,
k-channel) beats fp8-e4m3 at the same 1 byte (fp8 wastes bits on the exponent).
The scale axis (reduces over d_v) matches the per-k-channel decay diag(alpha), so
the large state entries keep ~bf16 precision and the error concentrates on small
entries that barely affect the readout. Storing cached states int8 gives ~2x the
cached-prefix capacity at fixed memory, and composes with host-offload
(HiMambaRadixCache) which it also halves.

This is strategy-agnostic: whether the active slot to be cached was produced by
the ``no_buffer`` donate (copy_from) or the ``extra_buffer`` ping-pong track
buffer (spec path), both converge on "an active slot becomes the cached
``mamba_value``" — which is exactly the (store_from_active) hook here. Slot
lifecycle is owned by the caller via the embedded ``MambaSlotAllocator``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.utils.common import is_npu

_is_npu = is_npu()

logger = logging.getLogger(__name__)


class Int8CheckpointStore:
    """int8 store for cached multi-layer linear-attn states.

    Tensors (slot index handed out by the caller's allocator):
        qdata : [L, num_slots, H, d_v, d_k]  int8        (the quantized state)
        scale : [L, num_slots, H, 1,   d_k]  scale_dtype  (per layer,slot,head,k-chan)

    A "state" spans all L mamba layers for one cached point (matching how the
    radix caches one full state per node). The reduction axis for the scale is
    d_v (dim=-2), so each (head, k-channel) gets its own scale — aligned with the
    per-k-channel decay diag(alpha).

    ``scale_dtype`` should match the source state's dtype (bf16 / fp16 / fp32) so
    that quantize and dequantize use the identical scale — it is NOT required to
    be bf16.
    """

    QMAX = 127

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        device: str,
        scale_dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.H = num_heads
        self.d_v = head_v_dim
        self.d_k = head_k_dim
        self.device = device
        self.qdata = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            head_v_dim,
            head_k_dim,
            dtype=torch.int8,
            device=device,
        )
        self.scale = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            1,
            head_k_dim,
            dtype=scale_dtype,
            device=device,
        )

    # ---- (de)quant math (also usable standalone for probes/tests) ----

    @classmethod
    def quantize(cls, state: torch.Tensor):
        """state [..., H, d_v, d_k] -> (qint8, scale[..., H, 1, d_k]).

        amax / scale / round are computed in float32 so quantizing a low-precision
        state doesn't lose precision in the intermediate (symmetric with
        ``dequantize``, which is already float32). The scale is rounded to the
        state dtype (its storage precision) BEFORE the division, so quantize and
        dequantize use the identical scale."""
        state_fp32 = state.to(torch.float32)
        amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = (amax / cls.QMAX).to(state.dtype)
        q = (
            torch.round(state_fp32 / scale.to(torch.float32))
            .clamp(-cls.QMAX, cls.QMAX)
            .to(torch.int8)
        )
        return q, scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype):
        return (q.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)

    # ---- store / load (caller supplies slot indices) ----

    def store(self, slots: torch.Tensor, state: torch.Tensor) -> None:
        """Quantize and write states. state: [L, N, H, d_v, d_k] for the N slots
        (or [L, H, d_v, d_k] when slots is a scalar/len-1)."""
        if state.dim() == 4:
            state = state.unsqueeze(1)
        q, scale = self.quantize(state)
        self.qdata[:, slots] = q
        self.scale[:, slots] = scale.to(self.scale.dtype)

    def load(self, slots: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize states at slots -> [L, N, H, d_v, d_k] in out_dtype."""
        return self.dequantize(self.qdata[:, slots], self.scale[:, slots], out_dtype)

    def copy_to_pool(
        self,
        dst_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Dequantize checkpoints at ``src_slots`` directly into the active pool
        tensor ``dst_temporal`` [L, pool_slots, H, d_v, d_k] at ``dst_slots`` (the
        copy-on-write on a cache hit). Output dtype follows ``dst_temporal``."""
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_pool(
        self,
        src_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Quantize states from an active pool tensor into checkpoint slots (cache
        store / donate)."""
        self.store(dst_slots, src_temporal[:, src_slots])

    def mem_usage_bytes(self) -> int:
        return (
            self.qdata.numel() * self.qdata.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots)


class MambaCheckpointPool:
    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        device: str,
        temporal_dtype: Optional[torch.dtype] = None,
    ):
        self.num_slots = num_slots
        self.device = device
        self.temporal = Int8CheckpointStore(
            num_layers=num_layers,
            num_slots=num_slots + 1,  # slot 0 reserved (matches MambaSlotAllocator)
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            device=device,
            # store the scale in the temporal state's own dtype so quantize and
            # dequantize use the identical scale (not hard-coded to bf16)
            scale_dtype=(
                temporal_dtype if temporal_dtype is not None else torch.bfloat16
            ),
        )
        # conv windows stay at their native dtype (small); one buffer per conv
        # tensor in the State
        self.conv = [
            torch.empty(
                (num_layers, num_slots + 1) + tuple(shape),
                dtype=conv_dtype,
                device=device,
            )
            for shape in conv_shapes
        ]
        self.allocator = MambaSlotAllocator(size=num_slots, device=device)

    # ---- lifecycle (delegates to the embedded allocator) ----

    def alloc(self, n: int = 1):
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor):
        self.allocator.free(slots)

    def available_size(self) -> int:
        return self.allocator.available_size()

    def clear(self) -> None:
        """Release every checkpoint slot (radix flush/reset). The int8 qdata is
        left as-is; slots are reused/overwritten on the next store."""
        self.allocator.clear()

    # ---- state transfer between the active MambaPool and this store ----

    def store_from_active(self, active_mamba_pool, active_slots, ckpt_slots) -> None:
        """Quantize temporal + copy conv from the active pool into checkpoint slots
        (the radix donate / cache-store)."""
        cache = active_mamba_pool.mamba_cache
        self.temporal.store_from_pool(cache.temporal, active_slots, ckpt_slots)
        for i, c in enumerate(self.conv):
            src = cache.conv[i][:, active_slots]
            if _is_npu:
                src = src.transpose(2, 3)
            c[:, ckpt_slots] = src

    def load_to_active(self, active_mamba_pool, ckpt_slots, active_slots) -> None:
        """Dequantize temporal + copy conv from checkpoint slots into the active pool
        (the cache-hit copy-on-write)."""
        cache = active_mamba_pool.mamba_cache
        self.temporal.copy_to_pool(cache.temporal, ckpt_slots, active_slots)
        for i, c in enumerate(self.conv):
            src = c[:, ckpt_slots].to(cache.conv[i].dtype)
            if _is_npu:
                src = src.transpose(2, 3) 
            cache.conv[i][:, active_slots] = src

    @staticmethod
    def estimate_mem_usage_bytes(
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        temporal_dtype: torch.dtype,
    ) -> dict:
        """Estimate the pool's HBM footprint (bytes) WITHOUT allocating, so a
        caller can check it against free memory before construction. Mirrors the
        real layout: int8 qdata + per-(head,k) scale + bf16 conv windows, including
        the reserved slot 0."""
        slots = num_slots + 1  # slot 0 reserved (matches MambaSlotAllocator)
        scale_isz = torch.empty((), dtype=temporal_dtype).element_size()
        conv_isz = torch.empty((), dtype=conv_dtype).element_size()
        qdata = num_layers * slots * num_heads * head_v_dim * head_k_dim  # int8 = 1B
        scale = num_layers * slots * num_heads * head_k_dim * scale_isz
        conv = 0
        for shape in conv_shapes:
            n = 1
            for s in shape:
                n *= int(s)
            conv += num_layers * slots * n * conv_isz
        return {
            "qdata": qdata,
            "scale": scale,
            "conv": conv,
            "total": qdata + scale + conv,
        }

    def mem_usage_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        return self.temporal.mem_usage_bytes() + conv_bytes


def maybe_init_int8_mamba_checkpoint_pool(
    *,
    mamba_size: int,
    cache_params,
    mamba_layer_ids: List[int],
    device: str,
) -> Optional[MambaCheckpointPool]:
    """Build the optional int8 ``MambaCheckpointPool`` when
    ``--enable-int8-mamba-checkpoint`` is set (and a global server-args context
    exists), else return ``None``. The radix caches states here (int8) instead of
    in the active bf16 pool -> ~2x cached-prefix capacity at fixed memory.

    Estimates the pool's HBM footprint and checks it against free memory BEFORE
    allocating, so an oversized ``--int8-mamba-ckpt-size`` fails with an actionable
    message instead of a cryptic mid-allocation CUDA OOM.
    """
    from sglang.srt.runtime_context import get_server_args

    try:
        _sa = get_server_args()
    except ValueError:
        # Some unit-test / mock runners construct HybridReqToTokenPool directly
        # without a global server-args context. The int8 checkpoint pool is opt-in
        # via a CLI flag, so an unset context unambiguously means it is off.
        _sa = None
    if not getattr(_sa, "enable_int8_mamba_checkpoint", False):
        return None

    GB = 1 << 30
    H, d_v, d_k = cache_params.shape.temporal
    ckpt_size = _sa.int8_mamba_ckpt_size or (2 * mamba_size)
    kwargs = dict(
        num_layers=len(mamba_layer_ids),
        num_slots=ckpt_size,
        num_heads=H,
        head_v_dim=d_v,
        head_k_dim=d_k,
        conv_shapes=list(cache_params.shape.conv),
        conv_dtype=cache_params.dtype.conv,
        temporal_dtype=cache_params.dtype.temporal,
    )

    est = MambaCheckpointPool.estimate_mem_usage_bytes(**kwargs)
    free_bytes = None
    if isinstance(device, str) and device.startswith("cuda"):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            free_bytes = None
    logger.info(
        f"int8 mamba checkpoint pool: {ckpt_size} slots, "
        f"{est['total'] / GB:.2f}GB (qdata {est['qdata'] / GB:.2f} + scale "
        f"{est['scale'] / GB:.2f} + conv {est['conv'] / GB:.2f}); active mamba "
        f"pool {mamba_size} slots"
        + (f"; free HBM {free_bytes / GB:.2f}GB" if free_bytes is not None else "")
    )
    if free_bytes is not None and est["total"] >= free_bytes:
        raise RuntimeError(
            f"int8 mamba checkpoint pool needs ~{est['total'] / GB:.2f}GB but only "
            f"{free_bytes / GB:.2f}GB HBM is free. Lower --int8-mamba-ckpt-size "
            f"(currently {ckpt_size}) or --mem-fraction-static."
        )

    pool = MambaCheckpointPool(device=device, **kwargs)
    # NOTE: this pool's HBM is NOT subtracted from the KV-cache budget
    # (max_total_num_tokens); it is allocated from --mem-fraction-static headroom.
    # The estimate check above guards against an oversized pool; accounting it in
    # the KV budget is a follow-up.
    logger.warning(
        f"int8 mamba checkpoint pool ({est['total'] / GB:.2f}GB) is allocated from "
        f"--mem-fraction-static headroom and is not reflected in "
        f"max_total_num_tokens; ensure headroom covers it."
    )
    return pool
