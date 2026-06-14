from __future__ import annotations

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
"""

"""
Int8CheckpointStore — int8-compressed storage for CACHED linear-attention
(KDA / GDN / Mamba2 gated-delta-rule) recurrent states.

Why
---
The recurrent state ``S in R^{H x d_v x d_k}`` is large (tens of MB across all
layers). The radix prefix cache stores one full state per reuse point, so the
mamba state pool saturates at a small fraction of the token-KV capacity
(measured on Kimi-Linear-48B-A3B: reuse collapses 0.94 -> 0.05 once distinct prefixes exceed
~1200). The fix is to store CACHED checkpoints compressed.

Why int8 (not fp8) and why it's safe
------------------------------------
A cached checkpoint is loaded ONCE on a cache hit, then decoding continues in
bf16/fp32 — so the only error is a single rounding of S (it never re-enters the
recurrence-with-quant that makes decode-kernel fp8 lossy + slow). Measured on
Kimi-Linear-48B-A3B:
  * int8-per-(head,k-channel) beats fp8-e4m3 at the same 1 byte (the KDA state is
    uniformly distributed, so fp8 wastes bits on the exponent): ~0.5% vs ~0.7%
    decode-output error.
  * End-to-end GSM8K with int8 checkpoints = 0.888 vs 0.898 bf16 (within +/-1.4%
    sampling noise) -> quality-safe.
So storing cached states int8 gives ~2x the cached-prefix capacity at fixed
memory, and composes with host-offload (HiMambaRadixCache) which it also halves.

Design
------
This store is SEPARATE from the active ``MambaPool`` (which stays bf16/fp32 and is
read directly by the kernels). Running requests use the active pool; the radix
caches finished/chunked states HERE in int8, freeing active slots. On a cache hit
the radix dequantizes a checkpoint back into a fresh active slot (copy-on-write).

Quantization is symmetric per ``(layer, slot, head, k-channel)`` — the scale axis
matches the per-k-channel decay structure of the state. The store owns only the
tensors + the (de)quant math; slot lifecycle (alloc / free / evict) is owned by
the caller (mirrors ``MambaPool`` + ``MambaSlotAllocator``).
"""


import torch


class Int8CheckpointStore:
    """int8 store for cached multi-layer linear-attn states.

    Tensors (slot index handed out by the caller's allocator):
        qdata : [L, num_slots, H, d_v, d_k]  int8   (the quantized state)
        scale : [L, num_slots, H, 1,   d_k]  scale_dtype  (per layer,slot,head,k-chan)

    A "state" spans all L mamba layers for one cached point (matching how the
    radix caches one full state per node). The reduction axis for the scale is
    d_v (dim=-2), so each (head, k-channel) gets its own scale — aligned with the
    per-k-channel decay diag(alpha).
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

        amax / scale / round are computed in float32 so quantizing a bf16/fp16
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

    def copy_to_bf16_pool(
        self,
        dst_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Dequantize checkpoints at ``src_slots`` directly into a bf16/fp32 active
        pool tensor ``dst_temporal`` [L, pool_slots, H, d_v, d_k] at ``dst_slots``
        (the copy-on-write on a cache hit)."""
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_bf16_pool(
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
