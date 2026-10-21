"""Fused per-head RMS norm of the Q and K slices of a QKV projection.

Fuses, per attention layer:
    q = rms_norm(qkv[:, :q_size].reshape(-1, head_dim)).reshape_as(q)
    k = rms_norm(qkv[:, q_size:q_size + kv_size].reshape(-1, head_dim)).reshape_as(k)

``q`` and ``k`` are strided slices of ``qkv``, so ``reshape(-1, head_dim)``
materializes a contiguous copy of each before either norm starts. This op reads
the slices in place and writes the two contiguous normed tensors.

Used in MuseSpark1x attention. The caller folds the query scale into the
attention softmax scale, so no scale is applied here.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.spec import CapabilityRequirement, KernelBackend

FP8_E4M3_MAX = 448.0


@triton.jit
def _qk_rms_norm_fwd_kernel(
    QKV,  # [T, qkv_width] source projection output
    Q_OUT,  # [T, NUM_Q_HEADS * D]
    K_OUT,  # [T, NUM_KV_HEADS * D]
    POS,  # [T] positions; only read when HAS_ROPE
    COS_SIN,  # [max_pos, D] fp32, cos then sin halves; only read when HAS_ROPE
    K_CACHE,  # [N, NUM_KV_HEADS * D] BF16/FP8 pool; written when HAS_KV_STORE
    V_CACHE,  # [N, NUM_KV_HEADS * D]
    CACHE_LOC,  # [T] pool row per token
    eps,
    stride_qkv,  # row stride of QKV, in elements
    n_rows,  # T * (NUM_Q_HEADS + NUM_KV_HEADS [+ NUM_KV_HEADS for V])
    kv_capacity,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_KV_STORE: tl.constexpr,
    FP8_KV_STORE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    FP8_MAX: tl.constexpr = FP8_E4M3_MAX,
):
    # One (token, head) row per lane of the block; fp32 RMS reduction over D.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        # ---- PDL: Trigger next kernel here to increase overlap
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]

    # Q, K (and V when storing the cache) are adjacent in the projection, so
    # one combined head index addresses all of them with one offset.
    if HAS_KV_STORE:
        HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    else:
        HEADS: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS

    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    rstd = tl.rsqrt(var + eps)
    x = x * rstd[:, None]

    if HAS_ROPE:
        # Rope reads the bf16-rounded row, as the separate rope kernel did.
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1
        xp = tl.load(
            QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(
            COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0
        )
        sin = tl.load(
            COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0
        )
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    if FP8_KV_STORE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
    if HAS_KV_STORE:
        x = tl.where(is_v[:, None], raw.to(tl.float32), x)
        if FP8_KV_STORE:
            x = tl.minimum(tl.maximum(x, -FP8_MAX), FP8_MAX)
        loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
        # Slot zero is reserved for CUDA-graph padding.
        kv_mask = mask & (loc != 0)[:, None]
        kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
        kv_dst = loc * (NUM_KV_HEADS * D) + kv_head * D
        tl.store(
            K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_k[:, None]
        )
        tl.store(
            V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_v[:, None]
        )

    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)
    k_dst = tl.where(is_k, (token * NUM_KV_HEADS + head - NUM_Q_HEADS) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])
    tl.store(K_OUT + k_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])


def _triton_config(n_rows: int) -> tuple[int, int]:
    """``(block_m, num_warps)``, swept on B300 (bf16, head_dim 64).

    num_warps was flat at 4 across the sweep.
    """
    if n_rows < 1024:
        return 4, 4
    if n_rows < 8192:
        return 8, 4
    if n_rows < 65536:
        return 16, 4
    return 32, 4


class FusedQKNorm(BaseFusedOp):
    """Per-head RMS norm of the Q and K slices of a QKV projection output."""

    op = "attention.muse_spark_qk_norm_rope_store"
    priority = (KernelBackend.TRITON, KernelBackend.HELION)
    capabilities = {
        KernelBackend.HELION: {CapabilityRequirement.CUDA},
        KernelBackend.TRITON: {
            CapabilityRequirement.CUDA,
        },
    }

    def __init__(
        self,
        *,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
    ):
        super().__init__()
        self._head_dim = head_dim
        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._eps = eps

    def _slices(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_size = self._num_q_heads * self._head_dim
        kv_size = self._num_kv_heads * self._head_dim
        return qkv[:, :q_size], qkv[:, q_size : q_size + kv_size]

    def forward_native(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head norm of q and k, optionally with interleaved rope and a
        KV-cache store of the final k and the raw v at ``cache_loc``.

        An FP8 cache also returns FP8 q, matching the TRT-LLM quantization the
        attention backend would otherwise do in its own kernel. Cache scales
        must be unity.
        """
        q, k = self._slices(qkv)
        d = self._head_dim
        normed = [
            torch.nn.functional.rms_norm(
                t.reshape(-1, d), (d,), None, self._eps
            ).reshape(t.shape)
            for t in (q, k)
        ]
        if positions is not None:
            normed = list(self._rope_native(normed, positions, cos_sin_cache))
        if kv_store is not None:
            self._store_kv(qkv, normed, kv_store)
        return normed[0], normed[1]

    def _store_kv(self, qkv, normed: list, kv_store) -> None:
        """Reference KV-cache write; an FP8 pool saturates as the kernel does.

        Rewrites ``normed[0]`` to FP8 for an FP8 pool, as the kernel does.
        """
        k_cache, v_cache, cache_loc = kv_store
        slots = cache_loc[: qkv.shape[0]].long()
        if torch.any((slots < 0) | (slots >= k_cache.shape[0])):
            raise ValueError("KV cache slot out of bounds")
        # Slot zero is reserved for CUDA-graph padding.
        live = slots != 0
        fp8 = k_cache.dtype == torch.float8_e4m3fn

        def cast(value):
            if fp8:
                value = value.float().clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            return value.to(k_cache.dtype)

        k_cache[slots[live]] = cast(normed[1][live])
        v_cache[slots[live]] = cast(self._v_slice(qkv)[live])
        if fp8:
            normed[0] = cast(normed[0])

    def _v_slice(self, qkv: torch.Tensor) -> torch.Tensor:
        q_size = self._num_q_heads * self._head_dim
        kv_size = self._num_kv_heads * self._head_dim
        return qkv[:, q_size + kv_size : q_size + 2 * kv_size]

    def forward_cpu(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward_native`` with the per-head norm on sgl-kernel.

        Only the norm moves; rope and the KV store stay on the reference path,
        which is where the remaining CPU work for this op is.
        """
        if qkv.dtype not in (torch.bfloat16, torch.float16):
            return self.forward_native(qkv, positions, cos_sin_cache, kv_store)

        q, k = self._slices(qkv)
        d = self._head_dim
        normed = [
            torch.ops.sgl_kernel.l2norm_cpu(t.reshape(-1, d), self._eps).reshape(
                t.shape
            )
            for t in (q, k)
        ]
        if positions is not None:
            normed = list(self._rope_native(normed, positions, cos_sin_cache))
        if kv_store is not None:
            self._store_kv(qkv, normed, kv_store)
        return normed[0], normed[1]

    def _rope_native(self, normed, positions, cos_sin_cache):
        d = self._head_dim
        cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)  # [T, d/2]
        out = []
        for t in normed:
            x = t.float().reshape(t.shape[0], -1, d // 2, 2)
            c = cos[:, None, :, None]
            s_ = sin[:, None, :, None]
            x0, x1 = x[..., 0:1], x[..., 1:2]
            rot = torch.cat([x0 * c - x1 * s_, x0 * s_ + x1 * c], dim=-1)
            out.append(rot.reshape(t.shape).to(t.dtype))
        return out[0], out[1]

    def backend_eligible(
        self,
        backend: KernelBackend,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> bool:
        if not super().backend_eligible(backend, qkv):
            return False
        if backend is KernelBackend.HELION:
            return self._helion_eligible(qkv, positions, cos_sin_cache, kv_store)
        if backend is not KernelBackend.TRITON:
            return True
        if not (qkv.dim() == 2 and qkv.numel() > 0 and qkv.stride(-1) == 1):
            return False
        if kv_store is not None and not (
            kv_store[0].dtype in (qkv.dtype, torch.float8_e4m3fn)
            and kv_store[0].dim() == 2
            and kv_store[0].shape[1] == self._num_kv_heads * self._head_dim
            and kv_store[1].shape == kv_store[0].shape
            and kv_store[1].dtype == kv_store[0].dtype
            and kv_store[2].shape[0] >= tokens_of(qkv)
        ):
            return False
        return positions is None or (
            cos_sin_cache.dtype == torch.float32
            and cos_sin_cache.shape[-1] == self._head_dim
            and self._head_dim % 2 == 0
        )

    def forward_triton(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_triton(qkv, positions, cos_sin_cache, kv_store)

    def _forward_triton(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Triton implementation without model-specific dispatch."""

        tokens = qkv.shape[0]
        d = self._head_dim
        fp8_kv = kv_store is not None and kv_store[0].dtype == torch.float8_e4m3fn
        q_out = torch.empty(
            tokens,
            self._num_q_heads * d,
            dtype=torch.float8_e4m3fn if fp8_kv else qkv.dtype,
            device=qkv.device,
        )
        k_out = torch.empty(
            tokens, self._num_kv_heads * d, dtype=qkv.dtype, device=qkv.device
        )
        has_kv = kv_store is not None
        heads = self._num_q_heads + self._num_kv_heads * (2 if has_kv else 1)
        n_rows = tokens * heads
        block_m, num_warps = _triton_config(n_rows)
        has_rope = positions is not None
        pdl = {"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}
        if has_rope:
            assert cos_sin_cache.is_contiguous() and positions.is_contiguous()
        if has_kv:
            k_cache, v_cache, cache_loc = kv_store
            assert k_cache.is_contiguous() and v_cache.is_contiguous()
            cache_loc = cache_loc.contiguous()
        _qk_rms_norm_fwd_kernel[(triton.cdiv(n_rows, block_m),)](
            qkv,
            q_out,
            k_out,
            positions if has_rope else qkv,
            cos_sin_cache if has_rope else qkv,
            k_cache if has_kv else qkv,
            v_cache if has_kv else qkv,
            cache_loc if has_kv else qkv,
            self._eps,
            qkv.stride(0),
            n_rows,
            k_cache.shape[0] if has_kv else 0,
            self._num_q_heads,
            self._num_kv_heads,
            d,
            triton.next_power_of_2(d),
            block_m,
            HAS_ROPE=has_rope,
            HAS_KV_STORE=has_kv,
            FP8_KV_STORE=fp8_kv,
            num_warps=num_warps,
            **pdl,
        )
        return q_out, k_out

    def _helion_eligible(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> bool:
        from sglang.kernels.jit.utils.helion_runtime import helion_supported

        return (
            helion_supported(qkv)
            and (kv_store is None or kv_store[0].dtype == qkv.dtype)
            and qkv.shape[0] > 0
            and qkv.dtype == torch.bfloat16
            and qkv.dim() == 2
            and qkv.stride(-1) == 1
            and self._head_dim == 64
            and positions is not None
            and cos_sin_cache is not None
            and self.backend_eligible(
                KernelBackend.TRITON,
                qkv,
                positions,
                cos_sin_cache,
                kv_store,
            )
        )

    def forward_helion(
        self,
        qkv: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        kv_store: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Helion behind the existing fused operation boundary."""

        from sglang.kernels.jit.utils.helion_runtime import (
            run_pinned_helion,
        )
        from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store_helion import (
            QK_NORM_CONFIGS,
            _qk_norm_helion_kernel,
        )

        if kv_store is None:
            k_cache = v_cache = cache_loc = None
        else:
            k_cache, v_cache, cache_loc = kv_store
        return run_pinned_helion(
            _qk_norm_helion_kernel,
            QK_NORM_CONFIGS,
            (
                qkv,
                positions,
                cos_sin_cache,
                k_cache,
                v_cache,
                cache_loc,
                self._eps,
                self._num_q_heads,
                self._num_kv_heads,
                self._head_dim,
            ),
            num_tokens=qkv.shape[0],
            launch_pdl=is_arch_support_pdl(),
        )


def tokens_of(qkv: torch.Tensor) -> int:
    return qkv.shape[0]
