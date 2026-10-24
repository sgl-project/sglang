"""Experimental pending KDA windows; callers own checkpoint/slot boundaries.

Enabled only by the experimental serving flag. Operations must run on the state owner's CUDA stream.
Stage only after all layers consumed the previous pending window. Flush before
any fallback forward, checkpoint snapshot, transfer, or slot ownership change.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
    kda_replayssm_exact_fold_kernel,
)


@triton.jit
def _stage(
    V,
    K,
    G,
    Beta,
    PV,
    PK,
    PG,
    PB,
    Counts,
    Slots,
    Accept,
    NS: tl.constexpr,
    H: tl.constexpr,
    SRC_LEN: tl.constexpr,
    CAP: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1)
    head = tl.program_id(2)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0:
        return
    n = tl.load(Accept + row)
    tl.device_assert((n >= 0) & (n <= CAP), "invalid pending replay length")
    base = (layer * NS + slot) * H + head
    x = tl.arange(0, BLOCK)
    valid = x < n * D
    src = base * SRC_LEN * D + x
    dst = base * CAP * D + x
    tl.store(PV + dst, tl.load(V + src, valid, 0), x < CAP * D)
    tl.store(PK + dst, tl.load(K + src, valid, 0), x < CAP * D)
    tl.store(PG + dst, tl.load(G + src, valid, 0), x < CAP * D)
    t = tl.arange(0, CAP)
    tl.store(PB + base * CAP + t, tl.load(Beta + base * SRC_LEN + t, t < n, 0))
    tl.store(Counts + base, n)


@triton.jit
def _clear_pending(
    Counts,
    Slots,
    H: tl.constexpr,
    STRIDE_LAYER: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row, head = x // H, x % H
    slot = tl.load(Slots + row, row < N, -1).to(tl.int64)
    tl.store(
        Counts + layer * STRIDE_LAYER + slot * H + head, 0, (row < N) & (slot >= 0)
    )


class KdaPendingReplay:
    """All-layer windows with per-head consumption counters for fused verify."""

    def __init__(self, temporal, capacity=8):
        layers, slots, heads, value_dim, key_dim = temporal.shape
        if (
            key_dim != 128
            or value_dim != 128
            or capacity != 8
            or temporal.dtype != torch.float32
            or not temporal.is_cuda
        ):
            raise ValueError("Pending replay currently supports K=V=128, capacity8")
        self.temporal = temporal
        shape = (layers, slots, heads, capacity, key_dim)
        self.rawv = torch.empty(shape, device=temporal.device, dtype=torch.bfloat16)
        self.rawk = torch.empty_like(self.rawv)
        self.g = torch.empty(shape, device=temporal.device, dtype=torch.float32)
        self.beta = torch.empty(shape[:-1], device=temporal.device, dtype=torch.float32)
        self.counts = torch.zeros(shape[:3], device=temporal.device, dtype=torch.int32)
        self.indices = torch.arange(slots, device=temporal.device, dtype=torch.int32)
        self.capacity = capacity
        self._may_have_pending = False
        self._stage_in_graph = False

    def mem_usage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.rawv,
                self.rawk,
                self.g,
                self.beta,
                self.counts,
                self.indices,
            )
        )

    def stage(self, rawv, rawk, g, beta, slots, accepted):
        """Snapshot accepted inputs for unique, valid slots (padding -1 allowed).

        Each accepted count must be within capacity. The owner must flush or
        consume any prior window before staging the next one for the same slot.
        """
        nl, ns, h, length, d = rawv.shape
        if (
            rawv.shape != rawk.shape
            or rawv.shape != g.shape
            or beta.shape != rawv.shape[:-1]
            or rawv.shape[:3] != self.counts.shape
            or length < self.capacity
            or d != 128
            or slots.ndim != 1
            or accepted.shape != slots.shape
            or rawv.dtype != torch.bfloat16
            or rawk.dtype != torch.bfloat16
            or g.dtype != torch.float32
            or beta.dtype != torch.float32
            or any(
                x.device != self.temporal.device
                for x in (rawv, rawk, g, beta, slots, accepted)
            )
            or not all(
                x.is_contiguous() for x in (rawv, rawk, g, beta, slots, accepted)
            )
        ):
            raise ValueError("Unsupported pending replay source layout")
        # Serving stages eagerly after acceptance. A captured stage can replay
        # without executing Python, so keep its fallback flush conservative.
        self._may_have_pending = True
        self._stage_in_graph |= torch.cuda.is_current_stream_capturing()
        _stage[(nl, slots.numel(), h)](
            rawv,
            rawk,
            g,
            beta,
            self.rawv,
            self.rawk,
            self.g,
            self.beta,
            self.counts,
            slots,
            accepted,
            ns,
            h,
            length,
            self.capacity,
            d,
            triton.next_power_of_2(self.capacity * d),
            num_warps=4,
        )

    def before_verify(self, *, batch_size, block_size):
        """Run on the forward stream, outside captured graphs, before verify.

        Leaving the supported batch range materializes pending state once.
        Changing batch size within that range does not require a global flush.
        The graph-stage case remains conservative because replays do not update
        Python flags.
        """
        if 1 <= batch_size <= 64 and block_size == 8:
            return
        if self._may_have_pending or self._stage_in_graph:
            self.flush()
            self._may_have_pending = False

    def layer_kwargs(self, layer):
        return dict(
            replay_rawv=self.rawv[layer],
            replay_rawk=self.rawk[layer],
            replay_g=self.g[layer],
            replay_beta=self.beta[layer],
            replay_lens=self.counts[layer],
        )

    def reset_slots(self, indices):
        self.counts.index_fill_(1, indices.to(torch.int64), 0)

    def flush(self, layer=None, indices=None):
        """Materialize every pending checkpoint, then clear counters.

        Also handles partial-layer consumption without replaying consumed heads.
        Boundary-only operation: scanning empty slots is intentional for now.
        """
        temporal = self.temporal if layer is None else self.temporal[layer : layer + 1]
        rawv = self.rawv if layer is None else self.rawv[layer : layer + 1]
        rawk = self.rawk if layer is None else self.rawk[layer : layer + 1]
        g = self.g if layer is None else self.g[layer : layer + 1]
        beta = self.beta if layer is None else self.beta[layer : layer + 1]
        counts = self.counts if layer is None else self.counts[layer : layer + 1]
        nl, ns, h, v, k = temporal.shape
        indices = self.indices if indices is None else indices.contiguous()
        n = indices.numel()
        if n == 0:
            return
        kda_replayssm_exact_fold_kernel[(triton.cdiv(v, 32), n, nl * h)](
            temporal,
            rawv,
            rawk,
            g,
            beta,
            indices,
            counts,
            indices,
            counts,
            temporal.stride(1),
            rawv.stride(1),
            rawk.stride(1),
            g.stride(1),
            beta.stride(1),
            temporal.stride(0),
            rawv.stride(0),
            rawk.stride(0),
            g.stride(0),
            beta.stride(0),
            1,
            counts.stride(1),
            0,
            0,
            H=h,
            HV=h,
            K=k,
            V=v,
            BK=128,
            BV=32,
            MAX_CACHE_LEN=self.capacity,
            USE_QK_L2NORM_IN_KERNEL=True,
            NULL_BLOCK_ID=-1,
            HAS_TRACK=False,
            ACCEPT_LAYER_STRIDE=counts.stride(0),
            ACCEPT_HEAD_STRIDE=1,
            ACCEPT_BY_SLOT=True,
            num_warps=1,
            num_stages=3,
        )
        _clear_pending[(nl, triton.cdiv(n * h, 128))](
            counts, indices, h, counts.stride(0), n, 128, num_warps=4
        )
