"""The Muse Spark expert-input side of the attention boundary for a T-token
spec-decode verify window (T = 8 or 16) on TP4, as one persistent Helion
megakernel: pooler, router top-8 routing and the pre-expert NVFP4 quant.

The input is the pre-FFN-normed hidden state (replicated), which the stock
O-projection, its all-reduce and FusedPostNormResidualPreNorm produce.

Roots (rank r):

    R1  router split-K partials                                         [S_r]
    P1  pooler split-K partials                                  [N/W/128, S_p]
    P2  pooler partial sum -> BF16 pooled quarter (symm)              [N/W/32]
    R2  router partial sum -> logit K-shard sum (symm)                [tokens]
    R3  rank-order logit sum and top-8 -> packed routes               [tokens]
    Q   every quarter -> RMSNorm and NVFP4                            [tokens]
"""

from __future__ import annotations

import helion
import helion.language as hl
import msgspec
import torch

from sglang.kernels.ops.muse_spark_v12.helion_megakernels.fp4_utils import (
    encode_block_scale,
    pack_e2m1x2,
)
from sglang.kernels.ops.muse_spark_v12.helion_megakernels.moe_tp4_verify_megakernel import (
    MoeVerifyInputs,
)
from sglang.kernels.ops.muse_spark_v12.helion_norm_gate import (
    _exp2_approx_ftz_f32 as exp2_approx_ftz_f32,
)
from sglang.kernels.ops.muse_spark_v12.helion_norm_gate import (
    _rsqrt_approx_ftz_f32 as rsqrt_approx_ftz_f32,
)

HIDDEN = 8192
EXPERT_HIDDEN = 4096
NUM_EXPERTS = 256
TOPK = 8
SPLIT_P = 16
SPLIT_R = 64
P_ROWS = 128
C_BLOCK = 32
# 1 / (fp8_e4m3_max * 6), the per-token NVFP4 global scale.
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
_LOG2_E = 1.4426950408889634


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _div_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "div.full.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def _columns(value: torch.Tensor, width: int) -> list[torch.Tensor]:
    """Split the last axis into ``width`` tensors in increasing column order."""

    if width == 1:
        return [value.reshape(value.shape[:-1])]
    even, odd = hl.split(value.reshape(*value.shape[:-1], width // 2, 2))
    out: list[torch.Tensor] = []
    for e, o in zip(_columns(even, width // 2), _columns(odd, width // 2), strict=True):
        out.extend((e, o))
    return out


def _chain_sq(cols: list[torch.Tensor]) -> torch.Tensor:
    """fmaf(v_i, v_i, acc) from acc = 0 in column order (one thread's chain)."""

    acc = cols[0] * cols[0]
    for col in cols[1:]:
        acc = _fma(col, col, acc)
    return acc


def _butterfly(value: torch.Tensor, width: int) -> torch.Tensor:
    """xor-shuffle reduction of the last axis, partner distance width/2 first.

    Each size-2 sum is one commutative add, so the result equals
    ``warp::reduce_sum<width>`` bit for bit.
    """

    lead = value.shape[:-1]
    while width > 1:
        value = value.reshape(*lead, 2, width // 2).sum(dim=-2)
        width //= 2
    return value.reshape(lead)


def _pick(values: torch.Tensor, index: int, size: int) -> torch.Tensor:
    """Scalar ``values[index]`` of a small vector (exact: one nonzero term)."""

    return torch.sum(torch.where(hl.arange(size) == index, values, 0.0), dim=-1)


def _route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    router_multiplier: float,
    route_scale: float,
    route_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased sigmoid, stable top-8 (key desc, id asc), renormalized weights."""

    scaled_logits = logits * router_multiplier
    expo = exp2_approx_ftz_f32((0.0 - scaled_logits) * _LOG2_E)
    score = _div_full(torch.ones_like(expo), expo + 1.0)
    key = score + bias
    # One int64 max per round: order-preserving key bits (+0.0 folds -0) above
    # the complemented id, so ties go to the lower id.
    kbits = (key + 0.0).view(dtype=torch.int32)
    ordered = torch.where(kbits >= 0, kbits, kbits ^ 0x7FFFFFFF).to(torch.int64)
    packed_key = (ordered << 32) | (num_experts - 1 - experts).to(torch.int64)
    taken = (hl.zeros([num_experts], dtype=torch.int64) - 1) << 63
    slots8 = hl.arange(8)
    sel_ids = hl.zeros([8], dtype=torch.int32)
    for kk in hl.static_range(8):
        best = hl.reduce(_max, packed_key, dim=0)
        best_id = (num_experts - 1) - (best & 0xFFFF).to(torch.int32)
        packed_key = torch.where(packed_key == best, taken, packed_key)
        sel_ids = torch.where(slots8 == kk, best_id, sel_ids)
    sel_scores = torch.sum(
        torch.where(experts[None, :] == sel_ids[:, None], score[None, :], 0.0), dim=-1
    )
    score_sum = _pick(sel_scores, 0, 8) + _pick(sel_scores, 1, 8)
    for kk in hl.static_range(2, 8):
        score_sum = score_sum + _pick(sel_scores, kk, 8)
    zeros8 = hl.zeros([8], dtype=torch.float32)
    scale = _div_full(zeros8 + route_scale, zeros8 + (score_sum + route_eps))
    return sel_scores * scale, sel_ids


def _attention_boundary_verify_megakernel(
    ffn_in: torch.Tensor,  # [T, N] bf16, the pre-FFN-normed hidden state
    wp: torch.Tensor,  # [Ne / W / P_ROWS, S_p, P_ROWS, N / S_p] bf16, pooler rows
    wr: torch.Tensor,  # [S_r, N / W / S_r, E] fp32, this rank's router K rows
    q_pre_w: torch.Tensor,  # [Ne] bf16
    router_bias: torch.Tensor,  # [E] fp32
    part_p: torch.Tensor,  # [S_p, Ne / W, T] fp32
    part_r: torch.Tensor,  # [S_r, T, E] fp32
    pooled: torch.Tensor,  # [T, Ne / W] bf16 symm
    logit_q: torch.Tensor,  # [T, E] fp32 symm
    x_q: torch.Tensor,  # [T, Ne / 2] u8
    x_sf: torch.Tensor,  # [T, Ne / 16] u8 (E4M3 bits)
    x_ts: torch.Tensor,  # [T] fp32
    packed: torch.Tensor,  # [T, 8] int32
    topk_ids: torch.Tensor,  # [T, 8] int32
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    quant_eps: hl.constexpr,
    global_scale_inv: hl.constexpr,
    router_multiplier: hl.constexpr,
    route_scale: hl.constexpr,
    route_eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(ffn_in.size())
    hl.specialize(ffn_in.stride())
    hl.specialize(wp.size())
    hl.specialize(wp.stride())
    hl.specialize(wr.size())
    hl.specialize(wr.stride())
    hl.specialize(q_pre_w.size())
    hl.specialize(q_pre_w.stride())
    hl.specialize(router_bias.size())
    hl.specialize(router_bias.stride())
    hl.specialize(part_p.size())
    hl.specialize(part_p.stride())
    hl.specialize(part_r.size())
    hl.specialize(part_r.stride())
    hl.specialize(pooled.size())
    hl.specialize(pooled.stride())
    hl.specialize(logit_q.size())
    hl.specialize(logit_q.stride())
    hl.specialize(x_q.size())
    hl.specialize(x_q.stride())
    hl.specialize(x_sf.size())
    hl.specialize(x_sf.stride())
    hl.specialize(x_ts.size())
    hl.specialize(x_ts.stride())
    hl.specialize(packed.size())
    hl.specialize(packed.stride())
    hl.specialize(topk_ids.size())
    hl.specialize(topk_ids.stride())

    tokens = ffn_in.size(0)
    hidden = ffn_in.size(1)
    s_p = wp.size(1)
    k_p = wp.size(3)
    s_r = wr.size(0)
    k_r = wr.size(1)
    num_experts = wr.size(2)
    topk = packed.size(1)
    n_local = wp.size(0) * wp.size(2)
    world = hidden // (s_r * k_r)
    quant_lanes = q_pre_w.size(0) // 16
    q_local = n_local // 16
    kp_blk = hl.register_block_size(k_p)
    kr_blk = hl.register_block_size(k_r)
    x = ffn_in.view(tokens, s_p, k_p)
    xr = ffn_in.view(tokens, world, s_r, k_r)
    remote_pooled = torch.ops.symm_mem.get_remote_tensors(pooled, group_name)
    remote_logit = torch.ops.symm_mem.get_remote_tensors(logit_q, group_name)

    hl.pdl_wait()
    # Trigger at entry: the MoE megakernel's CTAs land on SMs this grid frees
    # early, and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # R1. Stock: the router GEMM (gate, FP32); each rank takes one K-shard here.
    for r1_s in hl.tile(s_r, block_size=1):
        acc_r = hl.zeros([tokens, num_experts], dtype=torch.float32)
        for r1_k in hl.tile(k_r, block_size=kr_blk):
            acc_r = hl.dot(
                xr[:, rank, r1_s.begin, r1_k].to(torch.float32),
                hl.load(wr, [r1_s.begin, r1_k, slice(None)]),
                acc=acc_r,
            )
        part_r[r1_s.begin, :, :] = acc_r

    # P1. Stock: the pooler GEMM (pre_expert_proj).
    for p1_n, p1_s in hl.tile([n_local, s_p], block_size=[P_ROWS, 1]):
        acc = hl.zeros([p1_n, tokens], dtype=torch.float32)
        for p1_k in hl.tile(k_p, block_size=kp_blk):
            acc = hl.dot(
                hl.load(wp, [p1_n.id, p1_s.begin, slice(None), p1_k]),
                x[:, p1_s.begin, p1_k].permute(1, 0),
                acc=acc,
            )
        part_p[p1_s.begin, p1_n, :] = acc

    # P2. Stock: the pooler GEMM's split-K reduce.
    for p2_c in hl.tile(n_local, block_size=C_BLOCK):
        total = torch.sum(part_p[:, p2_c, :], dim=0)
        pooled[:, p2_c] = total.permute(1, 0).to(torch.bfloat16)

    # R2. Stock: the router GEMM's split-K reduce.
    for r2_t in hl.tile(tokens, block_size=1):
        logit_q[r2_t, :] = torch.sum(part_r[:, r2_t, :], dim=0)

    # R3. Stock: FusedMuseSparkRouting (top-8), after summing the K-shard logits.
    for r3_t in hl.tile(tokens, block_size=1):
        experts = hl.arange(num_experts)
        bias = router_bias[experts]
        logits = hl.zeros([num_experts], dtype=torch.float32)
        # Rank order on every rank, so the routes are replicated.
        for peer_logit in remote_logit:
            logits = logits + peer_logit[r3_t.begin, experts]
        weights, sel_ids = _route(
            logits,
            bias,
            experts,
            num_experts,
            router_multiplier,
            route_scale,
            route_eps,
        )
        bits = weights.to(torch.bfloat16).view(dtype=torch.int16).to(torch.int32)
        packed[r3_t.begin, hl.arange(topk)] = (sel_ids << 16) | (bits & 0xFFFF)
        topk_ids[r3_t.begin, hl.arange(topk)] = sel_ids

    # Q. Stock: the pooler's all-gather (gemm_ag), then fused_norm_fp4_quant.
    for q_t in hl.tile(tokens, block_size=1):
        lanes = hl.arange(quant_lanes)
        elem = lanes[:, None] * 16 + hl.arange(16)[None, :]
        norm_w = q_pre_w[elem].to(torch.float32)
        quarters = hl.zeros([world, q_local, 16], dtype=torch.float32)
        owner = hl.arange(world)[:, None, None]
        for peer, peer_pooled in enumerate(remote_pooled):
            row = peer_pooled[q_t.begin, hl.arange(n_local)].to(torch.float32)
            quarters = torch.where(owner == peer, row.reshape(1, q_local, 16), quarters)
        xv = quarters.reshape(quant_lanes, 16)
        # fused_norm_fp4_quant order: 16-column chains, then two butterflies.
        sum_lane = _chain_sq(_columns(xv, 16))
        warp_sums = _butterfly(sum_lane.reshape(quant_lanes // 32, 32), 32)
        sum_sq = _butterfly(warp_sums, quant_lanes // 32)
        rstd = rsqrt_approx_ftz_f32(sum_sq * (1.0 / q_pre_w.size(0)) + quant_eps)
        normed = (xv * rstd * norm_w).to(torch.bfloat16).to(torch.float32)
        # Max via hl.reduce: an amax lowers to a Helion helper that would cost
        # these waiting roots their dry pass.
        block_amax = hl.reduce(_max, torch.abs(normed), dim=1)
        token_scale = hl.reduce(_max, block_amax, dim=0) * global_scale_inv
        fp8_scale, out_scale = encode_block_scale(
            token_scale + hl.zeros([quant_lanes], dtype=torch.float32), block_amax
        )
        low, high = hl.split((normed * out_scale[:, None]).reshape(quant_lanes, 8, 2))
        fp4_bytes = pack_e2m1x2(low, high).to(torch.uint8)
        x_q[q_t.begin, lanes[:, None] * 8 + hl.arange(8)[None, :]] = fp4_bytes
        x_sf[q_t.begin, lanes] = fp8_scale.to(torch.uint8)
        x_ts[q_t.begin + hl.arange(1)] = token_scale + hl.zeros(
            [1], dtype=torch.float32
        )
    return packed


def make_config(
    *,
    kp_blk: int = 64,
    kr_blk: int = 32,
    num_warps: int = 8,
    num_stages: int = 3,
    pipeline: str = "dynamic",
    pid_type: str = "persistent_blocked",
    **extra: object,
) -> helion.Config:
    return helion.Config(
        block_sizes=[kp_blk, kr_blk],
        cross_loop_pipeline=pipeline,
        indexing="pointer",
        num_stages=num_stages,
        num_warps=num_warps,
        pid_type=pid_type,
        **extra,
    )


def build_kernel(config: helion.Config) -> object:
    # tf32x3 keeps the router near the stock FP32 SIMT GEMM; the BF16 dots ignore it.
    return helion.kernel(
        _attention_boundary_verify_megakernel,
        config=config,
        static_shapes=True,
        backend="triton",
        dot_precision="tf32x3",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


class AttentionBoundaryVerifyScalars(msgspec.Struct, frozen=True):
    """The pre-expert NVFP4 quant and routing scalars."""

    quant_eps: float
    router_multiplier: float
    route_scale: float
    route_eps: float


class AttentionBoundaryVerifyWeights(msgspec.Struct, frozen=True):
    """One layer's weights, as views (the router's K rows are a transposed copy)."""

    pooler: torch.Tensor  # [Ne / W / P_ROWS, S_p, P_ROWS, N / S_p] bf16
    router: torch.Tensor  # [N / W, E] fp32, this rank's router K rows
    pre_expert_norm: torch.Tensor  # [Ne] bf16
    router_bias: torch.Tensor  # [E] fp32


class MuseAttentionBoundaryVerifyMegakernel:
    """Per-rank buffers and call for the verify attention boundary.

    The call takes the pre-FFN-normed hidden state, writes the NVFP4 expert
    input and packed routes into the MoE's input slot (the first T activation
    rows), and returns the top-8 ids, replicated on every rank.
    """

    def __init__(
        self,
        device: torch.device,
        group: object,
        scalars: AttentionBoundaryVerifyScalars,
        config: helion.Config | None = None,
        *,
        tokens: int,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.kernel = build_kernel(config or make_config())
        self.group_name = group.group_name
        self.rank = group.rank()
        self.world = group.size()
        self.tokens = tokens
        n_local = EXPERT_HIDDEN // self.world
        self.pooled = symm_mem.empty(
            tokens, n_local, dtype=torch.bfloat16, device=device
        )
        self.logit_q = symm_mem.empty(
            tokens, NUM_EXPERTS, dtype=torch.float32, device=device
        )
        for buffer in (self.pooled, self.logit_q):
            symm_mem.rendezvous(buffer, self.group_name)
        self.part_p = torch.empty(
            SPLIT_P, n_local, tokens, dtype=torch.float32, device=device
        )
        self.part_r = torch.empty(
            SPLIT_R, tokens, NUM_EXPERTS, dtype=torch.float32, device=device
        )
        self.constants = (
            self.rank,
            scalars.quant_eps,
            GLOBAL_SCALE_INV,
            scalars.router_multiplier,
            scalars.route_scale,
            scalars.route_eps,
        )

    def prepare_weights(
        self,
        *,
        pre_expert: torch.Tensor,
        router: torch.Tensor,
        pre_expert_norm: torch.Tensor,
        router_bias: torch.Tensor,
        rank: int,
        world: int,
    ) -> AttentionBoundaryVerifyWeights:
        """One-time per layer: ``router`` is the full HF ``[E, N]`` FP32 weight,
        ``pre_expert`` the full ``[Ne, N]`` pooler."""

        if (rank, world) != (self.rank, self.world):
            raise ValueError(f"weights for rank {rank}/{world} on {self.rank}")
        n_local = pre_expert.size(0) // world
        k_local = router.size(1) // world
        return AttentionBoundaryVerifyWeights(
            # [N/W/P_ROWS, SPLIT_P, P_ROWS, K/SPLIT_P]: each P1 task's tile is dense.
            pooler=pre_expert[rank * n_local : (rank + 1) * n_local]
            .reshape(n_local // P_ROWS, P_ROWS, SPLIT_P, -1)
            .permute(0, 2, 1, 3)
            .contiguous(),
            router=router[:, rank * k_local : (rank + 1) * k_local]
            .t()
            .float()
            .contiguous(),
            pre_expert_norm=pre_expert_norm.reshape(-1).bfloat16().contiguous(),
            router_bias=router_bias.reshape(-1).float().contiguous(),
        )

    def __call__(
        self,
        ffn_in: torch.Tensor,
        weights: AttentionBoundaryVerifyWeights,
        expert_inputs: MoeVerifyInputs,
    ) -> torch.Tensor:
        topk_ids = torch.empty(
            self.tokens, TOPK, dtype=torch.int32, device=ffn_in.device
        )
        router = weights.router
        self.kernel(
            ffn_in,
            weights.pooler,
            router.view(SPLIT_R, -1, router.size(1)),
            weights.pre_expert_norm,
            weights.router_bias,
            self.part_p,
            self.part_r,
            self.pooled,
            self.logit_q,
            expert_inputs.fp4,
            expert_inputs.scales,
            expert_inputs.per_token_scale,
            expert_inputs.packed,
            topk_ids,
            self.group_name,
            *self.constants,
        )
        return topk_ids


__all__ = [
    "MuseAttentionBoundaryVerifyMegakernel",
    "AttentionBoundaryVerifyScalars",
    "AttentionBoundaryVerifyWeights",
    "build_kernel",
    "make_config",
]
