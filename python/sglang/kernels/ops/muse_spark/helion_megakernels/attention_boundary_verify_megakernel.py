"""The Muse Spark attention-output -> expert-input boundary for a T-token
spec-decode verify window (T = 8 or 16) on TP4, as one persistent Helion megakernel: O-projection,
all-reduce, post-attention norm/residual/pre-FFN norm, pooler, router, top-8
routing and the pre-expert NVFP4 quant.

The input is this rank's gated attention heads (norm * sigmoid(gate) already
applied by the attention megakernel).

Roots (rank r):

    A   O-projection K-shard GEMV -> BF16 partial rows (symm)     [N / R / CB, CB]
    B   rank-order sum of the partials -> BF16 branch, moments    [T, N / R / CB]
    C   norms -> new stream, FFN input                            [T, N / R / CB]
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

from sglang.kernels.ops.muse_spark.helion_megakernels.fp4_utils import (
    encode_block_scale,
    pack_e2m1x2,
)
from sglang.kernels.ops.muse_spark.helion_megakernels.moe_tp4_verify_megakernel import (
    MoeVerifyInputs,
)
from sglang.kernels.ops.muse_spark.helion_norm_gate import (
    _exp2_approx_ftz_f32 as exp2_approx_ftz_f32,
)
from sglang.kernels.ops.muse_spark.helion_norm_gate import (
    _rsqrt_approx_ftz_f32 as rsqrt_approx_ftz_f32,
)

HIDDEN = 8192
EXPERT_HIDDEN = 4096
NUM_EXPERTS = 256
ROWS = 64
CHUNK_BLOCKS = 16
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
        best = torch.amax(packed_key, dim=-1)
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
    gated: torch.Tensor,  # [T, H * D] bf16, this rank's gated attention heads
    wo: torch.Tensor,  # [N / R / CB, CB, R, H * D] bf16, this rank's K-shard
    stream: torch.Tensor,  # [T, N / R / CB, CB, R] fp32
    alpha: torch.Tensor,  # [N / R / CB, CB, R] fp32
    beta: torch.Tensor,  # [N / R / CB, CB, R] fp32
    pre_w: torch.Tensor,  # [N / R / CB, CB, R] bf16
    partial: torch.Tensor,  # [T, N / R / CB, CB, R] bf16 symm
    branch: torch.Tensor,  # [T, N / R / CB, CB, R] bf16 scratch
    moments: torch.Tensor,  # [N / R / CB, T, 4] fp32 scratch
    new_stream: torch.Tensor,  # [T, N / R / CB, CB, R] fp32
    ffn_in: torch.Tensor,  # [T, N] bf16
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
    post_eps: hl.constexpr,
    pre_eps: hl.constexpr,
    quant_eps: hl.constexpr,
    global_scale_inv: hl.constexpr,
    router_multiplier: hl.constexpr,
    route_scale: hl.constexpr,
    route_eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(gated.size())
    hl.specialize(gated.stride())
    hl.specialize(wo.size())
    hl.specialize(wo.stride())
    hl.specialize(stream.size())
    hl.specialize(stream.stride())
    hl.specialize(alpha.size())
    hl.specialize(alpha.stride())
    hl.specialize(beta.size())
    hl.specialize(beta.stride())
    hl.specialize(pre_w.size())
    hl.specialize(pre_w.stride())
    hl.specialize(partial.size())
    hl.specialize(partial.stride())
    hl.specialize(branch.size())
    hl.specialize(branch.stride())
    hl.specialize(moments.size())
    hl.specialize(moments.stride())
    hl.specialize(new_stream.size())
    hl.specialize(new_stream.stride())
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

    tokens = gated.size(0)
    k = gated.size(1)
    n_chunks = wo.size(0)
    chunk_blocks = wo.size(1)
    rows = wo.size(2)
    hidden = n_chunks * chunk_blocks * rows
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
    k_blk = hl.register_block_size(k)
    kp_blk = hl.register_block_size(k_p)
    kr_blk = hl.register_block_size(k_r)
    ffn_out = ffn_in.view(tokens, n_chunks, chunk_blocks, rows)
    x = ffn_in.view(tokens, s_p, k_p)
    xr = ffn_in.view(tokens, world, s_r, k_r)
    remote_partial = torch.ops.symm_mem.get_remote_tensors(partial, group_name)
    remote_pooled = torch.ops.symm_mem.get_remote_tensors(pooled, group_name)
    remote_logit = torch.ops.symm_mem.get_remote_tensors(logit_q, group_name)

    hl.pdl_wait()
    # Trigger at entry: the next kernel's CTAs land on SMs this grid frees early,
    # and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # A. Stock: the o_proj GEMM (RowParallelLinear, before its all-reduce).
    for a_c, a_b in hl.tile([n_chunks, chunk_blocks], block_size=[1, 1]):
        acc = hl.zeros([tokens, rows], dtype=torch.float32)
        for a_k in hl.tile(k, block_size=k_blk):
            acc = hl.dot(
                gated[:, a_k],
                hl.load(wo, [a_c.begin, a_b.begin, slice(None), a_k]).permute(1, 0),
                acc=acc,
            )
        partial[:, a_c.begin, a_b.begin, :] = acc.to(torch.bfloat16)

    # B. Stock: the o_proj one-shot all-reduce, plus the moments C's norms need.
    for b_t, b_c in hl.tile([tokens, n_chunks], block_size=[1, 1]):
        gated_stream = alpha[b_c.begin, :, :][None, :, :] * stream[b_t, b_c.begin, :, :]
        beta_c = beta[b_c.begin, :, :][None, :, :]
        total = hl.zeros([b_t, chunk_blocks, rows], dtype=torch.float32)
        # Rank order, as the one-shot all-reduce sums.
        for peer_partial in remote_partial:
            total = total + peer_partial[b_t, b_c.begin, :, :].to(torch.float32)
        v = total.to(torch.bfloat16)
        branch[b_t, b_c.begin, :, :] = v
        v = v.to(torch.float32)
        gated_branch = beta_c * v
        m = hl.arange(4)[None, :]
        moments[b_c.begin, b_t, :] = (
            torch.where(
                m == 0, torch.sum(torch.sum(v * v, dim=-1), dim=-1)[:, None], 0.0
            )
            + torch.where(
                m == 1,
                torch.sum(torch.sum(gated_stream * gated_stream, dim=-1), dim=-1)[
                    :, None
                ],
                0.0,
            )
            + torch.where(
                m == 2,
                torch.sum(torch.sum(gated_stream * gated_branch, dim=-1), dim=-1)[
                    :, None
                ],
                0.0,
            )
            + torch.where(
                m == 3,
                torch.sum(torch.sum(gated_branch * gated_branch, dim=-1), dim=-1)[
                    :, None
                ],
                0.0,
            )
        )

    # C. Stock: FusedPostNormResidualPreNorm (post-attention norm, residual,
    # pre-FFN norm).
    for c_t, c_c in hl.tile([tokens, n_chunks], block_size=[1, 1]):
        gated_stream = alpha[c_c.begin, :, :][None, :, :] * stream[c_t, c_c.begin, :, :]
        beta_c = beta[c_c.begin, :, :][None, :, :]
        pre = pre_w[c_c.begin, :, :].to(torch.float32)[None, :, :]
        total = torch.sum(moments[:, c_t, :], dim=0)
        m = hl.arange(4)[None, :]
        post_sq = torch.sum(torch.where(m == 0, total, 0.0), dim=-1)
        stream_sq = torch.sum(torch.where(m == 1, total, 0.0), dim=-1)
        cross = torch.sum(torch.where(m == 2, total, 0.0), dim=-1)
        branch_sq = torch.sum(torch.where(m == 3, total, 0.0), dim=-1)
        post_rstd = torch.rsqrt(post_sq / hidden + post_eps)
        pre_var = (
            stream_sq + 2.0 * post_rstd * cross + post_rstd * post_rstd * branch_sq
        ) / hidden
        pre_rstd = torch.rsqrt(pre_var + pre_eps)[:, None, None]
        v = branch[c_t, c_c.begin, :, :].to(torch.float32)
        normed = (v * post_rstd[:, None, None]).to(torch.bfloat16).to(torch.float32)
        updated = gated_stream + beta_c * normed
        new_stream[c_t, c_c.begin, :, :] = updated
        ffn_out[c_t, c_c.begin, :, :] = (updated * pre_rstd * pre).to(torch.bfloat16)

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
        block_amax = torch.amax(torch.abs(normed), dim=-1)
        token_scale = torch.amax(block_amax, dim=-1) * global_scale_inv
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
    k_blk: int = 256,
    kp_blk: int = 128,
    kr_blk: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    pipeline: str = "dynamic",
    pid_type: str = "persistent_blocked",
    **extra: object,
) -> helion.Config:
    return helion.Config(
        block_sizes=[k_blk, kp_blk, kr_blk],
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
    """Boundary norm eps, then the NVFP4 quant and routing scalars."""

    post_eps: float
    pre_eps: float
    quant_eps: float
    router_multiplier: float
    route_scale: float
    route_eps: float


class AttentionBoundaryVerifyWeights(msgspec.Struct, frozen=True):
    """One layer's weights, as views (the router's K rows are a transposed copy)."""

    o_proj: torch.Tensor  # [N, H * D / W] bf16, this rank's K-shard
    pooler: torch.Tensor  # [Ne / W / P_ROWS, S_p, P_ROWS, N / S_p] bf16
    router: torch.Tensor  # [N / W, E] fp32, this rank's router K rows
    alpha: torch.Tensor  # [N] fp32
    beta: torch.Tensor  # [N] fp32
    pre_ffn_norm: torch.Tensor  # [N] bf16
    pre_expert_norm: torch.Tensor  # [Ne] bf16
    router_bias: torch.Tensor  # [E] fp32


class MuseAttentionBoundaryVerifyMegakernel:
    """Per-rank buffers and call for the verify attention boundary.

    The call takes this rank's gated attention heads and the residual stream,
    writes the NVFP4 expert input and packed routes into the MoE's input slot
    (the first T activation rows), and returns the new stream, the FFN input and
    the top-8 ids, replicated on every rank.
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
        self.shape = (HIDDEN // ROWS // CHUNK_BLOCKS, CHUNK_BLOCKS, ROWS)
        n_local = EXPERT_HIDDEN // self.world
        self.partial = symm_mem.empty(
            tokens, *self.shape, dtype=torch.bfloat16, device=device
        )
        self.pooled = symm_mem.empty(
            tokens, n_local, dtype=torch.bfloat16, device=device
        )
        self.logit_q = symm_mem.empty(
            tokens, NUM_EXPERTS, dtype=torch.float32, device=device
        )
        for buffer in (self.partial, self.pooled, self.logit_q):
            symm_mem.rendezvous(buffer, self.group_name)
        self.branch = torch.empty(
            tokens, *self.shape, dtype=torch.bfloat16, device=device
        )
        self.moments = torch.empty(
            self.shape[0], tokens, 4, dtype=torch.float32, device=device
        )
        self.part_p = torch.empty(
            SPLIT_P, n_local, tokens, dtype=torch.float32, device=device
        )
        self.part_r = torch.empty(
            SPLIT_R, tokens, NUM_EXPERTS, dtype=torch.float32, device=device
        )
        self.constants = (
            self.rank,
            scalars.post_eps,
            scalars.pre_eps,
            scalars.quant_eps,
            GLOBAL_SCALE_INV,
            scalars.router_multiplier,
            scalars.route_scale,
            scalars.route_eps,
        )

    def prepare_weights(
        self,
        *,
        o_proj: torch.Tensor,
        pre_expert: torch.Tensor,
        router: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_ffn_norm: torch.Tensor,
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
            o_proj=o_proj.contiguous(),
            # [N/W/P_ROWS, SPLIT_P, P_ROWS, K/SPLIT_P]: each P1 task's tile is dense.
            pooler=pre_expert[rank * n_local : (rank + 1) * n_local]
            .reshape(n_local // P_ROWS, P_ROWS, SPLIT_P, -1)
            .permute(0, 2, 1, 3)
            .contiguous(),
            router=router[:, rank * k_local : (rank + 1) * k_local]
            .t()
            .float()
            .contiguous(),
            alpha=alpha.reshape(-1).float().contiguous(),
            beta=beta.reshape(-1).float().contiguous(),
            pre_ffn_norm=pre_ffn_norm.reshape(-1).bfloat16().contiguous(),
            pre_expert_norm=pre_expert_norm.reshape(-1).bfloat16().contiguous(),
            router_bias=router_bias.reshape(-1).float().contiguous(),
        )

    def __call__(
        self,
        gated: torch.Tensor,
        stream: torch.Tensor,
        weights: AttentionBoundaryVerifyWeights,
        expert_inputs: MoeVerifyInputs,
    ) -> dict[str, torch.Tensor]:
        shape, tokens, device = self.shape, self.tokens, stream.device
        new_stream = torch.empty_like(stream)
        ffn_in = torch.empty(stream.shape, dtype=torch.bfloat16, device=device)
        topk_ids = torch.empty(tokens, TOPK, dtype=torch.int32, device=device)
        o_proj, pooler, router = weights.o_proj, weights.pooler, weights.router
        self.kernel(
            gated.view(tokens, -1),
            o_proj.view(*shape, -1),
            stream.view(tokens, *shape),
            weights.alpha.view(shape),
            weights.beta.view(shape),
            weights.pre_ffn_norm.view(shape),
            self.partial,
            self.branch,
            self.moments,
            new_stream.view(tokens, *shape),
            ffn_in,
            pooler,
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
        return {"new_stream": new_stream, "ffn_in": ffn_in, "topk_ids": topk_ids}


__all__ = [
    "AttentionBoundaryVerifyScalars",
    "AttentionBoundaryVerifyWeights",
    "MuseAttentionBoundaryVerifyMegakernel",
    "build_kernel",
    "make_config",
]
