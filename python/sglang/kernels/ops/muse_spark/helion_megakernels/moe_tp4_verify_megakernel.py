"""Muse Spark 1.2 spec-decode verify expert-TP4 NVFP4 MoE with the
post-expert norm finalize, as one persistent Helion megakernel.

Every rank reads one weight bank: its FI TRT-LLM W13 I/W slices (aliased by
the stock parameters) followed by its H/W quarter of W2 over the full K, stored
as contiguous 256 x 256 B tiles. The routed activation z crosses ranks once; the
replicated y is gathered from every rank's quarter rows.

Roots (j / f: the routed experts in ascending order, ``slots`` bound):

    R1 W13 256-row tile, SwiGLU, NVFP4 z pushed to every rank     [2I/W/256, slots]
    R2 W2 quarter tile on all of z, routed rows and per-quarter
       sums of (norm_w * x)^2 scattered to symm by route          [H/W/256, slots]
    R3 per-route RMSNorm and weighted route sum of every rank's
       quarter -> y (replicated)                                  [tokens, H/W/128]

FC2 follows FlashInfer: the FC2 input scale is folded into ``up_alpha`` and
``out_alpha``, so z is quantized with a unit global scale.

The tail is a second launch: the unpooler on this rank's N/W rows in C-column
blocks, then the post-FFN RMSNorm, the gated residual and the next layer's
pre-attention RMSNorm. Each block's BF16 rows and four per-token moments
(|v|^2, |a s|^2, <a s, b v>, |b v|^2) cross ranks once; outputs are replicated.
Its last root is the next layer's QKV GEMV on that attention input.
"""

from __future__ import annotations

import helion
import helion.language as hl
import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.ops.muse_spark.helion_megakernels.fp4_utils import (
    encode_block_scale,
    pack_e2m1x2,
)

# dot_scaled needs N >= 16: a shorter verify window pads the MMA token rows to 16
# with zero activations; routes and outputs keep T rows.
MOE_TOKENS = 16
TOPK = 8
MMA_M = 128
TILE_M = 256
TILE_K = 256  # e2m1x2 bytes per K step
C_BLOCK = 16
N_BLOCKS = 16
QKV_ROWS = 32


def _or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a | b


def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def _popcount(v: torch.Tensor) -> torch.Tensor:
    """Set bits per int32 lane (SWAR: Triton cannot scan inside a WS loop)."""

    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def _distinct_table(
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routed-expert bitmap words [8], their popcounts and exclusive prefix, and the
    distinct count; tables this narrow reduce within one warp."""

    lanes = hl.arange(8)
    row = ids.unsqueeze(0)
    bits = torch.where((row >> 5) == lanes.unsqueeze(1), 1 << (row & 31), 0)
    words = hl.reduce(_or, bits, dim=1)
    counts = _popcount(words)
    before = hl.cumsum(counts, dim=0) - counts
    return words, counts, before, hl.reduce(_add, counts, dim=0)


def _expert_of(
    words: torch.Tensor, counts: torch.Tensor, before: torch.Tensor, j: object
) -> torch.Tensor:
    """The j-th distinct expert (ascending): its bitmap word, then the bit in it."""

    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(_or, torch.where(hit, words, 0), dim=0)
    key = hl.reduce(_or, torch.where(hit, before * 8 + hl.arange(8), 0), dim=0)
    upto = _popcount(word & ((2 << hl.arange(32)) - 1))
    bit = hl.reduce(_add, (upto <= j - (key >> 3)).to(torch.int32), dim=0)
    return (key & 7) * 32 + bit


def _scale_tile(chunks: torch.Tensor) -> torch.Tensor:
    """[2, 16, 256] SWIZZLE_32_4_4 bytes of two 128-row tiles -> [256 rows, 32]."""

    return chunks.reshape(2, 8, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(256, 32)


def _logical_rows(acc: torch.Tensor) -> torch.Tensor:
    """[256 physical rows, t] -> logical order, per 128-row FI tile."""

    t = acc.size(1)
    return acc.reshape(2, 4, 4, 8, t).permute(0, 1, 3, 2, 4).reshape(256, t)


def _swiglu(
    acc: torch.Tensor,
    row_scale: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_alpha: torch.Tensor,
) -> torch.Tensor:
    """Interleaved (up, gate) rows [256, t] -> SwiGLU [t, 128]."""

    t = acc.size(1)
    pairs = _logical_rows(acc).reshape(128, 2, t).permute(0, 2, 1)
    up_acc, gate_acc = hl.split(pairs)
    up = up_acc * row_scale[None, :] * up_alpha
    gate = gate_acc * row_scale[None, :] * gate_alpha
    silu = gate / (1.0 + torch.exp(-gate))
    return (silu * up).permute(1, 0)


def _quant_z(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[t, 128] -> (e2m1x2 bytes [t, 64], E4M3 block scale bytes [t, 8])."""

    t = act.size(0)
    blocks = act.reshape(t, 8, 16)
    amax = torch.amax(torch.abs(blocks), dim=-1)
    fp8, mult = encode_block_scale(hl.full([t, 8], 1.0, dtype=torch.float32), amax)
    low, high = hl.split((blocks * mult[:, :, None]).reshape(t, 8, 8, 2))
    zq = pack_e2m1x2(low, high).to(torch.uint8).reshape(t, 64)
    return zq, fp8.to(torch.uint8)


def _muse_moe_tp4_verify_megakernel(
    weights: torch.Tensor,
    weights2: torch.Tensor,
    scales: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    x_row_scale: torch.Tensor,
    packed_routing: torch.Tensor,
    routes: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_alpha: torch.Tensor,
    out_alpha: torch.Tensor,
    z: torch.Tensor,
    z_scale: torch.Tensor,
    gemm2_ws: torch.Tensor,
    sumsq_ws: torch.Tensor,
    norm_weight: torch.Tensor,
    y: torch.Tensor,
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    world: hl.constexpr,
    norm_eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(weights.size())
    hl.specialize(weights.stride())
    hl.specialize(weights2.size())
    hl.specialize(weights2.stride())
    hl.specialize(scales.size())
    hl.specialize(scales.stride())
    hl.specialize(x_q.size())
    hl.specialize(x_q.stride())
    hl.specialize(x_scale.size())
    hl.specialize(x_scale.stride())
    hl.specialize(x_row_scale.size())
    hl.specialize(x_row_scale.stride())
    hl.specialize(packed_routing.size())
    hl.specialize(packed_routing.stride())
    hl.specialize(routes.size())
    hl.specialize(routes.stride())
    hl.specialize(up_alpha.size())
    hl.specialize(up_alpha.stride())
    hl.specialize(gate_alpha.size())
    hl.specialize(gate_alpha.stride())
    hl.specialize(out_alpha.size())
    hl.specialize(out_alpha.stride())
    hl.specialize(z.size())
    hl.specialize(z.stride())
    hl.specialize(z_scale.size())
    hl.specialize(z_scale.stride())
    hl.specialize(gemm2_ws.size())
    hl.specialize(gemm2_ws.stride())
    hl.specialize(sumsq_ws.size())
    hl.specialize(sumsq_ws.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(y.size())
    hl.specialize(y.stride())

    hidden = y.size(1)
    local = gemm2_ws.size(1)
    topk = packed_routing.size(1)
    slots = routes.size(0)
    tokens = packed_routing.size(0)
    rows = x_q.size(0)  # MMA token rows, T padded to 16
    k_steps = weights.size(1)
    q_tiles = z.size(2) * 4 // world // 256
    h_tiles = sumsq_ws.size(1)
    w13_tiles = up_alpha.size(0) * q_tiles
    remote_z = torch.ops.symm_mem.get_remote_tensors(z, group_name)
    remote_z_scale = torch.ops.symm_mem.get_remote_tensors(z_scale, group_name)
    remote_g2 = torch.ops.symm_mem.get_remote_tensors(gemm2_ws, group_name)
    remote_sumsq = torch.ops.symm_mem.get_remote_tensors(sumsq_ws, group_name)
    y3 = y.view(tokens, world, local)
    # x and its scales as one [1, rows, K] slot, laid out like a z slot.
    x3 = x_q.view(1, rows, x_q.size(1))
    xs3 = x_scale.view(1, rows, x_scale.size(1))
    # Activation block scales load as bytes and are viewed as e4m3.
    e4m3 = torch.float8_e4m3fn

    # The previous launch writes the cross-loop state; dependents launch at exit.
    hl.pdl_wait()
    # R1. Stock: the FlashInfer TRT-LLM MoE's routing, GEMM1 (W13, SwiGLU) and
    # its NVFP4 quant of the intermediate.
    for w13_tile_q, w13_tile_j in hl.tile([q_tiles, slots], block_size=[1, 1]):
        words, counts, before, distinct = _distinct_table(routes[:] >> 16)
        # Slots past the distinct count are skipped; their z is never read.
        if w13_tile_j.begin < distinct:
            expert = _expert_of(words, counts, before, w13_tile_j.begin)
            wt = expert * q_tiles + w13_tile_q.begin
            acc = hl.zeros([256, rows], dtype=torch.float32)
            # Staggered K order across tasks spreads FI weight rows over DRAM channels.
            rot = 3 * (w13_tile_j.begin * q_tiles + w13_tile_q.begin)
            for w13_tile_k in hl.tile(k_steps, block_size=1):
                kk = (w13_tile_k.begin + rot) % k_steps
                # Weight bytes stream once: evict_first keeps L2 for the rest.
                lhs = hl.load(
                    weights,
                    [wt, kk, slice(None), slice(None)],
                    eviction_policy="evict_first",
                )
                lhs_scale = _scale_tile(
                    hl.load(
                        scales,
                        [wt, slice(None), kk * 16 + hl.arange(16), slice(None)],
                        eviction_policy="evict_first",
                    )
                )
                rhs = x3[0, :, kk * 256 + hl.arange(256)]
                rhs_scale = xs3[0, :, kk * 32 + hl.arange(32)].view(e4m3)
                acc = hl.dot_scaled(
                    lhs,
                    lhs_scale,
                    "e2m1",
                    rhs.permute(1, 0),
                    rhs_scale,
                    "e2m1",
                    acc=acc,
                    out_dtype=torch.float32,
                )
            act = _swiglu(
                acc,
                x_row_scale[hl.arange(rows)],
                up_alpha[expert],
                gate_alpha[expert],
            )
            zq, zsc = _quant_z(act)
            col = (rank * q_tiles + w13_tile_q.begin) * 64
            scol = (rank * q_tiles + w13_tile_q.begin) * 8
            for peer_z in remote_z:
                peer_z[w13_tile_j, :, col + hl.arange(64)] = zq[None, :, :]
            for peer_zs in remote_z_scale:
                peer_zs[w13_tile_j, :, scol + hl.arange(8)] = zsc[None, :, :]

    # R2. Stock: the FlashInfer TRT-LLM MoE's GEMM2 (W2).
    for w2_tile_h, w2_tile_f in hl.tile([h_tiles, slots], block_size=[1, 1]):
        words2, counts2, before2, distinct2 = _distinct_table(routes[:] >> 16)
        if w2_tile_f.begin < distinct2:
            expert2 = _expert_of(words2, counts2, before2, w2_tile_f.begin)
            wt2 = w13_tiles + expert2 * h_tiles + w2_tile_h.begin
            acc2 = hl.zeros([256, rows], dtype=torch.float32)
            # Same K staggering as R1.
            rot2 = 3 * (w2_tile_f.begin * h_tiles + w2_tile_h.begin)
            for w2_tile_k in hl.tile(k_steps, block_size=1):
                jj = (w2_tile_k.begin + rot2) % k_steps
                lhs2 = hl.load(
                    weights2,
                    [wt2, jj, slice(None), slice(None)],
                    eviction_policy="evict_first",
                )
                lhs2_scale = _scale_tile(
                    hl.load(
                        scales,
                        [wt2, slice(None), jj * 16 + hl.arange(16), slice(None)],
                        eviction_policy="evict_first",
                    )
                )
                rhs2 = z[w2_tile_f.begin, :, jj * 256 + hl.arange(256)]
                rhs2_scale = z_scale[w2_tile_f.begin, :, jj * 32 + hl.arange(32)].view(
                    e4m3
                )
                acc2 = hl.dot_scaled(
                    lhs2,
                    lhs2_scale,
                    "e2m1",
                    rhs2.permute(1, 0),
                    rhs2_scale,
                    "e2m1",
                    acc=acc2,
                    out_dtype=torch.float32,
                )
            val = (
                (_logical_rows(acc2) * out_alpha[expert2])
                .permute(1, 0)
                .to(torch.bfloat16)
            )
            # Each token routes an expert at most once: scatter to its route row.
            # MMA pad rows (T = 8) have no routes, so the scatter drops them.
            slot_id = hl.arange(rows)[:, None] * topk + hl.arange(topk)[None, :]
            live = slot_id < slots
            ids = hl.load(routes, [slot_id], extra_mask=live) >> 16
            match = ((ids == expert2) & live).to(torch.int32)
            route_row = torch.sum(match * slot_id, dim=-1)
            routed = torch.sum(match, dim=-1) > 0
            cols = w2_tile_h.begin * 256 + hl.arange(256)
            hl.store(gemm2_ws, [route_row, cols], val, extra_mask=routed[:, None])
            nw = norm_weight[rank * local + cols].to(torch.float32)
            gm = val.to(torch.float32) * nw[None, :]
            hl.store(
                sumsq_ws,
                [route_row, w2_tile_h.begin],
                torch.sum(gm * gm, dim=-1),
                extra_mask=routed,
            )

    # R3. Stock: the expert-TP all-reduce (reduce_partials), then
    # finalize_with_post_expert_norm (per-route RMSNorm, weighted top-k sum).
    for fin_tile_s, fin_tile_c in hl.tile([slots, local], block_size=[topk, 128]):
        # One token's routes: every rank's sum-of-squares partials and rows.
        bits = (routes[fin_tile_s] & 0xFFFF).to(torch.int16)
        weight = bits.view(torch.bfloat16).to(torch.float32)
        # Norm weights load before the first polled read, overlapping the wait.
        nws = [
            norm_weight[r * local + fin_tile_c.index] for r in hl.static_range(world)
        ]
        ss = hl.zeros([fin_tile_s], dtype=torch.float32)
        for peer_sumsq in remote_sumsq:
            ss = ss + torch.sum(peer_sumsq[fin_tile_s, :], dim=1)
        scale = torch.rsqrt(ss / hidden + norm_eps) * weight
        parts = [peer_g2[fin_tile_s, fin_tile_c] for peer_g2 in remote_g2]
        for r in hl.static_range(world):
            nw_r = nws[r].to(torch.float32)
            y3[fin_tile_s.id, r, fin_tile_c] = torch.sum(
                parts[r].to(torch.float32) * nw_r[None, :] * scale[:, None], dim=0
            ).to(torch.bfloat16)
    hl.pdl_launch_dependents()
    return y


def _muse_moe_tail_verify_megakernel(
    y: torch.Tensor,  # [T, K] bf16, replicated expert output
    wu: torch.Tensor,  # [N / W / C, C, K] bf16, this rank's unpooler rows
    stream: torch.Tensor,  # [T, W, N / W / C, C] fp32
    alpha: torch.Tensor,  # [W, N / W / C, C] fp32
    beta: torch.Tensor,  # [W, N / W / C, C] fp32
    post_w: torch.Tensor,  # [W, N / W / C, C] bf16
    pre_w: torch.Tensor,  # [W, N / W / C, C] bf16
    rows: torch.Tensor,  # [T, N / W / C, C] bf16 symm
    moments: torch.Tensor,  # [N / W / C, T, 4] fp32 symm
    new_stream: torch.Tensor,  # [T, W, N / W / C, C] fp32
    attn_in: torch.Tensor,  # [T, W, N / W / C, C] bf16
    wq: torch.Tensor,  # [Nq / CQ, CQ, N] bf16, the next layer's QKV rows
    qkv: torch.Tensor,  # [T, Nq / CQ, CQ] bf16
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    post_eps: hl.constexpr,
    pre_eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(y.size())
    hl.specialize(y.stride())
    hl.specialize(wu.size())
    hl.specialize(wu.stride())
    hl.specialize(stream.size())
    hl.specialize(stream.stride())
    hl.specialize(alpha.size())
    hl.specialize(alpha.stride())
    hl.specialize(beta.size())
    hl.specialize(beta.stride())
    hl.specialize(post_w.size())
    hl.specialize(post_w.stride())
    hl.specialize(pre_w.size())
    hl.specialize(pre_w.stride())
    hl.specialize(rows.size())
    hl.specialize(rows.stride())
    hl.specialize(moments.size())
    hl.specialize(moments.stride())
    hl.specialize(new_stream.size())
    hl.specialize(new_stream.stride())
    hl.specialize(attn_in.size())
    hl.specialize(attn_in.stride())
    hl.specialize(wq.size())
    hl.specialize(wq.stride())
    hl.specialize(qkv.size())
    hl.specialize(qkv.stride())

    tokens = y.size(0)
    k_u = y.size(1)
    n_blocks = wu.size(0)
    c_block = wu.size(1)
    world = stream.size(1)
    hidden = world * n_blocks * c_block
    k_blk = hl.register_block_size(k_u)
    q_blk = hl.register_block_size(hidden)
    x = attn_in.view(tokens, hidden)
    remote_rows = torch.ops.symm_mem.get_remote_tensors(rows, group_name)
    remote_moments = torch.ops.symm_mem.get_remote_tensors(moments, group_name)

    hl.pdl_wait()
    # Trigger at entry: the next kernel's CTAs land on SMs this grid frees early,
    # and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # Stock: the unpooler GEMM (post_expert_proj via gemm_local, this rank's rows).
    for u_b in hl.tile(n_blocks, block_size=1):
        acc = hl.zeros([c_block, tokens], dtype=torch.float32)
        for u_k in hl.tile(k_u, block_size=k_blk):
            acc = hl.dot(
                hl.load(wu, [u_b.begin, slice(None), u_k]),
                y[:, u_k].permute(1, 0),
                acc=acc,
            )
        post = post_w[rank, u_b.begin, :].to(torch.float32)
        gated_stream = (
            alpha[rank, u_b.begin, :][None, :] * stream[:, rank, u_b.begin, :]
        )
        beta_c = beta[rank, u_b.begin, :]
        unpooled = acc.permute(1, 0).to(torch.bfloat16)
        rows[:, u_b.begin, :] = unpooled
        # MuseSpark1xRMSNormWithInputScale multiplies in BF16.
        scaled = (unpooled.to(torch.float32) * post[None, :]).to(torch.bfloat16)
        scaled = scaled.to(torch.float32)
        gated_branch = beta_c[None, :] * scaled
        m = hl.arange(4)[None, :]
        moments[u_b.begin, :, :] = (
            torch.where(m == 0, torch.sum(scaled * scaled, dim=-1)[:, None], 0.0)
            + torch.where(
                m == 1, torch.sum(gated_stream * gated_stream, dim=-1)[:, None], 0.0
            )
            + torch.where(
                m == 2, torch.sum(gated_stream * gated_branch, dim=-1)[:, None], 0.0
            )
            + torch.where(
                m == 3, torch.sum(gated_branch * gated_branch, dim=-1)[:, None], 0.0
            )
        )

    # Stock: FusedPostNormResidualPreNorm.forward_with_ag (unpooler all-gather,
    # post-FFN norm, gated residual, next layer's pre-attention norm).
    for n_t, n_b in hl.tile([tokens, n_blocks], block_size=[1, N_BLOCKS]):
        owner = hl.arange(world)[:, None, None]
        post = post_w[:, n_b, :].to(torch.float32)
        gated_stream = alpha[:, n_b, :] * stream[n_t.begin, :, n_b, :]
        beta_c = beta[:, n_b, :]
        pre = pre_w[:, n_b, :].to(torch.float32)
        total = hl.zeros([4], dtype=torch.float32)
        # Rank order, so every rank and tile derives the same scalars.
        for peer_moments in remote_moments:
            total = total + torch.sum(peer_moments[:, n_t.begin, :], dim=0)
        branch = hl.zeros([world, n_b, c_block], dtype=torch.float32)
        for peer, peer_rows in enumerate(remote_rows):
            branch = torch.where(
                owner == peer,
                peer_rows[n_t.begin, n_b, :].to(torch.float32)[None, :, :],
                branch,
            )
        m = hl.arange(4)
        post_sq = torch.sum(torch.where(m == 0, total, 0.0))
        stream_sq = torch.sum(torch.where(m == 1, total, 0.0))
        cross = torch.sum(torch.where(m == 2, total, 0.0))
        branch_sq = torch.sum(torch.where(m == 3, total, 0.0))
        post_rstd = torch.rsqrt(post_sq / hidden + post_eps)
        pre_var = (
            stream_sq + 2.0 * post_rstd * cross + post_rstd * post_rstd * branch_sq
        ) / hidden
        pre_rstd = torch.rsqrt(pre_var + pre_eps)
        scaled = (branch * post).to(torch.bfloat16).to(torch.float32)
        normed = (scaled * post_rstd).to(torch.bfloat16).to(torch.float32)
        updated = gated_stream + beta_c * normed
        new_stream[n_t.begin, :, n_b, :] = updated
        attn_in[n_t.begin, :, n_b, :] = (updated * pre_rstd * pre).to(torch.bfloat16)

    # Stock: the next layer's QKVG GEMM (qkv_proj also holds the output-gate rows).
    for q_b in hl.tile(wq.size(0), block_size=1):
        acc = hl.zeros([wq.size(1), tokens], dtype=torch.float32)
        for q_k in hl.tile(hidden, block_size=q_blk):
            acc = hl.dot(wq[q_b.begin, :, q_k], x[:, q_k].permute(1, 0), acc=acc)
        qkv[:, q_b.begin, :] = acc.permute(1, 0).to(torch.bfloat16)
    return attn_in


def make_config(
    *,
    num_warps: int = 4,
    num_stages: int = 2,
    pipeline: str = "static",
    pid_type: str = "persistent_blocked",
    **extra: object,
) -> helion.Config:
    # R1/R2 K loops are warp-specialized 3-stage TMA rings. Activation scales, R1's
    # row scale and alphas, and the gemm2 scatter are small: pointers beat TMA there.
    indexing = ["tensor_descriptor"] * 43
    for slot in (4, 5, 6, 7, 20, 23):
        indexing[slot] = "pointer"
    return helion.Config(
        block_sizes=[],
        cross_loop_pipeline=pipeline,
        host_tensor_descriptors=True,
        indexing=indexing,
        num_sm_multiplier=1,
        num_stages=num_stages,
        num_warps=num_warps,
        pid_type=pid_type,
        range_num_stages=[0, 3, 0, 3, 0],
        range_warp_specializes=[True, None, True, None, None],
        **extra,
    )


def make_tail_config(
    *,
    k_blk: int = 512,
    q_blk: int = 512,
    num_warps: int = 2,
    num_stages: int = 5,
    pipeline: str = "dynamic",
    pid_type: str = "persistent_blocked",
    **extra: object,
) -> helion.Config:
    return helion.Config(
        block_sizes=[k_blk, q_blk],
        cross_loop_pipeline=pipeline,
        indexing="pointer",
        num_stages=num_stages,
        num_warps=num_warps,
        pid_type=pid_type,
        **extra,
    )


def _build(fn: object, config: helion.Config) -> object:
    return helion.kernel(
        fn,
        config=config,
        static_shapes=True,
        backend="triton",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def build_kernel(config: helion.Config) -> object:
    return _build(_muse_moe_tp4_verify_megakernel, config)


class MoeTp4VerifyBank(msgspec.Struct, frozen=True):
    """One rank's weight bank and the stock FI W13 views that alias its head."""

    weights: torch.Tensor  # [W13 tiles, K / 512, 256, 256] u8 view of [rows, K / 2]
    weights2: torch.Tensor  # [tiles, K / 512, 256, 256] u8; R2 reads its W2 tiles
    scales: torch.Tensor  # [tiles, 2, K / 512 * 16, 256] e4m3
    w13: torch.Tensor  # [E, 2I / W, H / 2] u8
    w13_scale: torch.Tensor  # FI-shuffled W13 block scales, flat e4m3


def build_moe_tp4_verify_bank(
    *,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group: object,
) -> MoeTp4VerifyBank:
    """Copy this rank's FI W13 into a new bank and append its H/W quarter of W2
    over the full K as contiguous tiles, all-to-all'd from every rank's K slice."""

    world = group.size()
    experts, two_local, packed_k = w13.shape
    hidden, packed_local = w2.size(1), w2.size(2)
    quarter = hidden // world
    w13_rows, w2_rows = experts * two_local, experts * quarter
    device = w13.device
    bank = torch.empty(w13_rows + w2_rows, packed_k, dtype=torch.uint8, device=device)
    bank[:w13_rows].copy_(w13.view(torch.uint8).reshape(w13_rows, packed_k))
    send = (
        w2.view(torch.uint8)
        .view(experts, world, quarter, packed_local)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # W2 tiles are [K steps, 256, 256 B] boxes: one 64 KiB DRAM run per TMA load.
    k_steps = packed_k // TILE_K
    w2_tiles = w2_rows // TILE_M
    rows2 = recv.permute(1, 2, 0, 3).reshape(w2_tiles, TILE_M, k_steps, TILE_K)
    bank[w13_rows:].view(w2_tiles, k_steps, TILE_M, TILE_K).copy_(
        rows2.permute(0, 2, 1, 3)
    )
    del send, recv, rows2

    # Block scales are contiguous [K steps, 32 * 128] chunks per 128-row tile.
    sf_bytes = packed_k // 8
    sbank = torch.empty(
        (w13_rows + w2_rows) * sf_bytes, dtype=torch.uint8, device=device
    )
    sbank[: w13_rows * sf_bytes].copy_(w13_scale.view(torch.uint8).reshape(-1))
    tile_bytes = MMA_M * packed_local // 8
    ssend = (
        w2_scale.view(torch.uint8)
        .reshape(experts, world, quarter // MMA_M, tile_bytes)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    srecv = torch.empty_like(ssend)
    dist.all_to_all_single(srecv, ssend, group=group)
    sbank[w13_rows * sf_bytes :].view(
        experts, quarter // MMA_M, world, tile_bytes
    ).copy_(srecv.permute(1, 2, 0, 3))

    tiles = (w13_rows + w2_rows) // TILE_M
    w13_tiles = w13_rows // TILE_M
    e4m3 = torch.float8_e4m3fn
    return MoeTp4VerifyBank(
        weights=bank[:w13_rows]
        .view(w13_tiles, TILE_M, k_steps, TILE_K)
        .permute(0, 2, 1, 3),
        weights2=bank.view(tiles, k_steps, TILE_M, TILE_K),
        scales=sbank.view(e4m3).view(tiles, 2, k_steps * 16, TILE_M),
        w13=bank[:w13_rows].view(experts, two_local, packed_k),
        w13_scale=sbank[: w13_rows * sf_bytes].view(e4m3),
    )


class MoeVerifyInputs(msgspec.Struct, frozen=True):
    """The MoE megakernel's NVFP4 input slot: 16 activation rows, T route rows."""

    fp4: torch.Tensor  # [16, H / 2] u8
    scales: torch.Tensor  # [16, H / 16] u8 (E4M3 bits)
    per_token_scale: torch.Tensor  # [16] fp32
    packed: torch.Tensor  # [T, 8] int32 (expert << 16 | bf16 weight bits)


class MuseMoeTp4VerifyMegakernel:
    """Per-rank symmetric buffers and call for the verify expert-TP MoE.

    The call takes the deferred FI MoE inputs and this rank's bank, and returns
    the post-expert-normed route sum y [T, H] (replicated on every rank). The
    input slot's activations hold 16 rows (the first T live, the rest zero) and
    its routes T rows.
    """

    def __init__(
        self,
        device: torch.device,
        group: object,
        config: helion.Config | None = None,
        *,
        tokens: int,
        intermediate: int = 4096,
        hidden: int = 4096,
        norm_eps: float = 1e-8,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.kernel = build_kernel(config or make_config())
        self.group_name = group.group_name
        self.rank = group.rank()
        self.world = group.size()
        self.tokens = tokens
        self.hidden = hidden
        self.norm_eps = norm_eps
        local = hidden // self.world
        slots = tokens * TOPK
        # [slot j, token, I / 2] routed NVFP4 activation of the j-th expert, with
        # its E4M3 block scale bytes.
        self.z = symm_mem.empty(
            slots, MOE_TOKENS, intermediate // 2, dtype=torch.uint8, device=device
        )
        self.z_scale = symm_mem.empty(
            slots, MOE_TOKENS, intermediate // 16, dtype=torch.uint8, device=device
        )
        # [route, H / W] BF16 rows and per-256-column sums of (norm_w * x)^2.
        self.gemm2_ws = symm_mem.empty(
            slots, local, dtype=torch.bfloat16, device=device
        )
        self.sumsq_ws = symm_mem.empty(
            slots, local // TILE_M, dtype=torch.float32, device=device
        )
        for buffer in (self.z, self.z_scale, self.gemm2_ws, self.sumsq_ws):
            symm_mem.rendezvous(buffer, self.group_name)
        # Producers write the first T activation rows; the pad rows stay zero.
        self.inputs = MoeVerifyInputs(
            fp4=torch.zeros(MOE_TOKENS, hidden // 2, dtype=torch.uint8, device=device),
            scales=torch.zeros(
                MOE_TOKENS, hidden // 16, dtype=torch.uint8, device=device
            ),
            per_token_scale=torch.zeros(MOE_TOKENS, dtype=torch.float32, device=device),
            packed=torch.zeros(tokens, TOPK, dtype=torch.int32, device=device),
        )

    def __call__(
        self,
        *,
        bank: MoeTp4VerifyBank,
        packed_routing: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        per_token_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.empty(
            self.tokens, self.hidden, dtype=torch.bfloat16, device=hidden_states.device
        )
        self.kernel(
            bank.weights,
            bank.weights2,
            bank.scales,
            hidden_states.view(torch.uint8),
            hidden_states_scale.view(torch.uint8).reshape(MOE_TOKENS, -1),
            per_token_scale,
            packed_routing,
            packed_routing.view(-1),
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            self.z,
            self.z_scale,
            self.gemm2_ws,
            self.sumsq_ws,
            norm_weight,
            y,
            self.group_name,
            self.rank,
            self.world,
            self.norm_eps,
        )
        return y


class MuseMoeTailVerifyMegakernel:
    """Per-rank symmetric buffers and call for the verify MoE tail.

    The call takes the first T rows of y, this rank's unpooler rows, the FFN
    boundary's [hidden] weights and optionally the next layer's QKV weight, and
    returns the new
    stream, the next attention input and its QKV (or None), replicated on every
    rank.
    """

    def __init__(
        self,
        device: torch.device,
        group: object,
        *,
        tokens: int,
        post_eps: float,
        pre_eps: float,
        config: helion.Config | None = None,
        hidden: int = 8192,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.kernel = _build(
            _muse_moe_tail_verify_megakernel, config or make_tail_config()
        )
        self.group_name = group.group_name
        self.world = group.size()
        self.tokens = tokens
        self.hidden = hidden
        n_blocks = hidden // self.world // C_BLOCK
        self.rows = symm_mem.empty(
            tokens, n_blocks, C_BLOCK, dtype=torch.bfloat16, device=device
        )
        self.moments = symm_mem.empty(
            n_blocks, tokens, 4, dtype=torch.float32, device=device
        )
        for buffer in (self.rows, self.moments):
            symm_mem.rendezvous(buffer, self.group_name)
        # Without a next layer, the QKV root runs on one throwaway block.
        self.no_qkv = torch.zeros(
            1, QKV_ROWS, hidden, dtype=torch.bfloat16, device=device
        )
        self.constants = (group.rank(), post_eps, pre_eps)

    def __call__(
        self,
        *,
        expert_output: torch.Tensor,
        unpool_weight: torch.Tensor,
        stream: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor,
        pre_weight: torch.Tensor,
        next_qkv_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        world, tokens = self.world, self.tokens
        device = stream.device
        new_stream = torch.empty_like(stream)
        attn_in = torch.empty(stream.shape, dtype=torch.bfloat16, device=device)
        wq = (
            self.no_qkv
            if next_qkv_weight is None
            else next_qkv_weight.view(-1, QKV_ROWS, self.hidden)
        )
        qkv = torch.empty(tokens, *wq.shape[:2], dtype=torch.bfloat16, device=device)
        self.kernel(
            expert_output,
            unpool_weight.view(-1, C_BLOCK, unpool_weight.size(1)),
            stream.view(tokens, world, -1, C_BLOCK),
            alpha.view(world, -1, C_BLOCK),
            beta.view(world, -1, C_BLOCK),
            post_weight.view(world, -1, C_BLOCK),
            pre_weight.view(world, -1, C_BLOCK),
            self.rows,
            self.moments,
            new_stream.view(tokens, world, -1, C_BLOCK),
            attn_in.view(tokens, world, -1, C_BLOCK),
            wq,
            qkv,
            self.group_name,
            *self.constants,
        )
        next_qkv = None if next_qkv_weight is None else qkv.view(tokens, -1)
        return new_stream, attn_in, next_qkv


_INSTANCES: dict[tuple[int, object, float], MuseMoeTp4VerifyMegakernel] = {}


def get_moe_tp4_verify_megakernel(
    *, device: torch.device, group: object, norm_eps: float
) -> MuseMoeTp4VerifyMegakernel:
    """Build once per (device, group, eps); the constructor rendezvouses
    symmetric buffers, so every rank must reach a new key together, outside
    graph capture."""

    key = (device.index, group, norm_eps)
    if key not in _INSTANCES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("build the Helion verify MoE megakernel before capture")
        _INSTANCES[key] = MuseMoeTp4VerifyMegakernel(
            device, group, tokens=MOE_TOKENS, norm_eps=norm_eps
        )
    return _INSTANCES[key]


__all__ = [
    "MOE_TOKENS",
    "MoeTp4VerifyBank",
    "MoeVerifyInputs",
    "MuseMoeTailVerifyMegakernel",
    "MuseMoeTp4VerifyMegakernel",
    "build_moe_tp4_verify_bank",
    "get_moe_tp4_verify_megakernel",
    "make_config",
    "make_tail_config",
]
