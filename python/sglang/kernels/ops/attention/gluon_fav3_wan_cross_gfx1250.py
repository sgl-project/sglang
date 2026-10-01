# SPDX-License-Identifier: MIT
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright (c) 2026 LightSeek Foundation

"""Exact Wan2.2 cross-attention specialization for AMD gfx1250.

This module contains only the measured ``B2/Sq176400/Sk512/H40/D128`` BF16
BSHD path. The kernel is derived from Triton's CDNA5 FlashAttention example:
https://github.com/triton-lang/triton/blob/b63e34c521dfc55a5f9e419cfeaa64df8263891f/third_party/amd/python/examples/gluon/f16_fa_cdna5.py
"""

from __future__ import annotations

import math

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import _AttentionConfig


@gluon.jit
def _join_columns(left, right, layout: gl.constexpr):
    joined = (
        gl.join(left, right)
        .permute(0, 2, 1)
        .reshape([left.shape[0], left.shape[1] + right.shape[1]])
    )
    return gl.convert_layout(joined, layout, assert_trivial=True)


@gluon.jit
def _join_rows(top, bottom, layout: gl.constexpr):
    joined = (
        gl.join(top, bottom)
        .permute(2, 0, 1)
        .reshape([top.shape[0] + bottom.shape[0], top.shape[1]])
    )
    return gl.convert_layout(joined, layout, assert_trivial=True)


@gluon.jit
def _qk_step(
    cfg,
    query,
    key_shared,
    scores0,
    scores1,
    n_fragment: gl.constexpr,
    k_fragment: gl.constexpr,
):
    query0 = gl.amd.slice(query, [64, 32], [0, k_fragment * 32])
    query1 = gl.amd.slice(query, [64, 32], [64, k_fragment * 32])
    key = (
        key_shared.slice(n_fragment * 16, 16, dim=0)
        .slice(k_fragment * 32, 32, dim=1)
        .permute([1, 0])
        .load(layout=cfg.k_layout)
    )
    return (
        gl.amd.cdna5.wmma(query0, key, scores0),
        gl.amd.cdna5.wmma(query1, key, scores1),
    )


@gluon.jit
def _qk(
    cfg,
    query,
    key_shared,
    macro_start,
    n_fragment: gl.constexpr,
):
    scores0 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
    scores1 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
    scores0, scores1 = _qk_step(cfg, query, key_shared, scores0, scores1, n_fragment, 0)
    scores0, scores1 = _qk_step(cfg, query, key_shared, scores0, scores1, n_fragment, 1)
    scores0, scores1 = _qk_step(cfg, query, key_shared, scores0, scores1, n_fragment, 2)
    scores0, scores1 = _qk_step(cfg, query, key_shared, scores0, scores1, n_fragment, 3)
    key_cols = (
        macro_start
        + n_fragment * 16
        + gl.arange(0, 16, layout=gl.SliceLayout(0, cfg.qk_layout))
    )
    valid = key_cols[None, :] < cfg.SEQLEN_K
    scores0 = gl.where(valid, scores0, float("-inf"))
    scores1 = gl.where(valid, scores1, float("-inf"))
    return scores0, scores1


@gluon.jit
def _qk_grouped(
    cfg,
    query,
    key_shared,
    macro_start,
    n_base: gl.constexpr,
):
    with gl.amd.warp_pipeline_stage("grouped_qk_load", priority=1):
        q0k0 = gl.amd.slice(query, [64, 32], [0, 0])
        q1k0 = gl.amd.slice(query, [64, 32], [64, 0])
        q0k1 = gl.amd.slice(query, [64, 32], [0, 32])
        q1k1 = gl.amd.slice(query, [64, 32], [64, 32])
        q0k2 = gl.amd.slice(query, [64, 32], [0, 64])
        q1k2 = gl.amd.slice(query, [64, 32], [64, 64])
        q0k3 = gl.amd.slice(query, [64, 32], [0, 96])
        q1k3 = gl.amd.slice(query, [64, 32], [64, 96])

        key_n0k0 = (
            key_shared.slice((n_base + 0) * 16, 16, dim=0)
            .slice(0, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n1k0 = (
            key_shared.slice((n_base + 1) * 16, 16, dim=0)
            .slice(0, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n2k0 = (
            key_shared.slice((n_base + 2) * 16, 16, dim=0)
            .slice(0, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n3k0 = (
            key_shared.slice((n_base + 3) * 16, 16, dim=0)
            .slice(0, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n0k1 = (
            key_shared.slice((n_base + 0) * 16, 16, dim=0)
            .slice(32, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n1k1 = (
            key_shared.slice((n_base + 1) * 16, 16, dim=0)
            .slice(32, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n2k1 = (
            key_shared.slice((n_base + 2) * 16, 16, dim=0)
            .slice(32, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n3k1 = (
            key_shared.slice((n_base + 3) * 16, 16, dim=0)
            .slice(32, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n0k2 = (
            key_shared.slice((n_base + 0) * 16, 16, dim=0)
            .slice(64, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n1k2 = (
            key_shared.slice((n_base + 1) * 16, 16, dim=0)
            .slice(64, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n2k2 = (
            key_shared.slice((n_base + 2) * 16, 16, dim=0)
            .slice(64, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n3k2 = (
            key_shared.slice((n_base + 3) * 16, 16, dim=0)
            .slice(64, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n0k3 = (
            key_shared.slice((n_base + 0) * 16, 16, dim=0)
            .slice(96, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n1k3 = (
            key_shared.slice((n_base + 1) * 16, 16, dim=0)
            .slice(96, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n2k3 = (
            key_shared.slice((n_base + 2) * 16, 16, dim=0)
            .slice(96, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )
        key_n3k3 = (
            key_shared.slice((n_base + 3) * 16, 16, dim=0)
            .slice(96, 32, dim=1)
            .permute([1, 0])
            .load(layout=cfg.k_layout)
        )

    with gl.amd.warp_pipeline_stage("grouped_qk_wmma", priority=0):
        score00 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score01 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score02 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score03 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score10 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score11 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score12 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)
        score13 = gl.zeros([64, 16], gl.float32, cfg.qk_layout)

        score00 = gl.amd.cdna5.wmma(q0k0, key_n0k0, score00)
        score10 = gl.amd.cdna5.wmma(q1k0, key_n0k0, score10)
        score01 = gl.amd.cdna5.wmma(q0k0, key_n1k0, score01)
        score11 = gl.amd.cdna5.wmma(q1k0, key_n1k0, score11)
        score02 = gl.amd.cdna5.wmma(q0k0, key_n2k0, score02)
        score12 = gl.amd.cdna5.wmma(q1k0, key_n2k0, score12)
        score03 = gl.amd.cdna5.wmma(q0k0, key_n3k0, score03)
        score13 = gl.amd.cdna5.wmma(q1k0, key_n3k0, score13)

        score00 = gl.amd.cdna5.wmma(q0k1, key_n0k1, score00)
        score10 = gl.amd.cdna5.wmma(q1k1, key_n0k1, score10)
        score01 = gl.amd.cdna5.wmma(q0k1, key_n1k1, score01)
        score11 = gl.amd.cdna5.wmma(q1k1, key_n1k1, score11)
        score02 = gl.amd.cdna5.wmma(q0k1, key_n2k1, score02)
        score12 = gl.amd.cdna5.wmma(q1k1, key_n2k1, score12)
        score03 = gl.amd.cdna5.wmma(q0k1, key_n3k1, score03)
        score13 = gl.amd.cdna5.wmma(q1k1, key_n3k1, score13)

        score00 = gl.amd.cdna5.wmma(q0k2, key_n0k2, score00)
        score10 = gl.amd.cdna5.wmma(q1k2, key_n0k2, score10)
        score01 = gl.amd.cdna5.wmma(q0k2, key_n1k2, score01)
        score11 = gl.amd.cdna5.wmma(q1k2, key_n1k2, score11)
        score02 = gl.amd.cdna5.wmma(q0k2, key_n2k2, score02)
        score12 = gl.amd.cdna5.wmma(q1k2, key_n2k2, score12)
        score03 = gl.amd.cdna5.wmma(q0k2, key_n3k2, score03)
        score13 = gl.amd.cdna5.wmma(q1k2, key_n3k2, score13)

        score00 = gl.amd.cdna5.wmma(q0k3, key_n0k3, score00)
        score10 = gl.amd.cdna5.wmma(q1k3, key_n0k3, score10)
        score01 = gl.amd.cdna5.wmma(q0k3, key_n1k3, score01)
        score11 = gl.amd.cdna5.wmma(q1k3, key_n1k3, score11)
        score02 = gl.amd.cdna5.wmma(q0k3, key_n2k3, score02)
        score12 = gl.amd.cdna5.wmma(q1k3, key_n2k3, score12)
        score03 = gl.amd.cdna5.wmma(q0k3, key_n3k3, score03)
        score13 = gl.amd.cdna5.wmma(q1k3, key_n3k3, score13)

    key_offsets = gl.arange(0, 16, layout=gl.SliceLayout(0, cfg.qk_layout))
    valid0 = (macro_start + (n_base + 0) * 16 + key_offsets)[None, :] < cfg.SEQLEN_K
    valid1 = (macro_start + (n_base + 1) * 16 + key_offsets)[None, :] < cfg.SEQLEN_K
    valid2 = (macro_start + (n_base + 2) * 16 + key_offsets)[None, :] < cfg.SEQLEN_K
    valid3 = (macro_start + (n_base + 3) * 16 + key_offsets)[None, :] < cfg.SEQLEN_K
    score00 = gl.where(valid0, score00, float("-inf"))
    score10 = gl.where(valid0, score10, float("-inf"))
    score01 = gl.where(valid1, score01, float("-inf"))
    score11 = gl.where(valid1, score11, float("-inf"))
    score02 = gl.where(valid2, score02, float("-inf"))
    score12 = gl.where(valid2, score12, float("-inf"))
    score03 = gl.where(valid3, score03, float("-inf"))
    score13 = gl.where(valid3, score13, float("-inf"))
    return (
        (score00, score01, score02, score03),
        (score10, score11, score12, score13),
    )


@gluon.jit
def _qk_all_grouped(cfg, query, key_shared, macro_start):
    group0 = _qk_grouped(cfg, query, key_shared, macro_start, 0)
    group1 = _qk_grouped(cfg, query, key_shared, macro_start, 4)
    group2 = _qk_grouped(cfg, query, key_shared, macro_start, 8)
    group3 = _qk_grouped(cfg, query, key_shared, macro_start, 12)
    return (
        (
            group0[0][0],
            group0[0][1],
            group0[0][2],
            group0[0][3],
            group1[0][0],
            group1[0][1],
            group1[0][2],
            group1[0][3],
            group2[0][0],
            group2[0][1],
            group2[0][2],
            group2[0][3],
            group3[0][0],
            group3[0][1],
            group3[0][2],
            group3[0][3],
        ),
        (
            group0[1][0],
            group0[1][1],
            group0[1][2],
            group0[1][3],
            group1[1][0],
            group1[1][1],
            group1[1][2],
            group1[1][3],
            group2[1][0],
            group2[1][1],
            group2[1][2],
            group2[1][3],
            group3[1][0],
            group3[1][1],
            group3[1][2],
            group3[1][3],
        ),
    )


@gluon.jit
def _max16(scores):
    result = gl.max(scores[0], 1)
    result = gl.maximum(result, gl.max(scores[1], 1))
    result = gl.maximum(result, gl.max(scores[2], 1))
    result = gl.maximum(result, gl.max(scores[3], 1))
    result = gl.maximum(result, gl.max(scores[4], 1))
    result = gl.maximum(result, gl.max(scores[5], 1))
    result = gl.maximum(result, gl.max(scores[6], 1))
    result = gl.maximum(result, gl.max(scores[7], 1))
    result = gl.maximum(result, gl.max(scores[8], 1))
    result = gl.maximum(result, gl.max(scores[9], 1))
    result = gl.maximum(result, gl.max(scores[10], 1))
    result = gl.maximum(result, gl.max(scores[11], 1))
    result = gl.maximum(result, gl.max(scores[12], 1))
    result = gl.maximum(result, gl.max(scores[13], 1))
    result = gl.maximum(result, gl.max(scores[14], 1))
    return gl.maximum(result, gl.max(scores[15], 1))


@gluon.jit
def _probability_fragment(
    score,
    scale_log2: gl.constexpr,
    row_max_scaled,
):
    probability = gl.exp2(score * scale_log2 - row_max_scaled[:, None])
    return (
        probability.to(gl.bfloat16, fp_downcast_rounding="rtz"),
        gl.sum(probability, 1),
    )


@gluon.jit
def _rescale8(accumulators, alpha):
    return (
        accumulators[0] * alpha[:, None],
        accumulators[1] * alpha[:, None],
        accumulators[2] * alpha[:, None],
        accumulators[3] * alpha[:, None],
        accumulators[4] * alpha[:, None],
        accumulators[5] * alpha[:, None],
        accumulators[6] * alpha[:, None],
        accumulators[7] * alpha[:, None],
    )


@gluon.jit
def _qk_probability_fragment(
    cfg,
    query,
    key_shared,
    macro_start,
    n_fragment: gl.constexpr,
    previous_score0,
    previous_score1,
    scale_log2: gl.constexpr,
    previous_scaled0,
    previous_scaled1,
):
    score0, score1 = _qk(cfg, query, key_shared, macro_start, n_fragment)
    probability0, row_sum0 = _probability_fragment(
        previous_score0, scale_log2, previous_scaled0
    )
    probability1, row_sum1 = _probability_fragment(
        previous_score1, scale_log2, previous_scaled1
    )
    return score0, score1, probability0, probability1, row_sum0, row_sum1


@gluon.jit
def _qk_all_with_probabilities(
    cfg,
    query,
    key_shared,
    macro_start,
    previous_scores0,
    previous_scores1,
    scale_log2: gl.constexpr,
    previous_scaled0,
    previous_scaled1,
):
    r0 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        0,
        previous_scores0[0],
        previous_scores1[0],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r1 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        1,
        previous_scores0[1],
        previous_scores1[1],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r2 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        2,
        previous_scores0[2],
        previous_scores1[2],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r3 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        3,
        previous_scores0[3],
        previous_scores1[3],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r4 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        4,
        previous_scores0[4],
        previous_scores1[4],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r5 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        5,
        previous_scores0[5],
        previous_scores1[5],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r6 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        6,
        previous_scores0[6],
        previous_scores1[6],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r7 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        7,
        previous_scores0[7],
        previous_scores1[7],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r8 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        8,
        previous_scores0[8],
        previous_scores1[8],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    r9 = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        9,
        previous_scores0[9],
        previous_scores1[9],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    ra = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        10,
        previous_scores0[10],
        previous_scores1[10],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    rb = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        11,
        previous_scores0[11],
        previous_scores1[11],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    rc = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        12,
        previous_scores0[12],
        previous_scores1[12],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    rd = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        13,
        previous_scores0[13],
        previous_scores1[13],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    re = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        14,
        previous_scores0[14],
        previous_scores1[14],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    rf = _qk_probability_fragment(
        cfg,
        query,
        key_shared,
        macro_start,
        15,
        previous_scores0[15],
        previous_scores1[15],
        scale_log2,
        previous_scaled0,
        previous_scaled1,
    )
    scores0 = (
        r0[0],
        r1[0],
        r2[0],
        r3[0],
        r4[0],
        r5[0],
        r6[0],
        r7[0],
        r8[0],
        r9[0],
        ra[0],
        rb[0],
        rc[0],
        rd[0],
        re[0],
        rf[0],
    )
    scores1 = (
        r0[1],
        r1[1],
        r2[1],
        r3[1],
        r4[1],
        r5[1],
        r6[1],
        r7[1],
        r8[1],
        r9[1],
        ra[1],
        rb[1],
        rc[1],
        rd[1],
        re[1],
        rf[1],
    )
    probabilities0 = (
        r0[2],
        r1[2],
        r2[2],
        r3[2],
        r4[2],
        r5[2],
        r6[2],
        r7[2],
        r8[2],
        r9[2],
        ra[2],
        rb[2],
        rc[2],
        rd[2],
        re[2],
        rf[2],
    )
    probabilities1 = (
        r0[3],
        r1[3],
        r2[3],
        r3[3],
        r4[3],
        r5[3],
        r6[3],
        r7[3],
        r8[3],
        r9[3],
        ra[3],
        rb[3],
        rc[3],
        rd[3],
        re[3],
        rf[3],
    )
    row_sum0 = (
        r0[4]
        + r1[4]
        + r2[4]
        + r3[4]
        + r4[4]
        + r5[4]
        + r6[4]
        + r7[4]
        + r8[4]
        + r9[4]
        + ra[4]
        + rb[4]
        + rc[4]
        + rd[4]
        + re[4]
        + rf[4]
    )
    row_sum1 = (
        r0[5]
        + r1[5]
        + r2[5]
        + r3[5]
        + r4[5]
        + r5[5]
        + r6[5]
        + r7[5]
        + r8[5]
        + r9[5]
        + ra[5]
        + rb[5]
        + rc[5]
        + rd[5]
        + re[5]
        + rf[5]
    )
    return (
        scores0,
        scores1,
        probabilities0,
        probabilities1,
        row_sum0,
        row_sum1,
    )


@gluon.jit
def _pv_d(
    cfg,
    probability0,
    probability1,
    value_shared,
    accumulators,
    n_fragment: gl.constexpr,
    d_fragment: gl.constexpr,
):
    value = (
        value_shared.slice(n_fragment * 32, 32, dim=0)
        .slice(d_fragment * 16, 16, dim=1)
        .load(layout=cfg.v_layout)
    )
    return (
        gl.amd.cdna5.wmma(probability0, value, accumulators[0][d_fragment]),
        gl.amd.cdna5.wmma(probability1, value, accumulators[1][d_fragment]),
    )


@gluon.jit
def _pv_fragment(
    cfg,
    probabilities0,
    probabilities1,
    value_shared,
    accumulators,
    n_fragment: gl.constexpr,
):
    probability_start: gl.constexpr = n_fragment * 2
    probability0 = _join_columns(
        probabilities0[probability_start],
        probabilities0[probability_start + 1],
        cfg.p_layout,
    )
    probability1 = _join_columns(
        probabilities1[probability_start],
        probabilities1[probability_start + 1],
        cfg.p_layout,
    )
    a00, a10 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 0
    )
    a01, a11 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 1
    )
    a02, a12 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 2
    )
    a03, a13 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 3
    )
    a04, a14 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 4
    )
    a05, a15 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 5
    )
    a06, a16 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 6
    )
    a07, a17 = _pv_d(
        cfg, probability0, probability1, value_shared, accumulators, n_fragment, 7
    )
    return (
        (a00, a01, a02, a03, a04, a05, a06, a07),
        (a10, a11, a12, a13, a14, a15, a16, a17),
    )


@gluon.jit
def _pv_all(
    cfg,
    probabilities0,
    probabilities1,
    value_shared,
    accumulators,
):
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 0
    )
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 1
    )
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 2
    )
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 3
    )
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 4
    )
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 5
    )
    accumulators = _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 6
    )
    return _pv_fragment(
        cfg, probabilities0, probabilities1, value_shared, accumulators, 7
    )


@gluon.jit
def _pv_with_probability_pair(
    cfg,
    previous_probabilities0,
    previous_probabilities1,
    value_shared,
    accumulators,
    next_scores0,
    next_scores1,
    scale_log2: gl.constexpr,
    next_scaled0,
    next_scaled1,
    n_fragment: gl.constexpr,
):
    score_start: gl.constexpr = n_fragment * 2
    probability00, row_sum00 = _probability_fragment(
        next_scores0[score_start], scale_log2, next_scaled0
    )
    probability01, row_sum01 = _probability_fragment(
        next_scores0[score_start + 1], scale_log2, next_scaled0
    )
    probability10, row_sum10 = _probability_fragment(
        next_scores1[score_start], scale_log2, next_scaled1
    )
    probability11, row_sum11 = _probability_fragment(
        next_scores1[score_start + 1], scale_log2, next_scaled1
    )
    accumulators = _pv_fragment(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        accumulators,
        n_fragment,
    )
    return (
        accumulators,
        probability00,
        probability01,
        probability10,
        probability11,
        row_sum00 + row_sum01,
        row_sum10 + row_sum11,
    )


@gluon.jit
def _pv_with_next_probabilities(
    cfg,
    previous_probabilities0,
    previous_probabilities1,
    value_shared,
    accumulators,
    next_scores0,
    next_scores1,
    scale_log2: gl.constexpr,
    next_scaled0,
    next_scaled1,
):
    r0 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        accumulators,
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        0,
    )
    r1 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r0[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        1,
    )
    r2 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r1[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        2,
    )
    r3 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r2[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        3,
    )
    r4 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r3[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        4,
    )
    r5 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r4[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        5,
    )
    r6 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r5[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        6,
    )
    r7 = _pv_with_probability_pair(
        cfg,
        previous_probabilities0,
        previous_probabilities1,
        value_shared,
        r6[0],
        next_scores0,
        next_scores1,
        scale_log2,
        next_scaled0,
        next_scaled1,
        7,
    )
    probabilities0 = (
        r0[1],
        r0[2],
        r1[1],
        r1[2],
        r2[1],
        r2[2],
        r3[1],
        r3[2],
        r4[1],
        r4[2],
        r5[1],
        r5[2],
        r6[1],
        r6[2],
        r7[1],
        r7[2],
    )
    probabilities1 = (
        r0[3],
        r0[4],
        r1[3],
        r1[4],
        r2[3],
        r2[4],
        r3[3],
        r3[4],
        r4[3],
        r4[4],
        r5[3],
        r5[4],
        r6[3],
        r6[4],
        r7[3],
        r7[4],
    )
    row_sum0 = r0[5] + r1[5] + r2[5] + r3[5] + r4[5] + r5[5] + r6[5] + r7[5]
    row_sum1 = r0[6] + r1[6] + r2[6] + r3[6] + r4[6] + r5[6] + r6[6] + r7[6]
    return r7[0], probabilities0, probabilities1, row_sum0, row_sum1


@gluon.jit
def _attention_tile(cfg, query, key_shared, value_shared, softmax_scale: gl.constexpr):
    row_max = (
        gl.full(
            [64],
            float("-inf"),
            gl.float32,
            gl.SliceLayout(1, cfg.pv_layout),
        ),
        gl.full(
            [64],
            float("-inf"),
            gl.float32,
            gl.SliceLayout(1, cfg.pv_layout),
        ),
    )
    row_sum = (
        gl.full([64], 1.0, gl.float32, gl.SliceLayout(1, cfg.pv_layout)),
        gl.full([64], 1.0, gl.float32, gl.SliceLayout(1, cfg.pv_layout)),
    )
    zero_accumulator = gl.zeros([64, 16], gl.float32, cfg.pv_layout)
    accumulators = (
        (
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
        ),
        (
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
            zero_accumulator,
        ),
    )
    scale_log2: gl.constexpr = softmax_scale * 1.4426950408889634
    block_n: gl.constexpr = 256

    scores00, scores01 = _qk_all_grouped(cfg, query, key_shared[0], 0)
    tile_max00 = _max16(scores00)
    tile_max01 = _max16(scores01)
    row_max00 = gl.maximum(row_max[0], tile_max00)
    row_max01 = gl.maximum(row_max[1], tile_max01)
    scaled00 = row_max00 * scale_log2
    scaled01 = row_max01 * scale_log2
    alpha00 = gl.exp2(row_max[0] * scale_log2 - scaled00)
    alpha01 = gl.exp2(row_max[1] * scale_log2 - scaled01)
    accumulators = (
        _rescale8(accumulators[0], alpha00),
        _rescale8(accumulators[1], alpha01),
    )
    row_max = (row_max00, row_max01)

    (
        scores10,
        scores11,
        probabilities00,
        probabilities01,
        tile_sum00,
        tile_sum01,
    ) = _qk_all_with_probabilities(
        cfg,
        query,
        key_shared[1],
        block_n,
        scores00,
        scores01,
        scale_log2,
        scaled00,
        scaled01,
    )
    row_sum = (
        row_sum[0] * alpha00 + tile_sum00,
        row_sum[1] * alpha01 + tile_sum01,
    )
    tile_max10 = _max16(scores10)
    tile_max11 = _max16(scores11)
    row_max10 = gl.maximum(row_max[0], tile_max10)
    row_max11 = gl.maximum(row_max[1], tile_max11)
    scaled10 = row_max10 * scale_log2
    scaled11 = row_max11 * scale_log2
    alpha10 = gl.exp2(row_max[0] * scale_log2 - scaled10)
    alpha11 = gl.exp2(row_max[1] * scale_log2 - scaled11)
    (
        accumulators,
        probabilities10,
        probabilities11,
        tile_sum10,
        tile_sum11,
    ) = _pv_with_next_probabilities(
        cfg,
        probabilities00,
        probabilities01,
        value_shared[0],
        accumulators,
        scores10,
        scores11,
        scale_log2,
        scaled10,
        scaled11,
    )
    accumulators = (
        _rescale8(accumulators[0], alpha10),
        _rescale8(accumulators[1], alpha11),
    )
    row_sum = (
        row_sum[0] * alpha10 + tile_sum10,
        row_sum[1] * alpha11 + tile_sum11,
    )
    accumulators = _pv_all(
        cfg,
        probabilities10,
        probabilities11,
        value_shared[1],
        accumulators,
    )

    output0 = _join_columns(
        _join_columns(
            _join_columns(accumulators[0][0], accumulators[0][1], cfg.pv_layout),
            _join_columns(accumulators[0][2], accumulators[0][3], cfg.pv_layout),
            cfg.pv_layout,
        ),
        _join_columns(
            _join_columns(accumulators[0][4], accumulators[0][5], cfg.pv_layout),
            _join_columns(accumulators[0][6], accumulators[0][7], cfg.pv_layout),
            cfg.pv_layout,
        ),
        cfg.pv_layout,
    )
    output1 = _join_columns(
        _join_columns(
            _join_columns(accumulators[1][0], accumulators[1][1], cfg.pv_layout),
            _join_columns(accumulators[1][2], accumulators[1][3], cfg.pv_layout),
            cfg.pv_layout,
        ),
        _join_columns(
            _join_columns(accumulators[1][4], accumulators[1][5], cfg.pv_layout),
            _join_columns(accumulators[1][6], accumulators[1][7], cfg.pv_layout),
            cfg.pv_layout,
        ),
        cfg.pv_layout,
    )
    output0 *= (1.0 / row_sum[0])[:, None]
    output1 *= (1.0 / row_sum[1])[:, None]
    return _join_rows(output0, output1, cfg.pv_layout)


@gluon.jit
def _wan_cross_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    SOFTMAX_SCALE: gl.constexpr,
):
    SEQLEN_Q: gl.constexpr = 176_400
    SEQLEN_K: gl.constexpr = 512
    NUM_HEADS: gl.constexpr = 40
    HEAD_SIZE: gl.constexpr = 128
    BLOCK_M: gl.constexpr = 128
    BLOCK_N: gl.constexpr = 256
    Q_TILES_PER_WG: gl.constexpr = 29
    NUM_QUERY_GROUPS: gl.constexpr = 48
    BASE_BLOCKS: gl.constexpr = 28
    EXTRA_GROUPS: gl.constexpr = 35
    NUM_WARPS: gl.constexpr = 4
    QUERY_BATCH_STRIDE: gl.constexpr = SEQLEN_Q * NUM_HEADS * HEAD_SIZE
    QUERY_HEAD_STRIDE: gl.constexpr = HEAD_SIZE
    QUERY_ROW_STRIDE: gl.constexpr = NUM_HEADS * HEAD_SIZE
    KV_BATCH_STRIDE: gl.constexpr = SEQLEN_K * NUM_HEADS * HEAD_SIZE
    KV_HEAD_STRIDE: gl.constexpr = HEAD_SIZE
    KV_ROW_STRIDE: gl.constexpr = NUM_HEADS * HEAD_SIZE
    OUTPUT_BATCH_STRIDE: gl.constexpr = QUERY_BATCH_STRIDE
    OUTPUT_HEAD_STRIDE: gl.constexpr = HEAD_SIZE
    OUTPUT_ROW_STRIDE: gl.constexpr = QUERY_ROW_STRIDE

    cfg = _AttentionConfig(
        SEQLEN_Q,
        SEQLEN_K,
        HEAD_SIZE,
        BLOCK_M,
        16,
        1,
        NUM_WARPS,
    )
    batch_id = gl.program_id(0)
    head_id = gl.program_id(1)
    query_group = gl.program_id(2)
    batch_id_i64 = batch_id.to(gl.int64)
    head_id_i64 = head_id.to(gl.int64)
    query_base = (
        q_ptr + QUERY_BATCH_STRIDE * batch_id_i64 + QUERY_HEAD_STRIDE * head_id_i64
    )
    key_base = k_ptr + KV_BATCH_STRIDE * batch_id_i64 + KV_HEAD_STRIDE * head_id_i64
    value_base = v_ptr + KV_BATCH_STRIDE * batch_id_i64 + KV_HEAD_STRIDE * head_id_i64
    output_base = (
        output_ptr
        + OUTPUT_BATCH_STRIDE * batch_id_i64
        + OUTPUT_HEAD_STRIDE * head_id_i64
    )

    key_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE, 8]], [BLOCK_N, HEAD_SIZE], [1, 0]
    )
    value_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE, 16]], [BLOCK_N, HEAD_SIZE], [1, 0]
    )
    key_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=key_base,
        shape=(SEQLEN_K, HEAD_SIZE),
        strides=(KV_ROW_STRIDE, 1),
        block_shape=(BLOCK_N, HEAD_SIZE),
        layout=key_layout,
    )
    value_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=value_base,
        shape=(SEQLEN_K, HEAD_SIZE),
        strides=(KV_ROW_STRIDE, 1),
        block_shape=(BLOCK_N, HEAD_SIZE),
        layout=value_layout,
    )
    key_shared = gl.allocate_shared_memory(
        key_desc.dtype,
        shape=[2] + key_desc.block_shape,
        layout=key_desc.layout,
    )
    value_shared = gl.allocate_shared_memory(
        value_desc.dtype,
        shape=[2] + value_desc.block_shape,
        layout=value_desc.layout,
    )

    query_prefetch_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE, 8]], [BLOCK_M, HEAD_SIZE], [1, 0]
    )
    query_prefetch_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=query_base,
        shape=(SEQLEN_Q, HEAD_SIZE),
        strides=(QUERY_ROW_STRIDE, 1),
        block_shape=(BLOCK_M, HEAD_SIZE),
        layout=query_prefetch_layout,
    )
    output_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_SIZE, 8]], [BLOCK_M, HEAD_SIZE], [1, 0]
    )
    output_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=output_base,
        shape=(SEQLEN_Q, HEAD_SIZE),
        strides=(OUTPUT_ROW_STRIDE, 1),
        block_shape=(BLOCK_M, HEAD_SIZE),
        layout=output_layout,
    )
    output_shared = gl.allocate_shared_memory(
        output_ptr.dtype.element_ty,
        shape=(BLOCK_M, HEAD_SIZE),
        layout=output_layout,
    )

    assert (
        NUM_QUERY_GROUPS * BASE_BLOCKS + EXTRA_GROUPS
        == (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    )
    group_size = BASE_BLOCKS + (query_group < EXTRA_GROUPS)
    group_start = query_group * BASE_BLOCKS + gl.minimum(query_group, EXTRA_GROUPS)

    initial_query_start = group_start * BLOCK_M
    if group_size > 0:
        gl.amd.cdna5.tdm.prefetch(
            query_prefetch_desc,
            [initial_query_start, 0],
            pred=initial_query_start < SEQLEN_Q,
            speculative=False,
        )

    key_desc0 = gl.amd.cdna5.tdm.update_tensor_descriptor(
        key_desc,
        add_offsets=[0, 0],
        pred=True,
        clamp_bounds=True,
    )
    key_desc1 = gl.amd.cdna5.tdm.update_tensor_descriptor(
        key_desc,
        add_offsets=[BLOCK_N, 0],
        pred=True,
        clamp_bounds=True,
    )
    value_desc0 = gl.amd.cdna5.tdm.update_tensor_descriptor(
        value_desc,
        add_offsets=[0, 0],
        pred=True,
        clamp_bounds=True,
    )
    value_desc1 = gl.amd.cdna5.tdm.update_tensor_descriptor(
        value_desc,
        add_offsets=[BLOCK_N, 0],
        pred=True,
        clamp_bounds=True,
    )
    gl.amd.cdna5.tdm.async_load_fused(
        [
            (key_desc0, key_shared.index(0), 0x1),
            (key_desc1, key_shared.index(1), 0x2),
            (value_desc0, value_shared.index(0), 0x4),
            (value_desc1, value_shared.index(1), 0x8),
        ]
    )
    gl.amd.cdna5.tdm.async_wait(0)

    key_tiles = (key_shared.index(0), key_shared.index(1))
    value_tiles = (value_shared.index(0), value_shared.index(1))
    for tile in range(Q_TILES_PER_WG):
        if tile < group_size:
            query_block = group_start + tile
            query_start = query_block * BLOCK_M
            future_tile = tile + 1
            if future_tile < group_size:
                future_query_start = (group_start + future_tile) * BLOCK_M
                gl.amd.cdna5.tdm.prefetch(
                    query_prefetch_desc,
                    [future_query_start, 0],
                    pred=future_query_start < SEQLEN_Q,
                    speculative=False,
                )

            query_rows = query_start + gl.arange(
                0,
                BLOCK_M,
                layout=gl.SliceLayout(1, cfg.q_layout),
            )
            query_dims = gl.arange(
                0,
                HEAD_SIZE,
                layout=gl.SliceLayout(0, cfg.q_layout),
            )
            query_offsets = QUERY_ROW_STRIDE * query_rows[:, None].to(
                gl.int64
            ) + query_dims[None, :].to(gl.int64)
            if query_start + BLOCK_M <= SEQLEN_Q:
                query = gl.amd.cdna5.buffer_load(
                    query_base,
                    query_offsets.to(gl.int32),
                )
            else:
                query = gl.amd.cdna5.buffer_load(
                    query_base,
                    query_offsets.to(gl.int32),
                    mask=query_rows[:, None] < SEQLEN_Q,
                    other=0.0,
                )

            output = _attention_tile(
                cfg,
                query,
                key_tiles,
                value_tiles,
                SOFTMAX_SCALE,
            )
            if tile > 0:
                gl.amd.cdna5.tdm.async_wait(0)
            output_shared.store(output.to(output_ptr.dtype.element_ty))
            gl.amd.cdna5.tdm.async_store(
                output_desc,
                [query_start, 0],
                output_shared,
            )

    key_shared._keep_alive()
    value_shared._keep_alive()
    gl.amd.cdna5.tdm.async_wait(0)
    output_shared._keep_alive()


def gluon_fav3_wan_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the exact Wan2.2 cross-attention shape on gfx1250."""
    expected_query_shape = (2, 176_400, 40, 128)
    expected_kv_shape = (2, 512, 40, 128)
    if tuple(query.shape) != expected_query_shape:
        raise ValueError(
            f"Wan cross FAv3 requires query shape {expected_query_shape}, "
            f"got {tuple(query.shape)}"
        )
    if tuple(key.shape) != expected_kv_shape or tuple(value.shape) != expected_kv_shape:
        raise ValueError(
            f"Wan cross FAv3 requires key/value shape {expected_kv_shape}, "
            f"got key={tuple(key.shape)}, value={tuple(value.shape)}"
        )
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise TypeError("Wan cross FAv3 requires BF16 query, key, and value")
    if not query.is_cuda or key.device != query.device or value.device != query.device:
        raise ValueError("Wan cross FAv3 requires Q/K/V on the same ROCm device")
    if (
        not query.is_contiguous()
        or not key.is_contiguous()
        or not value.is_contiguous()
    ):
        raise ValueError(
            "Wan cross FAv3 requires contiguous BSHD query, key, and value"
        )

    if softmax_scale is None:
        softmax_scale = 128**-0.5
    softmax_scale = float(softmax_scale)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")

    arch = getattr(torch.cuda.get_device_properties(query.device), "gcnArchName", "")
    if "gfx1250" not in arch:
        raise RuntimeError(
            f"Wan cross FAv3 requires gfx1250, found {arch or 'unknown GPU'}"
        )

    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    with torch.cuda.device(query.device):
        _wan_cross_kernel[(2, 40, 48)](
            query,
            key,
            value,
            output,
            softmax_scale,
            num_warps=4,
            waves_per_eu=1,
            llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp",
        )
    return output


__all__ = ["gluon_fav3_wan_cross_attention"]
