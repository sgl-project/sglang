// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "kernels.cuh"
#include "kernels_ops.cuh"

namespace omnidreams_singleview {

enum class OpType : uint32_t {
    Conv1x1,
    Conv3x3,
    MemBlock,
    TPool,
    TGrow,
    Upsample2x,
    Clamp
};

struct ConvParams {
    const half* weight;
    const half* bias; // nullable
    int Cin, Cout, groups;
    int stride; // for 3x3
    int pad;    // for 3x3
    Activation act;
};

struct MemBlockParams {
    const half* w1; const half* b1;
    const half* w2; const half* b2;
    const half* w3; const half* b3;
    const half* wskip; // null if identity
    int Cout;
};

struct PoolParams {
    int stride;
};

struct GrowParams {
    int stride;
};

struct UpsampleParams {
    int scale; // currently only 2 is supported
};

struct OpDesc {
    OpType type;
    union {
        ConvParams conv;
        MemBlockParams mem;
        PoolParams pool;
        GrowParams grow;
        UpsampleParams up;
    } u;
};

inline OpDesc MakeConv1x1(const half* w, const half* b, int Cin, int Cout, int groups, Activation act)
{
    OpDesc o;
    o.type = OpType::Conv1x1;
    o.u.conv.weight = w;
    o.u.conv.bias = b;
    o.u.conv.Cin = Cin;
    o.u.conv.Cout = Cout;
    o.u.conv.groups = groups;
    o.u.conv.stride = 1;
    o.u.conv.pad = 0;
    o.u.conv.act = act;
    return o;
}

inline OpDesc MakeConv3x3(const half* w, const half* b, int Cin, int Cout, int groups, int stride, int pad, Activation act)
{
    OpDesc o;
    o.type = OpType::Conv3x3;
    o.u.conv.weight = w;
    o.u.conv.bias = b;
    o.u.conv.Cin = Cin;
    o.u.conv.Cout = Cout;
    o.u.conv.groups = groups;
    o.u.conv.stride = stride;
    o.u.conv.pad = pad;
    o.u.conv.act = act;
    return o;
}

inline OpDesc MakeMemBlock(const half* w1, const half* b1, const half* w2, const half* b2, const half* w3, const half* b3, const half* wskip, int Cout)
{
    OpDesc o;
    o.type = OpType::MemBlock;
    o.u.mem.w1 = w1; o.u.mem.b1 = b1;
    o.u.mem.w2 = w2; o.u.mem.b2 = b2;
    o.u.mem.w3 = w3; o.u.mem.b3 = b3;
    o.u.mem.wskip = wskip;
    o.u.mem.Cout = Cout;
    return o;
}

inline OpDesc MakeTPool(int stride)
{
    OpDesc o;
    o.type = OpType::TPool;
    o.u.pool.stride = stride;
    return o;
}

inline OpDesc MakeTGrow(int stride)
{
    OpDesc o;
    o.type = OpType::TGrow;
    o.u.grow.stride = stride;
    return o;
}

inline OpDesc MakeUpsample2x()
{
    OpDesc o;
    o.type = OpType::Upsample2x;
    o.u.up.scale = 2;
    return o;
}

inline OpDesc MakeClamp()
{
    OpDesc o;
    o.type = OpType::Clamp;
    return o;
}

inline void execute_sequence(
    const OpDesc* ops, int num_ops,
    const half* x0,            // input [N, C0, H0, W0]
    half** stage_bufs,         // scratch buffers provided by caller
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(ops != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(num_ops >= 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(x0 != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(stage_bufs != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && C > 0 && H > 0 && W > 0);
    const half* x = x0;
    const half* past = nullptr; // Phase 1: caller must supply zero-past if MemBlock appears at t=0
    int Cin = C, Hc = H, Wc = W;

    for (int i = 0; i < num_ops; ++i) {
        const OpDesc& op = ops[i];
        switch (op.type) {
            case OpType::Conv1x1: {
                const ConvParams& p = op.u.conv;
                half* y = stage_bufs[i % 2];
                // NHWC-only path: reinterpret [N,C,H,W] as [N,H,W,C]
                conv2d_1x1_nhwc_half(
                    x, p.weight, p.bias, y,
                    /*N=*/N, /*H=*/Hc, /*W=*/Wc, /*Cin=*/p.Cin, /*Cout=*/p.Cout,
                    p.groups, p.act, stream);
                x = y; Cin = p.Cout;
                break;
            }
            case OpType::Conv3x3: {
                const ConvParams& p = op.u.conv;
                half* y = stage_bufs[i % 2];
                int H_out = conv3x3_out_dim(Hc, p.stride, p.pad);
                int W_out = conv3x3_out_dim(Wc, p.stride, p.pad);
                conv2d_3x3_nhwc_half(
                    x, p.weight, p.bias, y,
                    /*N=*/N, /*H_in=*/Hc, /*W_in=*/Wc, /*Cin=*/p.Cin, /*Cout=*/p.Cout,
                    p.groups, p.act, p.stride, p.pad, H_out, W_out, stream);
                x = y; Cin = p.Cout; Hc = H_out; Wc = W_out;
                break;
            }
            case OpType::MemBlock: {
                const MemBlockParams& m = op.u.mem;
                // stage_bufs: tmp_cat, tmp1, tmp2, out
                half* tmp_cat = stage_bufs[0];
                half* tmp1    = stage_bufs[1];
                half* tmp2    = stage_bufs[2];
                half* y       = stage_bufs[3];
                // Assert correct zero-past is supplied at t=0
                OMNIDREAMS_SINGLEVIEW_ASSERT(past != nullptr && "MemBlock requires explicit zero-past at t=0; supply a zero buffer");
                const half* past_in = past;
                memblock_forward_nhwc(x, past_in, m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.wskip, tmp_cat, tmp1, tmp2, y, N, Cin, Hc, Wc, m.Cout, stream);
                past = x; x = y; Cin = m.Cout;
                break;
            }
            case OpType::TPool: {
                // No-op; handled by higher level wrapper
                break;
            }
            case OpType::TGrow: {
                // No-op; handled by higher level wrapper
                break;
            }
            case OpType::Upsample2x: {
                const UpsampleParams& up = op.u.up;
                if (up.scale == 2) {
                    half* y = stage_bufs[i % 2];
                    dim3 block(16, 16, 1);
                    dim3 grid((Wc * 2 + block.x - 1) / block.x, (Hc * 2 + block.y - 1) / block.y, N);
                    upsample2x_nearest_nhwc(x, y, N, Hc, Wc, Cin, stream);
                    x = y; Hc *= 2; Wc *= 2;
                }
                break;
            }
            case OpType::Clamp: {
                // Apply elementwise ClampTanh3 in-place to current tensor x
                clamp_tanh3_inplace(const_cast<half*>(x), N, Cin, Hc, Wc, stream);
                break;
            }
        }
    }
}

inline void execute_with_parallel_past(
    const OpDesc* ops, int num_ops,
    const half* x_nt,            // input [N*T, C0, H0, W0]
    half** stage_bufs,           // scratch buffers provided by caller
    int N, int T, int C, int H, int W,
    cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(ops != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(x_nt != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(stage_bufs != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > 0 && C > 0 && H > 0 && W > 0);
    const half* x = x_nt;
    int Cin = C, Hc = H, Wc = W;
    // Allocate/build a shifted past buffer in stage_bufs[4]
    half* past = stage_bufs[4];
    OMNIDREAMS_SINGLEVIEW_ASSERT(past != nullptr);
    build_past_shifted_nhwc(x, past, N, T, Cin, Hc, Wc, stream);

    for (int i = 0; i < num_ops; ++i) {
        const OpDesc& op = ops[i];
        switch (op.type) {
            case OpType::Conv1x1: {
                const ConvParams& p = op.u.conv;
                half* y = stage_bufs[i % 2];
                conv2d_1x1_nhwc_half(
                    x, p.weight, p.bias, y,
                    /*N=*/N*T, /*H=*/Hc, /*W=*/Wc, /*Cin=*/p.Cin, /*Cout=*/p.Cout,
                    p.groups, p.act, stream);
                x = y; Cin = p.Cout;
                break;
            }
            case OpType::Conv3x3: {
                const ConvParams& p = op.u.conv;
                half* y = stage_bufs[i % 2];
                int H_out = conv3x3_out_dim(Hc, p.stride, p.pad);
                int W_out = conv3x3_out_dim(Wc, p.stride, p.pad);
                conv2d_3x3_nhwc_half(
                    x, p.weight, p.bias, y,
                    /*N=*/N*T, /*H_in=*/Hc, /*W_in=*/Wc, /*Cin=*/p.Cin, /*Cout=*/p.Cout,
                    p.groups, p.act, p.stride, p.pad, H_out, W_out, stream);
                x = y; Cin = p.Cout; Hc = H_out; Wc = W_out;
                break;
            }
            case OpType::MemBlock: {
                const MemBlockParams& m = op.u.mem;
                half* tmp_cat = stage_bufs[0];
                half* tmp1    = stage_bufs[1];
                half* tmp2    = stage_bufs[2];
                half* y       = stage_bufs[3];
                memblock_forward_nhwc(x, past, m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.wskip, tmp_cat, tmp1, tmp2, y, N*T, Cin, Hc, Wc, m.Cout, stream);
                // Update past to be shifted version of new x for next MemBlock layer
                past = stage_bufs[4];
                build_past_shifted_nhwc(y, past, N, T, m.Cout, Hc, Wc, stream);
                x = y; Cin = m.Cout;
                break;
            }
            case OpType::TPool: {
                // Coalesce with next Conv1x1 at higher level; no-op here
                break;
            }
            case OpType::TGrow: {
                // Coalesce with next Conv1x1 at higher level; no-op here
                break;
            }
            case OpType::Upsample2x: {
                const UpsampleParams& up = op.u.up;
                if (up.scale == 2) {
                    half* y = stage_bufs[i % 2];
                    dim3 block(16, 16, 1);
                    dim3 grid((Wc * 2 + block.x - 1) / block.x, (Hc * 2 + block.y - 1) / block.y, N*T);
                    upsample2x_nearest_nhwc(x, y, N*T, Hc, Wc, Cin, stream);
                    x = y; Hc *= 2; Wc *= 2;
                }
                break;
            }
            case OpType::Clamp: {
                clamp_tanh3_inplace(const_cast<half*>(x), N*T, Cin, Hc, Wc, stream);
                break;
            }
        }
    }
}

} // namespace taehv
