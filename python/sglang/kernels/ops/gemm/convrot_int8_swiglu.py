import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _multiply_ftz(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_ftz(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _swiglu_kernel(
    a_desc,
    w_desc,
    x_scale,
    w_scale,
    bias,
    out,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(m, BM)
    num_n = tl.cdiv(n, BN)
    group = pid // (GROUP_M * num_n)
    rows = tl.minimum(num_m - group * GROUP_M, GROUP_M)
    pid_m = group * GROUP_M + (pid % (GROUP_M * num_n)) % rows
    pid_n = (pid % (GROUP_M * num_n)) // rows
    gate_acc = tl.zeros((BM, BN), tl.int32)
    up_acc = tl.zeros((BM, BN), tl.int32)
    for block in range(tl.cdiv(k, BK)):
        a = a_desc.load([pid_m * BM, block * BK])
        gate = w_desc.load([pid_n * BN, block * BK])
        up = w_desc.load([n + pid_n * BN, block * BK])
        gate_acc = tl.dot(a, gate.T, gate_acc, out_dtype=tl.int32)
        up_acc = tl.dot(a, up.T, up_acc, out_dtype=tl.int32)

    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    xs = tl.load(x_scale + row, row < m, other=0)
    gs = tl.load(w_scale + col, col < n, other=0)
    us = tl.load(w_scale + n + col, col < n, other=0)
    # Match the CUTLASS epilogue's FTZ scale product and separate bias add.
    gate = _multiply_ftz(
        gate_acc.to(tl.float32), _multiply_ftz(xs[:, None], gs[None, :])
    )
    up = _multiply_ftz(up_acc.to(tl.float32), _multiply_ftz(xs[:, None], us[None, :]))
    if HAS_BIAS:
        gb = tl.load(bias + col, col < n, other=0).to(tl.float32)
        ub = tl.load(bias + n + col, col < n, other=0).to(tl.float32)
        gate = _add_ftz(gate, gb[None, :])
        up = _add_ftz(up, ub[None, :])
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    denominator = tl.extra.cuda.libdevice.add_rn(
        1.0, tl.extra.cuda.libdevice.exp(-gate)
    )
    activated = (
        tl.extra.cuda.libdevice.div_rn(gate, denominator).to(tl.bfloat16).to(tl.float32)
    )
    result = tl.extra.cuda.libdevice.mul_rn(activated, up)
    tl.store(
        out + row.to(tl.int64)[:, None] * n + col[None, :],
        result,
        (row[:, None] < m) & (col[None, :] < n),
    )


def swiglu_prequant(xq, xs, weight, weight_scale, bias, out):
    m, k = xq.shape
    n = weight.shape[0] // 2
    if m < 256:
        bm, bn, bk, warps, stages = 64, 64, 128, 4, 3
    elif m <= 2048:
        bm, bn, bk, warps, stages = 256, 64, 128, 8, 3
    else:
        bm, bn, bk, warps, stages = 128, 128, 64, 8, 4
    a_desc = TensorDescriptor(xq, list(xq.shape), list(xq.stride()), [bm, bk])
    w_desc = TensorDescriptor(
        weight, list(weight.shape), list(weight.stride()), [bn, bk]
    )
    _swiglu_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a_desc,
        w_desc,
        xs,
        weight_scale,
        bias,
        out,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        32,
        bias is not None,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
