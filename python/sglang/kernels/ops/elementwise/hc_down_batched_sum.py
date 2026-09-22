"""Large-T HC Down+Inject from raw residual and branch square sums.

The framework prepares folded weights after loading and shares their storage
with small-T HC. Each caller owns its scratch/output buffers. Projection uses
one branch-batched GEMM and one merge/nonlinear kernel, without token chunking
or weight packing on the inference path.
"""

import math
from dataclasses import dataclass

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@dataclass(frozen=True)
class BatchedDownWeights:
    # [C,H,Npad], a transpose view of contiguous [C,Npad,H] storage.
    matrix: torch.Tensor
    hidden_size: int
    lowrank: int
    hc_count: int = 4


@torch.no_grad()
def prepare_batched_down_weights(down, inject, norm_weight, hc_count=4):
    if hc_count != 4 or down.ndim != 2 or down.shape[1] % hc_count:
        raise ValueError("batched HC Down requires C=4 and [L,C*H] weights")
    lowrank, width = down.shape
    h = width // hc_count
    if not lowrank > 0 or not h > 0:
        raise ValueError("hidden size and lowrank must be positive")
    if not down.is_cuda or down.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("CUDA BF16/FP16 weights required")
    for tensor, shape in ((inject, (hc_count, width)), (norm_weight, (width,))):
        if (
            tensor.shape != shape
            or tensor.device != down.device
            or tensor.dtype != down.dtype
        ):
            raise ValueError("weight shape/device/dtype mismatch")
    logical_n = lowrank + hc_count
    padded_n = triton.cdiv(logical_n, 8) * 8
    folded = torch.zeros((padded_n, width), device=down.device, dtype=down.dtype)
    folded[:logical_n] = (
        torch.cat((down, inject)).float() * (1.0 + norm_weight.float())
    ).to(down.dtype)
    branch_major = folded.view(padded_n, hc_count, h).permute(1, 0, 2).contiguous()
    return BatchedDownWeights(branch_major.transpose(1, 2), h, lowrank, hc_count)


@gluon.jit
def _shuffle_inv_pair(values, source, ODD: gl.constexpr):
    # Layout is explicitly Blocked([2],[32],[4],[0]), not inferred from pack.
    # Each lane owns even/odd logical elements in two consecutive registers.
    # A logical group of32 elements occupies16 physical lanes. 'source' indexes
    # that group; branch parity selects the register, source//2 selects its lane.
    if ODD:
        result = gl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 lane, half, s0, s1;
                mov.u32 lane, %laneid;
                and.b32 half, lane, 16;
                shr.u32 s0, $4, 1;
                shr.u32 s1, $5, 1;
                or.b32 s0, s0, half;
                or.b32 s1, s1, half;
                shfl.sync.idx.b32 $0, $3, s0, 31, -1;
                shfl.sync.idx.b32 $1, $3, s1, 31, -1;
            }
            """,
            constraints="=&r,=&r,r,r,r,r",
            args=[values.to(gl.int32, bitcast=True), source],
            dtype=gl.int32,
            is_pure=True,
            pack=2,
        )
    else:
        result = gl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 lane, half, s0, s1;
                mov.u32 lane, %laneid;
                and.b32 half, lane, 16;
                shr.u32 s0, $4, 1;
                shr.u32 s1, $5, 1;
                or.b32 s0, s0, half;
                or.b32 s1, s1, half;
                shfl.sync.idx.b32 $0, $2, s0, 31, -1;
                shfl.sync.idx.b32 $1, $2, s1, 31, -1;
            }
            """,
            constraints="=&r,=&r,r,r,r,r",
            args=[values.to(gl.int32, bitcast=True), source],
            dtype=gl.int32,
            is_pure=True,
            pack=2,
        )
    return result.to(gl.float32, bitcast=True)


@gluon.jit(do_not_specialize=["BATCH_STRIDE"])
def _hc_down_batched_sum_post_kernel(
    partial,
    sums,
    down,
    alpha,
    numel,
    H: gl.constexpr,
    LOWRANK: gl.constexpr,
    C: gl.constexpr,
    NP: gl.constexpr,
    BATCH_STRIDE,
    WIDE_INDEX: gl.constexpr,
    EPS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    gl.static_assert(C == 4)
    gl.static_assert(LOWRANK > 0)
    gl.static_assert(BLOCK == 256)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [4], [0])
    n: gl.constexpr = LOWRANK + C
    # Only index width specializes on T, not the exact branch stride. The host
    # includes tail lanes in its range check; global address products stay wide.
    index_dtype: gl.constexpr = gl.int64 if WIDE_INDEX else gl.int32
    x = gl.program_id(0).to(index_dtype) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    live = x < numel
    row = x // n
    col = x % n

    # n>=5: <=8 intersecting token rows per32-output logical group,4 slots each.
    group_start = (x // 32) * 32
    first_row = group_start // n
    last_row = (group_start + 31) // n
    carrier_row = first_row + (x % 32) // C
    carrier_branch = x % C
    carrier_live = (carrier_row <= last_row) & (carrier_row.to(gl.int64) * n < numel)
    ss = gl.load(
        sums + carrier_row.to(gl.int64) * C + carrier_branch, carrier_live, other=0.0
    )
    carrier_inv = gl.rsqrt(ss * (1.0 / H) + EPS)

    row64 = row.to(gl.int64)
    # The validated contiguous [C,T,NP] buffer keeps this alignment for every
    # T. Preserve vector loads without specializing on the stride's value.
    batch_stride = gl.multiple_of(BATCH_STRIDE.to(gl.int64), NP & -NP)
    total = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for branch in gl.static_range(C):
        # Shuffle lane IDs are local to a 32-value group, not global offsets.
        source = ((row - first_row) * C + branch).to(gl.int32)
        inv = _shuffle_inv_pair(carrier_inv, source, branch % 2)
        p = gl.load(
            partial + row64 * NP + col + branch * batch_stride,
            live,
            other=0.0,
        )
        # Preserve the scalar post's FP32 fused multiply-add rounding. In
        # particular, do not let pair packing lower this into FMUL2 + FADD.
        total = gl.fma(p, inv, total)
    raw = total.to(down.dtype.element_ty).to(gl.float32)
    scaled = raw * (1.0 / C)
    sigmoid = 1 / (1 + gl.exp(-scaled))
    gl.store(down + row64 * LOWRANK + col, scaled * sigmoid, live & (col < LOWRANK))
    gl.store(alpha + row64 * C + col - LOWRANK, 2.0 * sigmoid, live & (col >= LOWRANK))


def _max_hc_rows(weights):
    # Global offsets are int64. Only CUDA grid-x limits remain: C*T for
    # bootstrap, ceil(T*H/256) for Mix, and ceil(T*(L+C)/256) for Down post.
    # Npad >= L+C is a conservative bound on the last grid.
    grid_limit = 2**31 - 1
    return min(
        grid_limit // weights.hc_count,
        grid_limit * 256 // max(weights.hidden_size, weights.matrix.shape[-1]),
    )


def _validate_batched_rows(rows, weights):
    if type(rows) is not int or not 0 <= rows <= _max_hc_rows(weights):
        raise ValueError("HC shape exceeds the CUDA grid-x limit")


def allocate_batched_down_buffers(rows, weights):
    _validate_batched_rows(rows, weights)
    w = weights
    partial = torch.empty(
        (w.hc_count, rows, w.matrix.shape[-1]),
        device=w.matrix.device,
        dtype=torch.float32,
    )
    down = torch.empty((rows, w.lowrank), device=w.matrix.device, dtype=w.matrix.dtype)
    alpha = torch.empty((rows, w.hc_count), device=w.matrix.device, dtype=torch.float32)
    return partial, down, alpha


def batched_down_from_sum(residual, sum_sq, weights, buffers, rms_eps=1e-6):
    """One batched GEMM call + one scaled-merge/nonlinear kernel.

    sums contain square sums, NOT inverse RMS. They may describe the accepted
    pre-storage-rounding FP32 update. The producer owns initialization. This
    consumer never modifies residual, sums, or prepared weights.
    """
    w = weights
    if not math.isfinite(rms_eps) or rms_eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    if residual.ndim != 2 or residual.shape[1] != w.hc_count * w.hidden_size:
        raise ValueError("residual must have shape [T,C*H]")
    t = residual.shape[0]
    _validate_batched_rows(t, w)
    partial, down, alpha = buffers
    for tensor, shape, dtype in (
        (residual, (t, w.hc_count * w.hidden_size), w.matrix.dtype),
        (sum_sq, (t, w.hc_count), torch.float32),
        (partial, (w.hc_count, t, w.matrix.shape[-1]), torch.float32),
        (down, (t, w.lowrank), w.matrix.dtype),
        (alpha, (t, w.hc_count), torch.float32),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != w.matrix.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("batched Down buffer shape/device/dtype/layout mismatch")
    # All caller-owned outputs must be disjoint from every input and output.
    inputs = (residual, sum_sq, w.matrix)
    for i, destination in enumerate(buffers):
        begin = destination.data_ptr()
        end = begin + destination.numel() * destination.element_size()
        for other in inputs + tuple(buffers[:i]):
            ob = other.data_ptr()
            oe = ob + other.numel() * other.element_size()
            if begin < oe and ob < end:
                raise ValueError("batched Down writable storage must not alias")
    if t == 0:
        return down, alpha
    # This view does not replicate the full weight or copy the residual.
    operand = residual.view(t, w.hc_count, w.hidden_size).transpose(0, 1)
    torch.bmm(operand, w.matrix, out_dtype=torch.float32, out=partial)
    n = t * (w.lowrank + w.hc_count)
    # Two outputs per lane; branch inverses stay in warp registers (no SMEM).
    _hc_down_batched_sum_post_kernel[(triton.cdiv(n, 256),)](
        partial,
        sum_sq,
        down,
        alpha,
        n,
        H=w.hidden_size,
        LOWRANK=w.lowrank,
        C=w.hc_count,
        NP=w.matrix.shape[-1],
        BATCH_STRIDE=partial.stride(0),
        WIDE_INDEX=triton.cdiv(n, 256) * 256 > 2**31,
        EPS=rms_eps,
        BLOCK=256,
        num_warps=4,
    )
    return down, alpha
