# Adapted from OpenAI-Partners/artemis-kernel-integrations PR #11.
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _column_tile(NT: gl.constexpr, GROUP: gl.constexpr):
    pid = gl.program_id(0)
    if GROUP == 0:
        pid = (pid & 85) << 1 | pid >> 1 & 85
        pid = (pid & 51) << 2 | pid >> 2 & 51
        pid = (pid << 4 | pid >> 4) & 255
    elif GROUP == 8 and NT == 512:
        pid = (pid & 7) * (NT // 8) + (pid >> 3)
    elif GROUP == 16:
        pid = (pid & 15) * (NT // 16) + (pid >> 4)
    else:
        pid = pid % GROUP * (NT // GROUP) + pid // GROUP
    return pid


@gluon.jit
def _store_rows(
    Y,
    result,
    col,
    M: gl.constexpr,
    N: gl.constexpr,
    COMPACT: gl.constexpr = False,
    BUFFERED: gl.constexpr = False,
):
    BM: gl.constexpr = result.type.shape[0]
    BN: gl.constexpr = result.type.shape[1]
    ol: gl.constexpr = result.type.layout
    om = gl.arange(0, BM, gl.SliceLayout(1, ol))
    on = col + gl.arange(0, BN, gl.SliceLayout(0, ol))
    if BUFFERED:
        gl.amd.cdna4.buffer_store(
            stored_value=result.to(gl.bfloat16),
            ptr=Y,
            offsets=om[:, None] * N + on[None, :],
            mask=om[:, None] < M,
        )
    elif COMPACT:
        gl.store(
            Y + (om[:, None] * N + on[None, :]),
            result.to(gl.bfloat16),
            (om[:, None] < M) & (on[None, :] < N),
        )
    elif M == BM:
        gl.store(Y + om[:, None] * N + on[None, :], result.to(gl.bfloat16))
    else:
        gl.store(
            Y + om[:, None] * N + on[None, :], result.to(gl.bfloat16), om[:, None] < M
        )


@gluon.jit
def _reduce_folds(acc, FOLD: gl.constexpr):
    gl.static_assert(FOLD == 4)
    gl.static_assert(len(acc.type.shape) == 2 or len(acc.type.shape) == 3)
    PREFIX: gl.constexpr = (acc.type.shape[0],) if len(acc.type.shape) == 3 else ()
    RANK: gl.constexpr = len(PREFIX)
    BM: gl.constexpr = acc.type.shape[-2] // FOLD
    BN: gl.constexpr = acc.type.shape[-1] // FOLD
    fragments = acc.reshape(PREFIX + (BM, FOLD, BN * FOLD))
    ORDER: gl.constexpr = (0, 1, 3, 2) if RANK == 1 else (0, 2, 1)
    fragments = fragments.permute(ORDER)
    even, odd = fragments.reshape(PREFIX + (BM, BN * FOLD, 2, 2)).split()
    c0, c2 = even.split()
    c1, c3 = odd.split()
    cl: gl.constexpr = c0.type.layout
    if RANK == 1:
        cf = gl.arange(0, BN * FOLD, gl.SliceLayout(0, gl.SliceLayout(1, cl)))
        cf = cf[None, None, :]
    else:
        gl.static_assert(RANK == 0)
        cf = gl.arange(0, BN * FOLD, gl.SliceLayout(0, cl))[None, :]
    lower = gl.where(cf % 2 == 0, c0, c1)
    upper = gl.where(cf % 2 == 0, c2, c3)
    diagonal = gl.where(cf % FOLD < 2, lower, upper)
    result = gl.sum(diagonal.reshape(PREFIX + (BM, BN, FOLD)), RANK + 2)
    return result


@gluon.jit
def _stage_activation(
    X,
    M: gl.constexpr,
    XM: gl.constexpr,
    XK: gl.constexpr,
    BM: gl.constexpr,
    BK: gl.constexpr,
    FOLD: gl.constexpr,
    STRIPE: gl.constexpr,
    KP: gl.constexpr,
    WARPS: gl.constexpr,
):
    if BM == 2:
        xl: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, 4], [1, 0])
        xm = gl.arange(0, BM, gl.SliceLayout(1, xl))
        xk = gl.arange(0, KP, gl.SliceLayout(0, xl))
        x = gl.load(X + xm[:, None] * XM + xk[None, :] * XK, xm[:, None] < M, 0)
        x = x.reshape((BM, KP // (BK * FOLD), BK // STRIPE, FOLD, STRIPE))
        x = x.permute(1, 0, 3, 2, 4).reshape((KP // BK * BM, BK))
        shared_x = gl.allocate_shared_memory(
            gl.bfloat16,
            (KP // BK * BM, BK),
            gl.SwizzledSharedLayout(8, 1, 8, [1, 0]),
            x,
        )
        shared_x = shared_x.reshape((KP // (BK * FOLD), BM * FOLD, BK))
        return shared_x._reinterpret(layout=gl.SwizzledSharedLayout(8, 1, 8, [1, 0]))
    else:
        PHASE: gl.constexpr = 4 if BM == 1 else 8
        if BM == 8:
            xl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
        else:
            xl: gl.constexpr = gl.BlockedLayout(
                [1, 8], [1, 64], [min(BM, WARPS), max(WARPS // BM, 1)], [1, 0]
            )
        xm = gl.arange(0, BM, gl.SliceLayout(1, xl))
        xk = gl.arange(0, KP, gl.SliceLayout(0, xl))
        if M == BM:
            x = gl.load(X + (xm[:, None] * XM + xk[None, :] * XK))
        else:
            x = gl.load(X + (gl.minimum(xm[:, None], M - 1) * XM + xk[None, :] * XK))
        x = x.reshape((BM, KP // (BK * FOLD), BK // STRIPE, FOLD, STRIPE))
        x = x.permute(1, 0, 3, 2, 4).reshape((KP // (BK * FOLD), BM * FOLD, BK))
        shared_x = gl.allocate_shared_memory(
            gl.bfloat16,
            (KP // (BK * FOLD), BM * FOLD, BK),
            gl.SwizzledSharedLayout(16, 1, PHASE, [2, 1, 0]),
            x,
        )
        return shared_x._reinterpret(
            layout=gl.SwizzledSharedLayout(16, 1, PHASE, [1, 0])
        )


@gluon.jit
def _m4_gemm(
    X,
    W,
    Y,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    XM: gl.constexpr,
    XK: gl.constexpr,
    WN: gl.constexpr,
    WK: gl.constexpr,
):
    BM: gl.constexpr = 4
    BN: gl.constexpr = 16
    BK: gl.constexpr = 256
    FOLD: gl.constexpr = 4
    STRIPE: gl.constexpr = 16
    tile = _column_tile(N // BN, 16)
    tile = tile ^ tile >> 1
    col = tile * BN
    W = W + col * WN
    Y = Y + col
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [1, 4])
    acc = gl.zeros((BM * FOLD, BN * FOLD), gl.float32, ml)
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0])
    shared_x = _stage_activation(X, M, XM, XK, BM, BK, FOLD, STRIPE, K, 4)
    nb = gl.arange(0, BN * FOLD, gl.SliceLayout(0, bl))
    kb = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bk = (kb[:, None] // STRIPE * FOLD + nb[None, :] % FOLD) * STRIPE + kb[
        :, None
    ] % STRIPE
    for start in range(K // (BK * FOLD)):
        wp = W + start * BK * FOLD * WK
        bo = nb[None, :] // FOLD * WN + bk * WK
        b = gl.amd.cdna4.buffer_load(wp, bo)
        panel = shared_x.index(start)
        a = panel.load(gl.DotOperandLayout(0, ml, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, ml, 8))
        acc = gl.amd.cdna4.mfma(a, b, acc)
    _store_rows(Y, _reduce_folds(acc, FOLD), 0, M, N, BUFFERED=True)


def bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exact M=4, N=4096, K=2048 target Q-B projection."""
    assert x.shape == (4, 2048) and weight.shape == (4096, 2048)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    out = torch.empty((4, 4096), dtype=torch.bfloat16, device=x.device)
    _m4_gemm[triton.cdiv(4096, 16),](
        x,
        weight,
        out,
        4,
        4096,
        2048,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        num_warps=4,
    )
    return out
