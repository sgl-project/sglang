"""Single-request BF16 bidirectional attention with grouped query-head GEMMs.

The caller supplies a static prefix capacity bounding the GPU indptr length.
Masked gathers never dereference padding indices, including during graph replay.
Separate K and V tensors are required; this is not shared-K/V MLA attention.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv(
    K,
    V,
    PK,
    PV,
    IDX,
    PTR,
    OK,
    OV,
    CAP: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    valid = x < H * (CAP + M) * D
    d = x % D
    t = x // D % (CAP + M)
    h = x // ((CAP + M) * D)
    begin = tl.load(PTR)
    n = tl.load(PTR + 1) - begin
    prefix = (t < CAP) & (t < n) & valid
    idx = tl.load(IDX + begin + t, mask=prefix, other=0)
    pk = tl.load(PK + idx * KS + h * D + d, mask=prefix, other=0)
    pv = tl.load(PV + idx * VS + h * D + d, mask=prefix, other=0)
    canvas = (t >= CAP) & valid
    ck = tl.load(K + (t - CAP) * H * D + h * D + d, mask=canvas, other=0)
    cv = tl.load(V + (t - CAP) * H * D + h * D + d, mask=canvas, other=0)
    tl.store(OK + x, pk + ck, mask=valid)
    tl.store(OV + x, pv + cv, mask=valid)


@triton.jit
def _masked_softmax(
    S, P, PTR, N: tl.constexpr, CAP: tl.constexpr, SCALE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    n = tl.load(PTR + 1) - tl.load(PTR)
    s = tl.load(S + row * N + c, mask=c < N, other=-float("inf")) * SCALE
    s = tl.where((c < n) | ((c >= CAP) & (c < N)), s, -float("inf"))
    e = tl.exp(s - tl.max(s, 0))
    p = e / tl.sum(e, 0)
    tl.store(P + row * N + c, p, mask=c < N)


@triton.jit
def _bmm_output_copy(
    INPUT,
    OUTPUT,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < M * H * D
    d = x % D
    h = x // D % H
    m = x // (D * H)
    value = tl.load(INPUT + (h * M + m) * D + d, valid, other=0)
    tl.store(OUTPUT + m * OUTPUT_STRIDE + h * D + d, value.to(tl.bfloat16), valid)


def _copy_bmm_output(source, output):
    """Round FP32 head-major values into an existing BF16 token-major buffer."""
    m, h, d = output.shape
    assert source.is_contiguous() and source.numel() == output.numel()
    assert source.dtype == torch.float32 and output.dtype == torch.bfloat16
    assert output.stride(2) == 1 and output.stride(1) == d
    _bmm_output_copy[(triton.cdiv(output.numel(), 2048),)](
        source, output, m, h, d, output.stride(0), 2048, num_warps=4
    )
    return output


def bidirectional_bmm_attention(q, k, v, pk, pv, ids, ptr, cap, scale=1.0, out=None):
    # K/V canvas and the last two pool dimensions must be contiguous. Q may
    # retain the token stride of a packed QKV projection.
    assert k.is_contiguous() and v.is_contiguous()
    assert pk.stride(1) == pv.stride(1) == q.shape[-1]
    assert pk.stride(2) == pv.stride(2) == 1
    m, hq, d = q.shape
    hk = k.shape[1]
    n = cap + m
    keys = torch.empty((hk, n, d), device=q.device, dtype=q.dtype)
    vals = torch.empty_like(keys)
    _gather_kv[(triton.cdiv(keys.numel(), 1024),)](
        k,
        v,
        pk,
        pv,
        ids,
        ptr,
        keys,
        vals,
        cap,
        m,
        hk,
        d,
        pk.stride(0),
        pv.stride(0),
        1024,
    )
    # Consecutive query heads share one KV head. Fold them into GEMM's M
    # dimension to avoid repeating the large prefix K/V matrices.
    qq = q.transpose(0, 1).reshape(hk, -1, d)
    scores = torch.bmm(qq, keys.transpose(1, 2), out_dtype=torch.float32)
    probs = torch.empty_like(scores, dtype=torch.bfloat16)
    _masked_softmax[(hk * qq.shape[1],)](
        scores,
        probs,
        ptr,
        n,
        cap,
        scale,
        triton.next_power_of_2(n),
        num_warps=4 if n <= 2048 else (8 if n >= 8192 else 16),
    )
    result = torch.bmm(probs, vals, out_dtype=torch.float32)
    if out is None:
        out = torch.empty((m, hq, d), device=q.device, dtype=q.dtype)
    assert out.shape == (m, hq, d)
    return _copy_bmm_output(result, out)
