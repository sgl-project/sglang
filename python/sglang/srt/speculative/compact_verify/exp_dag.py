"""Normal-range exp sequence audited against the pinned SM100 Torch binary."""

import triton
import triton.language as tl


@triton.jit
def reference_exp(x):
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 a,n,k,r,e,s;
        .reg .b32 bits;
        fma.rn.sat.f32 a, $1, 0f3bbb989d, 0f3f000000;
        fma.rm.f32 n, a, 0f437c0000, 0f4b400001;
        add.rn.f32 k, n, 0fcb40007f;
        neg.f32 k, k;
        fma.rn.f32 r, $1, 0f3fb8aa3b, k;
        fma.rn.f32 r, $1, 0f32a57060, r;
        ex2.approx.ftz.f32 e, r;
        mov.b32 bits, n;
        shl.b32 bits, bits, 23;
        mov.b32 s, bits;
        mul.rn.f32 $0, s, e;
        }""",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def partial_stats(
    Z,
    Maximum,
    Targets,
    Out,
    R: tl.constexpr,
    L: tl.constexpr,
    T: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(Z + row * L + col, col < L, other=0).to(tl.float32)
    m = tl.load(Maximum + row)
    e = reference_exp(z - m)
    e = tl.where(col < L, e, 0)
    target = tl.load(Targets + row)
    total = tl.sum(e.to(tl.float64), 0)
    selected = tl.sum(tl.where(col + START == target, e, 0).to(tl.float64), 0)
    tl.store(Out + row * T + tile, total)
    tl.store(Out + R * T + row * T + tile, selected)


@triton.jit
def gather_peer_rows(
    P0, P1, P2, P3, Rows, Out, L: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr
):
    slot, tile = tl.program_id(0), tl.program_id(1)
    row = tl.maximum(tl.load(Rows + slot), 0)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    owner, local_col = col // L, col % L
    offset = row * L + local_col
    a = tl.load(P0 + offset, (col < V) & (owner == 0), other=0, volatile=True)
    b = tl.load(P1 + offset, (col < V) & (owner == 1), other=0, volatile=True)
    c = tl.load(P2 + offset, (col < V) & (owner == 2), other=0, volatile=True)
    d = tl.load(P3 + offset, (col < V) & (owner == 3), other=0, volatile=True)
    value = tl.where(owner == 0, a, tl.where(owner == 1, b, tl.where(owner == 2, c, d)))
    tl.store(Out + slot * V + col, value.to(tl.float32), col < V)


@triton.jit
def patch_from_softmax(
    P, Probs, Rows, Targets, N: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr
):
    slot = tl.arange(0, BLOCK)
    row = tl.load(Rows + slot, slot < N, other=-1)
    token = tl.load(Targets + tl.maximum(row, 0), (slot < N) & (row >= 0), other=0)
    value = tl.load(Probs + slot * V + token, (slot < N) & (row >= 0), other=0)
    tl.store(P + row, value, (slot < N) & (row >= 0))
