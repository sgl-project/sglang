"""fp32 router GEMV for the Laguna proxy MoE on gfx1250 (FlyDSL), v1.

logits[M, E] = x[M, K].float() @ W[E, K]^T, fp32 weights and accumulation (as LagunaMoEGate).

v1 layout: wave32; a block is 8 independent waves. Wave w owns EPW experts x one of KS K-chunks.
Per step a lane loads 4 consecutive fp32 weights per expert (16 B) and 4 bf16 x per token, FMAs into
acc[m][e]; the wave sums each acc with shuffle_xor, lane 0 stores ws[ks, m, e]. Stage 2 sums the KS
partials in a fixed order (torch, for now). Tokens are compiled in (M <= MAXM per launch).

Usage (test + bench): python3 router_gemv_gfx1250.py [--ms 1 2 4 8 16 32 64]
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl

E, K = 512, 16384
WAVE, WAVES_PER_BLOCK = 32, 8
BLOCK = WAVE * WAVES_PER_BLOCK
VEC = 4
MAXM = 16


def _make_io(elem):
    """Vector load of `vec` elements of `elem` at element offset `off` from byte address `addr`."""

    def load(addr, off, vec):
        align = max(1, elem.width * vec // 8)
        pt = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, align)
        view = rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(pt, fx.Int64(addr)), fx.make_layout((vec, 1), (1, 1))))
        )
        frag = fx.make_rmem_tensor(vec, elem)
        fx.copy(fx.make_copy_atom(rocdl.BufferCopy(elem.width * vec), elem), fx.slice(view, (None, off)), frag)
        return fx.Vector(frag.load())

    def store1(addr, off, value):
        pt = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, max(1, elem.width // 8))
        view = rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(pt, fx.Int64(addr)), fx.make_layout((1, 1), (1, 1))))
        )
        frag = fx.make_rmem_tensor(1, elem)
        frag.store(fx.Vector.from_elements([value], elem))
        fx.copy(fx.make_copy_atom(rocdl.BufferCopy(elem.width), elem), frag, fx.slice(view, (None, off)))

    return load, store1


@functools.lru_cache(maxsize=64)
def compile_router_gemv(m: int, epw: int = 4, ks: int = 8):
    assert 1 <= m <= MAXM and E % epw == 0 and K % (ks * WAVE * VEC) == 0
    kchunk = K // ks
    steps = kchunk // (WAVE * VEC)
    n_eg = E // epw
    n_waves = n_eg * ks
    assert n_waves % WAVES_PER_BLOCK == 0
    n_blocks = n_waves // WAVES_PER_BLOCK

    @flyc.kernel(name=f"laguna_router_gemv_f32_m{m}_e{epw}_k{ks}", known_block_size=[BLOCK, 1, 1])
    def gemv(x_ptr: fx.Int64, w_ptr: fx.Int64, ws_ptr: fx.Int64):
        load_w, store_f32 = _make_io(fx.Float32)
        load_x, _ = _make_io(fx.BFloat16)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(WAVE)
        gw = fx.Int32(gpu.block_id("x")) * fx.Int32(WAVES_PER_BLOCK) + tid // fx.Int32(WAVE)
        eg = gw % fx.Int32(n_eg)
        kc = gw // fx.Int32(n_eg)
        e0 = eg * fx.Int32(epw)
        kbase = kc * fx.Int32(kchunk) + lane * fx.Int32(VEC)

        acc = [[fx.Float32(0.0) for _ in range(epw)] for _ in range(m)]
        for s in range_constexpr(steps):
            k = kbase + fx.Int32(s * WAVE * VEC)
            xs = [load_x(x_ptr, fx.Int32(t * K) + k, VEC).extf(fx.T.vec(VEC, fx.T.f32)) for t in range(m)]
            ws_ = [load_w(w_ptr, (e0 + fx.Int32(e)) * fx.Int32(K) + k, VEC) for e in range(epw)]
            for t in range_constexpr(m):
                for e in range_constexpr(epw):
                    a = acc[t][e]
                    for j in range_constexpr(VEC):
                        a = a + fx.Float32(xs[t][j]) * fx.Float32(ws_[e][j])
                    acc[t][e] = a

        for t in range_constexpr(m):
            for e in range_constexpr(epw):
                v = acc[t][e]
                for sh in range_constexpr(5):  # wave32 butterfly
                    v = v + fx.Float32(v.shuffle_xor(WAVE >> (sh + 1), WAVE))
                if lane == fx.Int32(0):
                    store_f32(ws_ptr, (kc * fx.Int32(m) + fx.Int32(t)) * fx.Int32(E) + e0 + fx.Int32(e), v)

    @flyc.jit
    def launch(x_ptr: fx.Int64, w_ptr: fx.Int64, ws_ptr: fx.Int64, stream: fx.Stream):
        gemv(x_ptr, w_ptr, ws_ptr).launch(grid=(n_blocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


MAXM_V2 = 64


@functools.lru_cache(maxsize=64)
def compile_router_gemv_v2(m: int, epw: int = 1, ks: int = 32, vec: int = 8):
    """v2: runtime K loop with loop-carried acc[m][epw] (no full unroll), `vec` K per lane per step
    (x: one 16 B load per token at vec=8; W: vec/4 16 B loads per expert), all M tokens per W read."""
    assert 1 <= m <= MAXM_V2 and E % epw == 0 and vec % 4 == 0 and K % (ks * WAVE * vec) == 0
    kchunk = K // ks
    steps = kchunk // (WAVE * vec)
    n_eg = E // epw
    n_waves = n_eg * ks
    assert n_waves % WAVES_PER_BLOCK == 0
    n_blocks = n_waves // WAVES_PER_BLOCK

    @flyc.kernel(name=f"laguna_router_gemv2_f32_m{m}_e{epw}_k{ks}_v{vec}", known_block_size=[BLOCK, 1, 1])
    def gemv(x_ptr: fx.Int64, w_ptr: fx.Int64, ws_ptr: fx.Int64):
        load_w, store_f32 = _make_io(fx.Float32)
        load_x, _ = _make_io(fx.BFloat16)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(WAVE)
        gw = fx.Int32(gpu.block_id("x")) * fx.Int32(WAVES_PER_BLOCK) + tid // fx.Int32(WAVE)
        eg = gw % fx.Int32(n_eg)
        kc = gw // fx.Int32(n_eg)
        e0 = eg * fx.Int32(epw)
        kbase = kc * fx.Int32(kchunk) + lane * fx.Int32(vec)
        f32v = fx.T.vec(vec, fx.T.f32)

        init = [fx.Float32(0.0) for _ in range(m * epw)]
        for it, acc in range(fx.Int64(0), fx.Int64(steps), fx.Int64(1), init=init):
            k = kbase + fx.Int32(it) * fx.Int32(WAVE * vec)
            wv = []
            for e in range_constexpr(epw):
                row = (e0 + fx.Int32(e)) * fx.Int32(K) + k
                parts = [load_w(w_ptr, row + fx.Int32(4 * q), 4) for q in range(vec // 4)]
                wv.append([parts[j // 4][j % 4] for j in range(vec)])
            new = list(acc) if isinstance(acc, (list, tuple)) else [acc]  # one carried value comes back bare
            for t in range_constexpr(m):
                xv = load_x(x_ptr, fx.Int32(t * K) + k, vec).extf(f32v)
                for e in range_constexpr(epw):
                    a = fx.Float32(new[t * epw + e])
                    for j in range_constexpr(vec):
                        a = a + fx.Float32(xv[j]) * fx.Float32(wv[e][j])
                    new[t * epw + e] = a
            results = yield new
        results = list(results) if isinstance(results, (list, tuple)) else [results]

        for t in range_constexpr(m):
            for e in range_constexpr(epw):
                v = fx.Float32(results[t * epw + e])
                for sh in range_constexpr(5):  # wave32 butterfly
                    v = v + fx.Float32(v.shuffle_xor(WAVE >> (sh + 1), WAVE))
                if lane == fx.Int32(0):
                    store_f32(ws_ptr, (kc * fx.Int32(m) + fx.Int32(t)) * fx.Int32(E) + e0 + fx.Int32(e), v)

    @flyc.jit
    def launch(x_ptr: fx.Int64, w_ptr: fx.Int64, ws_ptr: fx.Int64, stream: fx.Stream):
        gemv(x_ptr, w_ptr, ws_ptr).launch(grid=(n_blocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


def router_gemv_v2(x, w, epw: int = 1, ks: int = 32, vec: int = 8):
    """v2 entry: x [M, K] bf16, w [E, K] fp32 -> logits [M, E] fp32; one launch per 64 tokens."""
    import torch

    assert x.dtype == torch.bfloat16 and w.dtype == torch.float32 and x.is_contiguous() and w.is_contiguous()
    m_all = x.shape[0]
    out = torch.empty(m_all, E, device=x.device, dtype=torch.float32)
    stream = fx.Stream(torch.cuda.current_stream(x.device))
    for m0 in range(0, m_all, MAXM_V2):
        m = min(MAXM_V2, m_all - m0)
        ws = torch.empty(ks, m, E, device=x.device, dtype=torch.float32)
        compile_router_gemv_v2(m, epw, ks, vec)(x[m0:m0 + m].data_ptr(), w.data_ptr(), ws.data_ptr(), stream)
        torch.sum(ws, dim=0, out=out[m0:m0 + m])
    return out


@functools.lru_cache(maxsize=64)
def compile_router_gemv_wmma(m: int, ks: int = 32):
    """v3: fp32 WMMA (v_wmma_f32_16x16x4_f32). WMMA rows = tokens, cols = experts. A wave owns 16
    experts x one K chunk x all ceil(m/16) token tiles. Per 16-k step a lane loads 8 fp32 W of its
    expert (k0 + 8h .. +7, h = lane // 16) and 8 bf16 x per token tile, then issues 4 WMMAs; instr i
    feeds slots (8h + 2i, 8h + 2i + 1) to both A and B (same k permutation on both sides, see
    wmma_f32_layout_probe.py). Rows >= m load row m-1 and are not stored."""
    from flydsl._mlir.dialects import rocdl as rocdl_raw

    assert 1 <= m <= MAXM_V2 and K % (ks * 16) == 0
    tiles = (m + 15) // 16
    kchunk = K // ks
    steps = kchunk // 16
    n_et = E // 16
    n_waves = n_et * ks
    assert n_waves % WAVES_PER_BLOCK == 0
    n_blocks = n_waves // WAVES_PER_BLOCK

    @flyc.kernel(name=f"laguna_router_gemv_wmma_f32_m{m}_k{ks}", known_block_size=[BLOCK, 1, 1])
    def gemv(x_ptr: fx.Int64, w_ptr: fx.Int64, ws_ptr: fx.Int64):
        load_w, store_f32 = _make_io(fx.Float32)
        load_x, _ = _make_io(fx.BFloat16)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(WAVE)
        gw = fx.Int32(gpu.block_id("x")) * fx.Int32(WAVES_PER_BLOCK) + tid // fx.Int32(WAVE)
        et = gw % fx.Int32(n_et)
        kc = gw // fx.Int32(n_et)
        col = lane % fx.Int32(16)
        half = lane // fx.Int32(16)
        e = et * fx.Int32(16) + col
        kbase = kc * fx.Int32(kchunk) + half * fx.Int32(8)
        rows = []
        for tt in range_constexpr(tiles):
            r = fx.Int32(tt * 16) + col
            rows.append((r < fx.Int32(m)).select(r, fx.Int32(m - 1)))
        v8 = fx.T.vec(8, fx.T.f32)
        zero8 = fx.Vector.from_elements([fx.Float32(0.0) for _ in range(8)], fx.Float32)

        init = [zero8 for _ in range(tiles)]
        for it, acc in range(fx.Int64(0), fx.Int64(steps), fx.Int64(1), init=init):
            acc = list(acc) if isinstance(acc, (list, tuple)) else [acc]
            k = kbase + fx.Int32(it) * fx.Int32(16)
            wrow = e * fx.Int32(K) + k
            w4 = [load_w(w_ptr, wrow + fx.Int32(4 * q), 4) for q in range(2)]
            wv = [w4[j // 4][j % 4] for j in range(8)]
            new = []
            for tt in range_constexpr(tiles):
                xv = load_x(x_ptr, rows[tt] * fx.Int32(K) + k, 8).extf(v8)
                c = acc[tt]
                for i in range_constexpr(4):
                    a = fx.Vector.from_elements([fx.Float32(xv[2 * i]), fx.Float32(xv[2 * i + 1])], fx.Float32)
                    b = fx.Vector.from_elements([wv[2 * i], wv[2 * i + 1]], fx.Float32)
                    c = fx.Vector(rocdl_raw.wmma_f32_16x16x4_f32(v8, a.ir_value(), b.ir_value(), fx.Vector(c).ir_value()).result)
                new.append(c)
            results = yield new
        results = list(results) if isinstance(results, (list, tuple)) else [results]

        for tt in range_constexpr(tiles):
            cv = fx.Vector(results[tt])
            for v in range_constexpr(8):
                mrow = fx.Int32(tt * 16 + v) + half * fx.Int32(8)
                if mrow < fx.Int32(m):
                    store_f32(ws_ptr, (kc * fx.Int32(m) + mrow) * fx.Int32(E) + e, fx.Float32(cv[v]))

    @flyc.jit
    def launch(x_ptr: fx.Int64, w_ptr: fx.Int64, ws_ptr: fx.Int64, stream: fx.Stream):
        gemv(x_ptr, w_ptr, ws_ptr).launch(grid=(n_blocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


def router_gemv_wmma(x, w, ks: int = 32):
    """v3 entry: x [M, K] bf16, w [E, K] fp32 -> logits [M, E] fp32; one launch per 64 tokens."""
    import torch

    assert x.dtype == torch.bfloat16 and w.dtype == torch.float32 and x.is_contiguous() and w.is_contiguous()
    m_all = x.shape[0]
    out = torch.empty(m_all, E, device=x.device, dtype=torch.float32)
    stream = fx.Stream(torch.cuda.current_stream(x.device))
    for m0 in range(0, m_all, MAXM_V2):
        m = min(MAXM_V2, m_all - m0)
        ws = torch.empty(ks, m, E, device=x.device, dtype=torch.float32)
        compile_router_gemv_wmma(m, ks)(x[m0:m0 + m].data_ptr(), w.data_ptr(), ws.data_ptr(), stream)
        torch.sum(ws, dim=0, out=out[m0:m0 + m])
    return out


def router_logits_fp32(x, w):
    """Laguna router logits x[M, K] (bf16) @ w[E, K]^T (fp32) -> [M, E] fp32, or None if not covered.

    M <= 2 takes the VALU path (v2), 3..64 the fp32 WMMA path (v3); choice and ks measured on gfx1250
    (proxy-moe/MI455_RUN_h13-17.md). Callers fall back to F.linear on None (e.g. prefill M > 64)."""
    import torch

    if x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.float32 or tuple(w.shape) != (E, K):
        return None
    m = x.shape[0]
    if m == 0 or m > MAXM_V2:
        return None
    x = x.contiguous()
    if m <= 2:
        return router_gemv_v2(x, w, epw=1, ks=32)
    return router_gemv_wmma(x, w, ks=64)


def router_gemv(x, w, epw: int = 4, ks: int = 8):
    """x [M, K] bf16 contiguous, w [E, K] fp32 contiguous -> logits [M, E] fp32."""
    import torch

    assert x.dtype == torch.bfloat16 and w.dtype == torch.float32 and x.is_contiguous() and w.is_contiguous()
    m_all = x.shape[0]
    out = torch.empty(m_all, E, device=x.device, dtype=torch.float32)
    stream = fx.Stream(torch.cuda.current_stream(x.device))
    for m0 in range(0, m_all, MAXM):
        m = min(MAXM, m_all - m0)
        ws = torch.empty(ks, m, E, device=x.device, dtype=torch.float32)
        compile_router_gemv(m, epw, ks)(x[m0:m0 + m].data_ptr(), w.data_ptr(), ws.data_ptr(), stream)
        torch.sum(ws, dim=0, out=out[m0:m0 + m])
    return out


if __name__ == "__main__":
    import argparse

    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--epw", type=int, default=4)
    ap.add_argument("--ks", type=int, default=8)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v3", action="store_true", help="fp32 WMMA path")
    ap.add_argument("--vec", type=int, default=8)
    a = ap.parse_args()
    if a.v3:
        run = lambda x, w: router_gemv_wmma(x, w, a.ks)
    elif a.v2:
        run = lambda x, w: router_gemv_v2(x, w, a.epw, a.ks, a.vec)
    else:
        run = lambda x, w: router_gemv(x, w, a.epw, a.ks)
    torch.manual_seed(0)
    Ws = [torch.randn(E, K, device="cuda", dtype=torch.float32) * 0.01 for _ in range(a.layers)]

    def timed(fn):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
        ts = []
        for _ in range(20):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / len(Ws))
        return sorted(ts)[len(ts) // 2]

    for m in a.ms:
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        ref64 = x.double() @ Ws[0].double().t()
        top64 = torch.topk(ref64, 16, dim=-1).indices.sort(-1).values
        res = {}
        for name, out in (("F.linear", F.linear(x.float(), Ws[0])), ("flydsl", run(x, Ws[0]))):
            rel = ((out.double() - ref64).norm() / ref64.norm()).item()
            same = (torch.topk(out, 16, dim=-1).indices.sort(-1).values == top64).all(-1).float().mean().item()
            res[name] = f"{name} rel64={rel:.1e} top16={same * 100:5.1f}%"
        t_ref = timed(lambda: [F.linear(x.float(), W) for W in Ws])
        t_new = timed(lambda: [run(x, W) for W in Ws])
        print(f"M={m:4d} [{res['F.linear']} | {res['flydsl']}]  F.linear {t_ref:7.1f} us  flydsl {t_new:6.1f} us  "
              f"({t_ref / t_new:5.1f}x, {E * K * 4 / t_new / 1e6:5.2f} TB/s eff)")
