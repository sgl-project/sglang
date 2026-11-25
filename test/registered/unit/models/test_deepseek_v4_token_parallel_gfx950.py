"""Row layout and collectives of DeepSeek-V4 token-parallel prefill on gfx950.

Four gloo ranks run the module's layout and collective helpers. Every gathered,
reduce-scattered or selected row must equal the full-row reference bit for bit:
the feature's accuracy rests on all gathers being exact, and on the exit gather
returning exactly the tail rows the rest of the forward reads. Summed values are
small integers, so fp32 / bf16 reductions are exact in any order.
"""

import socket
import traceback
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.models.deepseek_common.amd import deepseek_v4_token_parallel as tp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

WORLD = 4


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality, any dtype."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


class _GlooCollectives(tp.Collectives):
    """Exact all-gather of raw bytes; reduce-scatter as an fp32 all-reduce and a
    slice (gloo has no reduce-scatter)."""

    name = "gloo"

    def __init__(self):
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()

    def all_gather(self, out, inp):
        o, i = tp.gather_view(out), tp.gather_view(inp.contiguous())
        o = o.view(o.shape[0], -1).view(torch.uint8)
        i = i.view(i.shape[0], -1).view(torch.uint8)
        parts = list(o.chunk(self.world, 0))
        dist.all_gather(parts, i)
        for k, p in enumerate(parts):
            o[k * i.shape[0] : (k + 1) * i.shape[0]].copy_(p)

    def reduce_scatter(self, out, inp):
        full = inp.float().clone()
        dist.all_reduce(full)
        out.copy_(full.chunk(self.world, 0)[self.rank].to(out.dtype))


class _AuxPacker:
    def __init__(self):
        self.buf = None

    def slot(self, like):
        self.buf = torch.empty_like(like)
        return self.buf


def _full(seed, shape, dtype, ints=False):
    g = torch.Generator().manual_seed(seed)
    if ints:
        return torch.randint(-8, 9, shape, generator=g).to(dtype)
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, generator=g, dtype=torch.int64).to(dtype)
    if dtype == torch.int32:
        return torch.randint(-(2**30), 2**30, shape, generator=g, dtype=torch.int64).to(
            dtype
        )
    if dtype == torch.float8_e4m3fn:
        return (torch.randn(shape, generator=g) * 4).to(dtype)
    return torch.randn(shape, generator=g).to(dtype)


def _state(comm, num_rows):
    return tp.TokenParallelState(num_rows, None, comm, end_layer=20)


def _check_layout(rank, res):
    for T in (2048, 2049, 4095, 12345, 16384, 65, 1):
        lay = tp.RowLayout(T, WORLD, rank)
        spans = [tp.RowLayout(T, WORLD, r).rank_range(r) for r in range(WORLD)]
        covered = [i for lo, hi in spans for i in range(lo, hi)]
        res.append((f"layout T={T} partitions the rows", covered == list(range(T))))
        res.append(
            (
                f"layout T={T} alignment",
                lay.padded % (WORLD * tp.ROW_ALIGN) == 0
                and lay.local % tp.ROW_ALIGN == 0,
            )
        )
        if T >= 2048:
            res.append(
                (f"layout T={T} rows on every rank", all(hi > lo for lo, hi in spans))
            )


def _check_gathers(comm, res):
    for T, dtype, tail in (
        (2049, torch.bfloat16, (4, 8)),
        (2050, torch.float32, (16,)),
        (2051, torch.float8_e4m3fn, (64,)),
        (2052, torch.uint8, (10,)),
        (2053, torch.int32, (7,)),
    ):
        x = _full(T, (T, *tail), dtype)
        st = _state(comm, T)
        got = st.gather(st.scatter(x))
        res.append((f"gather {dtype} T={T}", _same(got[:T], x)))
        res.append(
            (
                f"gather {dtype} T={T} zero padding",
                bool((got[T:].view(torch.uint8) == 0).all()),
            )
        )

    # The attention latent: fp8 q codes, a strided bf16 kv slice and ue8m0 scales.
    T = 3001
    qkv = _full(1, (T, 96), torch.bfloat16)
    q8 = _full(2, (T, 64), torch.float8_e4m3fn)
    sc = _full(3, (T, 2), torch.uint8)
    st = _state(comm, T)
    packer = tp.RowPacker(
        [("q", (64,), q8.dtype), ("kv", (32,), qkv.dtype), ("scale", (2,), sc.dtype)]
    )
    got = st.gather_packed(
        packer,
        {"q": st.scatter(q8), "kv": st.scatter(qkv)[:, 64:], "scale": st.scatter(sc)},
    )
    res.append(("packed row is 16-byte aligned", packer.row_bytes % 16 == 0))
    res.append(("packed q", _same(got["q"][:T].contiguous(), q8)))
    res.append(
        (
            "packed strided kv",
            _same(got["kv"][:T].contiguous(), qkv[:, 64:].contiguous()),
        )
    )
    res.append(("packed scale", _same(got["scale"][:T].contiguous(), sc)))


def _check_reduce_scatter(comm, rank, res):
    for T in (2048, 2051):
        partials = [
            _full(100 + r + T, (T, 24), torch.bfloat16, ints=True) for r in range(WORLD)
        ]
        total = torch.stack([p.float() for p in partials]).sum(0).to(torch.bfloat16)
        st = _state(comm, T)
        out = st.reduce_scatter(partials[rank])
        real = st.hi - st.lo
        res.append((f"reduce_scatter T={T}", _same(out[:real], total[st.lo : st.hi])))
        res.append(
            (f"reduce_scatter T={T} zero padding", bool((out[real:] == 0).all()))
        )


def _check_tail_exit(comm, res):
    ext = [700, 5, 1300, 129, 1, 2000]
    tails = [128, 5, 128, 128, 1, 128]
    rows = tp.tail_global_rows(ext, tails)
    starts = [sum(ext[:b]) for b in range(len(ext))]
    ref_rows = [s + n - t + j for s, n, t in zip(starts, ext, tails) for j in range(t)]
    res.append(("tail_global_rows", rows == ref_rows))

    T, H = sum(ext), 16
    pieces = {
        "residual": _full(7, (T, 4, H), torch.bfloat16),
        "pre": _full(8, (T, 4), torch.float32),
        "input": _full(9, (T, H), torch.bfloat16),
    }
    st = _state(comm, T)
    got = st.gather_selected({k: st.scatter(v) for k, v in pieces.items()}, rows)
    idx = torch.tensor(rows)
    for k, v in pieces.items():
        res.append((f"exit gather {k}", _same(got[k], v.index_select(0, idx))))
    per_rank, cmax = st.layout.selection(rows)
    res.append(("tail rows straddle ranks", sum(1 for p in per_rank if p) >= 2))
    res.append(("exit gather moves only the tail rows", WORLD * cmax < T // 4))
    try:
        st.layout.selection([5, 3])
        res.append(("unsorted rows rejected", False))
    except ValueError:
        res.append(("unsorted rows rejected", True))

    # DSpark capture: the stream mean on local rows, gathered for the draft's rows.
    def mean_fn(x, out):
        out.copy_(x.float().mean(1).to(out.dtype))

    ref = torch.empty((T, H), dtype=torch.bfloat16)
    mean_fn(pieces["residual"], ref)
    hc = SimpleNamespace(residual=st.scatter(pieces["residual"]))
    fb = SimpleNamespace(extend_seq_lens_cpu=ext)
    for name, tail, want in (
        (
            "aux mean, tail rows",
            SimpleNamespace(extend_seq_lens_cpu=tails),
            ref.index_select(0, idx),
        ),
        ("aux mean, all rows", None, ref),
    ):
        packer = _AuxPacker()
        tp.capture_dspark_aux(st, hc, tail, fb, packer, mean_fn=mean_fn)
        res.append((name, _same(packer.buf, want)))


def _check_toy_layer(comm, rank, res):
    """Local projection, exact gather, a row-mixing op on all rows, a TP-sharded
    output projection with reduce-scatter and the tail-row exit equal the
    replicated layer with an all-reduce."""
    T, H, K = 2051, 32, 8
    x = _full(20, (T, H), torch.float32, ints=True)
    w_in = _full(21, (H, K), torch.float32, ints=True)
    w_out = [_full(30 + r, (K, H), torch.float32, ints=True) for r in range(WORLD)]
    ref = torch.cumsum(x @ w_in, 0) @ torch.stack(w_out).sum(0) + x

    st = _state(comm, T)
    x_loc = st.scatter(x)
    lat = st.gather(x_loc @ w_in)[:T]
    out_loc = st.reduce_scatter(torch.cumsum(lat, 0) @ w_out[rank]) + x_loc
    sel = tp.tail_global_rows([1000, 1051], [128, 1051])
    out = st.gather_selected({"h": out_loc}, sel)["h"]
    res.append(
        ("toy layer == replicated", _same(out, ref.index_select(0, torch.tensor(sel))))
    )


def _worker(rank, port, queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD,
        )
        comm = _GlooCollectives()
        res = []
        _check_layout(rank, res)
        _check_gathers(comm, res)
        _check_reduce_scatter(comm, rank, res)
        _check_tail_exit(comm, res)
        _check_toy_layer(comm, rank, res)
        dist.barrier()
        dist.destroy_process_group()
        queue.put((rank, res, None))
    except Exception:
        queue.put((rank, None, traceback.format_exc()))


class TestTokenParallelLayout(unittest.TestCase):
    def test_rows_and_collectives_are_exact(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(r, port, queue)) for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        out = [queue.get(timeout=300) for _ in procs]
        for p in procs:
            p.join(60)
        errors = [e for _, _, e in out if e]
        self.assertFalse(errors, errors[0] if errors else "")
        for rank, res, _ in out:
            for name, ok in res:
                with self.subTest(rank=rank, check=name):
                    self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
