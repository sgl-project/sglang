"""TOP chunked KDA: decode and chain verify through the row engine are bitwise the pinned chunk-64 KDA prefill.

Every rollout KDA output must equal row t of ``top_kda_kernel().extend`` (the trainer's prefill) over the whole sequence
from zero state, and E, the FP32 state at the start of each request's open chunk, must equal that prefill's state at the
same 64-token boundary. Shapes are GLM-5.3-Flash's KDA (64 heads of 128). All comparisons are bitwise.

``ChunkedKDAEngine`` is driven as serving drives one layer: prefills with tails off the 64-token grid, then plain
decode, or (under speculation) chain verify windows of the root and its drafts whose drafts past a random accept
length carry garbage rows, interleaved with spec-mode decode steps, eagerly and under a CUDA graph with padding rows.
The verify convolution is checked against one-token decode updates too.

Radix reuse: every state an extend track or a deferred-close track (``close_to``) hands to the radix cache must be the
prefill's state at that boundary, and a request resumed from one, on a request row whose cache holds another request's
rows, must produce the prefill's rows and states from there on.
"""

import random
import unittest
from contextlib import ExitStack

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

_MODULE_ENV = ExitStack()


def setUpModule():
    # The KDA kernels pin their configs at import under the chunked mode, as the trainer's prefill imports them.
    _MODULE_ENV.enter_context(envs.SGLANG_TOP_KDA_MODE.override("chunked"))


def tearDownModule():
    _MODULE_ENV.close()


H, D, LB = 64, 128, -5.0
C = H * D
CHUNK = 64
dev = "cuda"


def _bits(x):
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class _Base(unittest.TestCase):
    # (prompt length, total length): prompts on and off the grid, one ending at row 63, one a single token.
    REQS = ((100, 300), (64, 200), (3, 140), (127, 260), (1, 90), (190, 330))
    # Added to dt_bias; the default gates decay E below FP32's range within a chunk.
    GATE_BIAS = 0.0

    @classmethod
    def setUpClass(cls):
        from sglang.srt.true_on_policy.kda_chunked import top_kda_kernel

        cls.kern = top_kda_kernel()
        gen = torch.Generator(device=dev).manual_seed(3)

        def rows(n, width, scale=1.0, silu=True):
            x = torch.randn(n, width, device=dev, generator=gen) * scale
            return (torch.nn.functional.silu(x) if silu else x).to(torch.bfloat16)

        cls.rows = staticmethod(rows)
        cls.seqs = []
        for _, n in cls.REQS:
            cls.seqs.append(
                dict(
                    mixed=torch.cat([rows(n, C), rows(n, C), rows(n, C)], -1),
                    f=rows(n, C, 2.0, silu=False),
                    b=rows(n, H, silu=False),
                )
            )
        cls.A_log = torch.randn(H, device=dev, generator=gen) * 0.5
        cls.dt_bias = torch.randn(C, device=dev, generator=gen) * 0.5 + cls.GATE_BIAS
        cls.ref = [cls.prefill(sq, sq["mixed"].shape[0]) for sq in cls.seqs]

    @classmethod
    def prefill(cls, sq, n):
        """The trainer's program over rows [0, n) from zero state: outputs [n, H, D] and the final FP32 state."""
        pool = torch.zeros(1, H, D, D, device=dev)
        if n == 0:
            return None, pool[0]
        q, k, v = (
            sq["mixed"][:n, i * C : (i + 1) * C].reshape(1, n, H, D).clone()
            for i in range(3)
        )
        o = cls.kern.extend(
            q,
            k,
            v,
            sq["f"][:n].reshape(1, n, H, D).contiguous(),
            sq["b"][:n].reshape(1, n, H).contiguous(),
            ssm_states=pool,
            cache_indices=torch.zeros(1, dtype=torch.int32, device=dev),
            query_start_loc=torch.tensor([0, n], dtype=torch.int32, device=dev),
            A_log=cls.A_log.view(1, 1, H, 1),
            dt_bias=cls.dt_bias,
            lower_bound=LB,
            beta_is_raw=True,
        )
        return o[0], pool[0]

    def same(self, name, x, y):
        self.assertEqual(x.shape, y.shape, name)
        if not torch.equal(_bits(x), _bits(y)):
            d = (x.float() - y.float()).abs()
            self.fail(
                f"{name}: max |diff| {d.max().item():.3e}, {(d > 0).sum().item()} of {d.numel()} differ"
            )

    def gates(self):
        return dict(A_log=self.A_log, dt_bias=self.dt_bias, lower_bound=LB)

    def engine(self, *, spec, n):
        from sglang.srt.true_on_policy.kda_chunked import ChunkedKDAEngine

        # Ring slots differ from state slots, as req-pool rows differ from mamba slots in serving.
        self.ring = [5 + 2 * i for i in range(n)]
        self.slot = [n - 1 - i for i in range(n)]
        self.state = torch.zeros(n, H, D, D, device=dev)
        return ChunkedKDAEngine(
            num_layers=1, num_slots=32, H=H, K=D, device=dev, max_bs=32, spec=spec
        )

    def extend_prompts(self, engine):
        items = [(i, p) for i, (p, _) in enumerate(self.REQS)]
        cat = {
            name: torch.cat([self.seqs[i][name][:p] for i, p in items])
            for name in ("mixed", "f", "b")
        }
        T = cat["mixed"].shape[0]
        plan = engine.build_extend_plan(
            prefix=[0] * len(items),
            lens=[p for _, p in items],
            ring=self.ring,
            state=self.slot,
        )
        q, k, v = (
            cat["mixed"][:, i * C : (i + 1) * C].reshape(1, T, H, D).clone()
            for i in range(3)
        )
        o = engine.extend(
            layer=0,
            plan=plan,
            q=q,
            k=k,
            v=v,
            f=cat["f"].reshape(1, T, H, D),
            b=cat["b"].reshape(1, T, H),
            A_log=self.A_log.view(1, 1, H, 1),
            dt_bias=self.dt_bias,
            lower_bound=LB,
            state=self.state,
            cache_indices=torch.tensor(self.slot, dtype=torch.int32, device=dev),
        )
        off = 0
        for i, p in items:
            self.same(f"req {i} prefill rows", o[0, off : off + p], self.ref[i][0][:p])
            off += p
        return [p for _, p in items]

    def window(self, i, pos, accept, nmax):
        sq = self.seqs[i]
        real = slice(pos, pos + accept + 1)
        garbage = nmax - accept - 1
        g = {
            "mixed": torch.cat([self.rows(garbage, C) for _ in range(3)], -1),
            "f": self.rows(garbage, C, 2.0, silu=False),
            "b": self.rows(garbage, H, silu=False),
        }
        return {
            name: torch.cat([sq[name][real], g[name]]) for name in ("mixed", "f", "b")
        }

    def accept_length(self, rng, i, pos, nmax):
        # The sampler never accepts a draft past the root's chunk end (eagle_utils._top_chunked_cut_drafts).
        n = self.REQS[i][1]
        return rng.randint(0, min(nmax - 1, CHUNK - 1 - pos % CHUNK, n - 1 - pos))

    def check_states(self, pos):
        """E of each request equals the prefill's state at the start of the chunk holding its last row."""
        for i, (_, n) in enumerate(self.REQS):
            boundary = (pos[i] - 1) // CHUNK * CHUNK
            self.same(
                f"req {i} E at {boundary}",
                self.state[self.slot[i]],
                self.prefill(self.seqs[i], boundary)[1],
            )


class TestRadixReuse(_Base):
    # Mild gates (about half of E survives a chunk), so a boundary state handed to the radix cache depends on the E
    # its chunk started from.
    GATE_BIAS = -8.0

    def extend_rows(self, engine, items, track=None):
        """One extend of (request, ring, state slot, start, end) rows; with ``track`` (per item, a track length or
        None) returns the track states [items, H, V, K]."""
        cat = {
            name: torch.cat([self.seqs[i][name][p0:p1] for i, _, _, p0, p1 in items])
            for name in ("mixed", "f", "b")
        }
        T = cat["mixed"].shape[0]
        plan = engine.build_extend_plan(
            prefix=[p0 for *_, p0, _ in items],
            lens=[p1 - p0 for *_, p0, p1 in items],
            ring=[ring for _, ring, *_ in items],
            state=[slot for _, _, slot, *_ in items],
            track_mask=None if track is None else [t is not None for t in track],
            track_lens=None if track is None else [t or -1 for t in track],
        )
        track_state = (
            None if track is None else self.state.new_zeros(len(items), H, D, D)
        )
        q, k, v = (
            cat["mixed"][:, j * C : (j + 1) * C].reshape(1, T, H, D).clone()
            for j in range(3)
        )
        o = engine.extend(
            layer=0,
            plan=plan,
            q=q,
            k=k,
            v=v,
            f=cat["f"].reshape(1, T, H, D),
            b=cat["b"].reshape(1, T, H),
            A_log=self.A_log.view(1, 1, H, 1),
            dt_bias=self.dt_bias,
            lower_bound=LB,
            state=self.state,
            cache_indices=torch.tensor(
                [slot for _, _, slot, *_ in items], dtype=torch.int32, device=dev
            ),
            track_state=track_state,
        )
        off = 0
        for i, _, _, p0, p1 in items:
            self.same(
                f"req {i} rows {p0}..{p1}",
                o[0, off : off + p1 - p0],
                self.ref[i][0][p0:p1],
            )
            off += p1 - p0
        return track_state

    def decode_rows(self, engine, i, *, ring, slot, start, end):
        for pos in range(start, end):
            step = engine.build_step(
                positions=torch.tensor([pos], device=dev),
                ring=torch.tensor([ring], device=dev),
                state=torch.tensor([slot], device=dev),
                bs=1,
                static=False,
            )
            o = engine.decode(
                layer=0,
                step=step,
                mixed_qkv=self.seqs[i]["mixed"][pos : pos + 1],
                f=self.seqs[i]["f"][pos : pos + 1],
                b=self.seqs[i]["b"][pos : pos + 1],
                state=self.state,
                **self.gates(),
            )
            self.same(f"req {i} row {pos}", o[0], self.ref[i][0][pos])

    def state_at(self, i, pos):
        return self.prefill(self.seqs[i], pos)[1]

    def test_extend_tracks_and_resumed_requests(self):
        """Extend tracks inside the prefill call and at its final boundary; requests resumed from radix states, one
        with tail rows only (a retraction's re-entry), one prefilling chunks, on request rows another request used."""
        engine = self.engine(spec=False, n=len(self.REQS))
        self.state = torch.zeros(6, H, D, D, device=dev)
        # Request 5 extends [0, 300) tracked at 192, inside its prefill (256 is its last boundary); request 0 extends
        # [0, 200) tracked at its last boundary 192 (the forced-h length 193 names the final state).
        tracks = self.extend_rows(
            engine, [(5, 5, 0, 0, 300), (0, 7, 1, 0, 200)], track=[193, 193]
        )
        self.same("track inside the prefill", tracks[0], self.state_at(5, 192))
        self.same("track at the final boundary", tracks[1], self.state_at(0, 192))
        # Request 0 decodes on, so request row 7 caches its rows of chunk 3.
        self.decode_rows(engine, 0, ring=7, slot=1, start=200, end=230)
        # Request 5 again hits the node at 192 on request row 7 and a fresh slot; its extend has tail rows only.
        self.state[2] = tracks[0]
        self.extend_rows(engine, [(5, 7, 2, 192, 250)])
        self.decode_rows(engine, 5, ring=7, slot=2, start=250, end=330)
        self.same("resumed request 5 E", self.state[2], self.state_at(5, 320))
        # Request 3 resumes at 64 on request row 5 (request 5's first row) and prefills chunks before its tail.
        self.state[3] = self.state_at(3, 64)
        self.extend_rows(engine, [(3, 5, 3, 64, 200)])
        self.decode_rows(engine, 3, ring=5, slot=3, start=200, end=260)
        self.same("resumed request 3 E", self.state[3], self.state_at(3, 256))

    def test_spec_tracks_close_into_their_slot(self):
        """Under speculation a track whose committed rows end on row 63 (a verify or a spec-mode decode) closes the
        chunk into its track slot; E stays at the chunk start, and a request resumed from that slot is the prefill."""
        nmax, rng = 6, random.Random(5)
        n = len(self.REQS)
        engine = self.engine(spec=True, n=n)
        self.state = torch.zeros(n + 16, H, D, D, device=dev)
        free = list(range(n, n + 16))
        pos = self.extend_prompts(engine)
        T = [t for _, t in self.REQS]
        tracked, step_id = [], 0
        while any(pos[i] < T[i] for i in range(n)):
            live = [i for i in range(n) if pos[i] < T[i]]
            step_id += 1
            ring = torch.tensor([self.ring[i] for i in live], device=dev)
            state = torch.tensor([self.slot[i] for i in live], device=dev)
            if step_id % 5 == 0:
                step = engine.build_step(
                    positions=torch.tensor([pos[i] for i in live], device=dev),
                    ring=ring,
                    state=state,
                    bs=len(live),
                    static=False,
                )
                engine.decode(
                    layer=0,
                    step=step,
                    mixed_qkv=torch.stack(
                        [self.seqs[i]["mixed"][pos[i]] for i in live]
                    ),
                    f=torch.stack([self.seqs[i]["f"][pos[i]] for i in live]),
                    b=torch.stack([self.seqs[i]["b"][pos[i]] for i in live]),
                    state=self.state,
                    **self.gates(),
                )
                ends = [pos[i] + 1 for i in live]
            else:
                accepts = []
                for i in live:
                    to_end = CHUNK - 1 - pos[i] % CHUNK
                    last = T[i] - 1 - pos[i]
                    # Land on row 63 whenever the window reaches it, so tracks fire.
                    a = to_end if to_end < nmax and to_end <= last else None
                    accepts.append(
                        a if a is not None else self.accept_length(rng, i, pos[i], nmax)
                    )
                wins = [self.window(i, pos[i], a, nmax) for i, a in zip(live, accepts)]
                step = engine.build_step(
                    positions=torch.tensor(
                        [pos[i] + j for i in live for j in range(nmax)], device=dev
                    ),
                    ring=ring,
                    state=state,
                    bs=len(live),
                    static=False,
                    nmax=nmax,
                )
                o = engine.verify(
                    layer=0,
                    step=step,
                    mixed_qkv=torch.cat([w["mixed"] for w in wins]),
                    f=torch.cat([w["f"] for w in wins]),
                    b=torch.cat([w["b"] for w in wins]),
                    state=self.state,
                    **self.gates(),
                ).view(len(live), nmax, H, D)
                for j, (i, a) in enumerate(zip(live, accepts)):
                    self.same(
                        f"req {i} verify rows",
                        o[j, : a + 1],
                        self.ref[i][0][pos[i] : pos[i] + a + 1],
                    )
                ends = [pos[i] + a + 1 for i, a in zip(live, accepts)]
            dst = [-1] * len(live)
            for j, (i, end) in enumerate(zip(live, ends)):
                if end % CHUNK == 0 and free:
                    dst[j] = free.pop()
                    tracked.append((i, end, dst[j]))
            if any(d >= 0 for d in dst):
                engine.close_to(
                    state=self.state.unsqueeze(0),
                    ring=ring,
                    slots=state,
                    dst=torch.tensor(dst, device=dev),
                    gates=[(self.A_log, self.dt_bias, LB)],
                )
            for i, end in zip(live, ends):
                pos[i] = end
        self.assertGreaterEqual(len(tracked), 4)
        for i, end, slot in tracked:
            self.same(
                f"req {i} track at {end}", self.state[slot], self.state_at(i, end)
            )
        self.check_states(pos)
        # A new request resumes a decode track on a request row no request used.
        i, end, slot = next(t for t in tracked if t[1] + 70 <= T[t[0]])
        self.extend_rows(engine, [(i, 30, slot, end, end + 70)])
        self.decode_rows(engine, i, ring=30, slot=slot, start=end + 70, end=T[i])


class TestDecode(_Base):
    def test_decode_rows_and_states(self):
        engine = self.engine(spec=False, n=len(self.REQS))
        pos = self.extend_prompts(engine)
        T = [n for _, n in self.REQS]
        while True:
            live = [i for i in range(len(T)) if pos[i] < T[i]]
            if not live:
                break
            step = engine.build_step(
                positions=torch.tensor([pos[i] for i in live], device=dev),
                ring=torch.tensor([self.ring[i] for i in live], device=dev),
                state=torch.tensor([self.slot[i] for i in live], device=dev),
                bs=len(live),
                static=False,
            )
            o = engine.decode(
                layer=0,
                step=step,
                mixed_qkv=torch.stack([self.seqs[i]["mixed"][pos[i]] for i in live]),
                f=torch.stack([self.seqs[i]["f"][pos[i]] for i in live]),
                b=torch.stack([self.seqs[i]["b"][pos[i]] for i in live]),
                state=self.state,
                **self.gates(),
            )
            for j, i in enumerate(live):
                self.same(f"req {i} row {pos[i]}", o[j], self.ref[i][0][pos[i]])
                pos[i] += 1
        self.check_states(pos)


class TestChainVerify(_Base):
    """Verify windows of the root and nmax - 1 drafts: rows up to a random accept length are the sequence's, the rest
    garbage (rejected drafts); every committed row and E must be the prefill's."""

    def run_eager(self, nmax, seed):
        rng = random.Random(seed)
        engine = self.engine(spec=True, n=len(self.REQS))
        pos = self.extend_prompts(engine)
        T = [n for _, n in self.REQS]
        step_id = 0
        while True:
            live = [i for i in range(len(T)) if pos[i] < T[i]]
            if not live:
                break
            step_id += 1
            ring = torch.tensor([self.ring[i] for i in live], device=dev)
            state = torch.tensor([self.slot[i] for i in live], device=dev)
            if step_id % 5 == 0:
                # A spec-mode decode step defers its close like a verify.
                step = engine.build_step(
                    positions=torch.tensor([pos[i] for i in live], device=dev),
                    ring=ring,
                    state=state,
                    bs=len(live),
                    static=False,
                )
                o = engine.decode(
                    layer=0,
                    step=step,
                    mixed_qkv=torch.stack(
                        [self.seqs[i]["mixed"][pos[i]] for i in live]
                    ),
                    f=torch.stack([self.seqs[i]["f"][pos[i]] for i in live]),
                    b=torch.stack([self.seqs[i]["b"][pos[i]] for i in live]),
                    state=self.state,
                    **self.gates(),
                )
                for j, i in enumerate(live):
                    self.same(
                        f"req {i} decode row {pos[i]}", o[j], self.ref[i][0][pos[i]]
                    )
                    pos[i] += 1
                continue
            accepts = [self.accept_length(rng, i, pos[i], nmax) for i in live]
            wins = [self.window(i, pos[i], a, nmax) for i, a in zip(live, accepts)]
            step = engine.build_step(
                positions=torch.tensor(
                    [pos[i] + j for i in live for j in range(nmax)], device=dev
                ),
                ring=ring,
                state=state,
                bs=len(live),
                static=False,
                nmax=nmax,
            )
            o = engine.verify(
                layer=0,
                step=step,
                mixed_qkv=torch.cat([w["mixed"] for w in wins]),
                f=torch.cat([w["f"] for w in wins]),
                b=torch.cat([w["b"] for w in wins]),
                state=self.state,
                **self.gates(),
            ).view(len(live), nmax, H, D)
            for j, (i, a) in enumerate(zip(live, accepts)):
                self.same(
                    f"req {i} verify rows {pos[i]}..{pos[i] + a}",
                    o[j, : a + 1],
                    self.ref[i][0][pos[i] : pos[i] + a + 1],
                )
                past = CHUNK - pos[i] % CHUNK
                if past < nmax:
                    self.assertEqual(
                        o[j, past:].abs().max().item(),
                        0.0,
                        "rows past the chunk end are zeros",
                    )
                pos[i] += a + 1
        self.check_states(pos)

    def test_two_token_windows(self):
        self.run_eager(nmax=2, seed=0)

    def test_six_token_windows(self):
        self.run_eager(nmax=6, seed=1)

    def test_six_token_windows_under_a_cuda_graph(self):
        """The serving shape: one captured verify at a fixed batch, coordinates refreshed into the graph buffers each
        step, finished requests replayed as padding rows (req-pool row 0, mamba slot -1)."""
        nmax, rng = 6, random.Random(2)
        n = len(self.REQS)
        engine = self.engine(spec=True, n=n)
        pos = self.extend_prompts(engine)
        T = [t for _, t in self.REQS]
        engine.alloc_graph_bufs(n, nmax)
        static = {
            "mixed": torch.zeros(n * nmax, 3 * C, dtype=torch.bfloat16, device=dev),
            "f": torch.zeros(n * nmax, C, dtype=torch.bfloat16, device=dev),
            "b": torch.zeros(n * nmax, H, dtype=torch.bfloat16, device=dev),
        }
        coords = {
            "positions": torch.zeros(n * nmax, dtype=torch.int64, device=dev),
            "ring": torch.zeros(n, dtype=torch.int64, device=dev),
            "state": torch.full((n,), -1, dtype=torch.int32, device=dev),
        }

        def refresh():
            return engine.build_step(**coords, bs=n, static=True, nmax=nmax)

        def run(step):
            return engine.verify(
                layer=0,
                step=step,
                mixed_qkv=static["mixed"],
                f=static["f"],
                b=static["b"],
                state=self.state,
                **self.gates(),
            )

        # Warm up (compile) on padding rows only: they touch the pad slot and no state.
        step = refresh()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(step)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(step)
        while any(pos[i] < T[i] for i in range(n)):
            accepts = {}
            for i in range(n):
                rows = slice(i * nmax, (i + 1) * nmax)
                if pos[i] >= T[i]:
                    coords["positions"][rows] = 0
                    coords["ring"][i] = 0
                    coords["state"][i] = -1
                    for name in static:
                        static[name][rows] = 0
                    continue
                accepts[i] = self.accept_length(rng, i, pos[i], nmax)
                w = self.window(i, pos[i], accepts[i], nmax)
                coords["positions"][rows] = torch.arange(
                    pos[i], pos[i] + nmax, device=dev
                )
                coords["ring"][i] = self.ring[i]
                coords["state"][i] = self.slot[i]
                for name in static:
                    static[name][rows] = w[name]
            refresh()
            graph.replay()
            o = out.view(n, nmax, H, D)
            for i, a in accepts.items():
                self.same(
                    f"req {i} graph verify rows {pos[i]}..{pos[i] + a}",
                    o[i, : a + 1],
                    self.ref[i][0][pos[i] : pos[i] + a + 1],
                )
                pos[i] += a + 1
        self.check_states(pos)


class TestVerifyConvolution(_Base):
    def test_window_conv_equals_decode_updates(self):
        """causal_conv1d_update over a verify window (with its intermediate windows) equals one-token updates."""
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

        bs, nmax, width, dim = 4, 6, 4, 3 * C
        gen = torch.Generator(device=dev).manual_seed(7)
        weight = (torch.randn(dim, width, device=dev, generator=gen) * 0.5).to(
            torch.bfloat16
        )
        bias = (torch.randn(dim, device=dev, generator=gen) * 0.1).to(torch.bfloat16)
        x = torch.randn(bs, nmax, dim, device=dev, generator=gen).to(torch.bfloat16)
        conv0 = torch.randn(bs + 1, width - 1, dim, device=dev, generator=gen).to(
            torch.bfloat16
        )
        idx = torch.arange(1, bs + 1, dtype=torch.int32, device=dev)

        conv = conv0.clone()
        windows = torch.zeros(
            bs + 1, nmax, width - 1, dim, dtype=torch.bfloat16, device=dev
        )
        got = causal_conv1d_update(
            x.transpose(1, 2),
            conv.transpose(-1, -2),
            weight,
            bias,
            activation="silu",
            conv_state_indices=idx,
            intermediate_conv_window=windows.transpose(-1, -2),
            intermediate_state_indices=idx,
        ).transpose(1, 2)

        conv = conv0.clone()
        for j in range(nmax):
            want = causal_conv1d_update(
                x[:, j].contiguous(),
                conv.transpose(-1, -2),
                weight,
                bias,
                activation="silu",
                conv_state_indices=idx,
            )
            self.same(f"depth {j}", got[:, j], want)


if __name__ == "__main__":
    unittest.main()
