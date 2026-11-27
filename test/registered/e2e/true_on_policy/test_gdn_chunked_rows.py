"""TOP chunked GDN: the row engine and its serving driver are bitwise FLA's fixed-tile chunk-64 prefill.

Every rollout GDN output must equal row t of ``chunk_gated_delta_rule(fixed_output_tile=True)`` over the whole
sequence from zero state (the trainer's plain prefill), and every FP32 state handed to the radix cache must equal
that prefill's state at the same 64-token boundary. Shapes are Qwen3.5-35B-A3B's GDN (16 key heads, 32 value
heads, 128-wide). All comparisons are bitwise (torch.equal on the raw bits).

Kernel tests check one-row decode, chain verify with garbage rejected drafts, deferred close and continuation
against that prefill; the engine tests drive ``ChunkedGDNEngine`` (what the serving backend runs per
layer) through prefill segments, decode, chain verify, radix track snapshots, restores and off-grid continuations.
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
    # The kkt config is pinned at import under the chunked mode; the trainer's prefill imports the kernels the same way,
    # and every FLA import below runs after this.
    _MODULE_ENV.enter_context(envs.SGLANG_TOP_GDN_MODE.override("chunked"))


def tearDownModule():
    _MODULE_ENV.close()


Hg, H, K, V = 16, 32, 128, 128
D = 2 * Hg * K + H * V
dev = "cuda"


def _bits(x):
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class _Base(unittest.TestCase):
    # Gates: A_log ~ U(A_LOG), a ~ N(0, A_STD). The default decays a chunk's state to almost nothing; a chunk decay
    # near 1 (TestEngineWeakDecay) is where the close's last bit depends on its exact rounding (fma in the state step).
    A_LOG = (-1.0, 1.0)
    A_STD = 2.0

    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
            fused_gdn_gating,
        )

        cls.fla = staticmethod(chunk_gated_delta_rule)
        cls.gating = staticmethod(fused_gdn_gating)
        torch.manual_seed(0)
        lo, hi = cls.A_LOG
        cls.A_log = (torch.rand(H, device=dev) * (hi - lo) + lo).float()
        cls.dt_bias = (torch.randn(H, device=dev) * 0.5).to(torch.bfloat16)

    def same(self, name, x, y):
        self.assertEqual(x.shape, y.shape, name)
        if not torch.equal(_bits(x), _bits(y)):
            d = (x.float() - y.float()).abs()
            self.fail(
                f"{name}: max |diff| {d.max().item():.3e}, {(d > 0).sum().item()} of {d.numel()} differ"
            )

    @classmethod
    def inputs(cls, L, seed):
        gen = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(L, D, device=dev, generator=gen).to(torch.bfloat16)
        a = (torch.randn(L, H, device=dev, generator=gen) * cls.A_STD).to(
            torch.bfloat16
        )
        b = (torch.randn(L, H, device=dev, generator=gen) * 2).to(torch.bfloat16)
        return x, a, b

    @staticmethod
    def split(x):
        L = x.shape[0]
        return (
            x[:, : Hg * K].reshape(1, L, Hg, K),
            x[:, Hg * K : 2 * Hg * K].reshape(1, L, Hg, K),
            x[:, 2 * Hg * K :].reshape(1, L, H, V),
        )

    def prefill(self, x, a, b):
        """The trainer's program: FLA prefill of one sequence from zero; outputs [L, H, V], final FP32 state."""
        L = x.shape[0]
        q, k, v = self.split(x)
        g, beta = self.gating(self.A_log, a, b, self.dt_bias)
        pool = torch.zeros(1, H, V, K, device=dev)
        o = self.fla(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            initial_state_indices=torch.zeros(1, dtype=torch.int32, device=dev),
            cu_seqlens=torch.tensor([0, L], device=dev, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            fixed_output_tile=True,
        )[0]
        return o[0], pool[0]


class TestKktPinned(_Base):
    def test_one_config(self):
        from sglang.kernels.ops.attention.fla import chunk_rows
        from sglang.kernels.ops.attention.fla.chunk_fwd import TOP_CHUNKED_KKT_CONFIG

        tuner = chunk_rows.kkt_autotuner()
        self.assertEqual(len(tuner.configs), 1)
        self.assertEqual(chunk_rows.kkt_config(), TOP_CHUNKED_KKT_CONFIG)


class TestNegativeControl(_Base):
    def test_native_decode_differs(self):
        """The comparison has teeth: SGLang's native recurrent decode, from the prefill's exact FP32 state, does not
        reproduce the prefill's rows (TOP rollout never runs it)."""
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        P, N = 100, 40
        x, a, b = self.inputs(P + N, 3)
        ref, _ = self.prefill(x, a, b)
        state = self.prefill(x[:P], a[:P], b[:P])[1][None].clone()
        differ = 0
        for t in range(P, P + N):
            out = torch.empty(1, 1, H, V, device=dev, dtype=torch.bfloat16)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=x[t : t + 1],
                a=a[t : t + 1],
                b=b[t : t + 1],
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=K**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=torch.zeros(1, dtype=torch.int32, device=dev),
                use_qk_l2norm_in_kernel=True,
            )
            differ += not torch.equal(_bits(out[0, 0]), _bits(ref[t]))
        self.assertGreater(differ, N // 2)


class _Seq:
    """One request's sequence, reference and engine coordinates."""

    def __init__(self, test, L, seed, ring, slot):
        self.x, self.a, self.b = test.inputs(L, seed)
        self.ref, _ = test.prefill(self.x, self.a, self.b)
        self.ring, self.slot = ring, slot
        self.pos = 0
        self.test = test

    def state_at(self, n):
        return self.test.prefill(self.x[:n], self.a[:n], self.b[:n])[1]


class TestEngine(_Base):
    """``ChunkedGDNEngine`` as the backend drives it, one GDN layer."""

    def engine(self, spec, nmax=1, slots=12, reqs=10):
        from sglang.srt.true_on_policy.gdn_chunked_engine import ChunkedGDNEngine

        eng = ChunkedGDNEngine(
            num_layers=1,
            num_slots=reqs,
            Hg=Hg,
            H=H,
            K=K,
            V=V,
            device=dev,
            spec=spec,
            nmax=nmax,
            max_bs=reqs,
        )
        state = torch.zeros(1, slots, H, V, K, device=dev)
        return eng, state

    def extend(self, eng, state, seqs, lens, track=None):
        """One extend forward over ``seqs`` (each from its current position), as the backend runs it; checks the
        outputs and returns the track states [batch, H, V, K] when ``track`` (per request: track length or None)."""
        prefix = [s.pos for s in seqs]
        xs = [s.x[s.pos : s.pos + n] for s, n in zip(seqs, lens)]
        x = torch.cat(xs)
        a = torch.cat([s.a[s.pos : s.pos + n] for s, n in zip(seqs, lens)])
        b = torch.cat([s.b[s.pos : s.pos + n] for s, n in zip(seqs, lens)])
        q, k, v = self.split(x)
        g, beta = self.gating(self.A_log, a, b, self.dt_bias)
        for s in seqs:
            if s.pos == 0:
                state[0, s.slot].zero_()  # SGLang clears a fresh mamba slot
        plan = eng.build_extend_plan(
            prefix=prefix,
            lens=lens,
            ring=[s.ring for s in seqs],
            state=[s.slot for s in seqs],
            track_mask=None if track is None else [t is not None for t in track],
            track_lens=None if track is None else [t or 0 for t in track],
        )
        track_state = None
        if track is not None:
            track_state = torch.zeros(len(seqs), H, V, K, device=dev)
        cache_indices = torch.tensor(
            [s.slot for s in seqs], device=dev, dtype=torch.int32
        )
        out = eng.extend(
            0,
            plan,
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g,
            beta,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state[0],
            cache_indices,
            track_state,
        )[0]
        r = 0
        for s, n in zip(seqs, lens):
            self.same(
                f"extend rows {s.pos}..{s.pos + n}",
                out[r : r + n],
                s.ref[s.pos : s.pos + n],
            )
            r += n
            s.pos += n
        return track_state

    def decode(self, eng, state, seqs, steps, track_slots=None):
        """``steps`` decode forwards of every request in ``seqs``; with ``track_slots`` (per request), a decode
        track copies the boundary state there when a token lands on row 63 (checked against the prefill)."""
        for _ in range(steps):
            pos = torch.tensor([s.pos for s in seqs], device=dev)
            step = eng.build_step(
                positions=pos,
                ring=torch.tensor([s.ring for s in seqs], device=dev),
                state=torch.tensor([s.slot for s in seqs], device=dev),
                bs=len(seqs),
                nmax=1,
            )
            x = torch.stack([s.x[s.pos] for s in seqs])
            a = torch.stack([s.a[s.pos] for s in seqs])
            b = torch.stack([s.b[s.pos] for s in seqs])
            out = eng.decode(0, step, x, a, b, self.A_log, self.dt_bias, state[0])
            for i, s in enumerate(seqs):
                self.same(f"decode t={s.pos}", out[i], s.ref[s.pos])
                s.pos += 1
            if track_slots is not None:
                done = [s.pos % 64 == 0 for s in seqs]
                if any(done):
                    if eng.spec:
                        dst = torch.tensor(
                            [t if d else -1 for t, d in zip(track_slots, done)],
                            device=dev,
                        )
                        eng.close_to(
                            state=state, ring=step.ring, slots=step.state, dst=dst
                        )
                    else:
                        for t, d, s in zip(track_slots, done, seqs):
                            if d:
                                state[0, t] = state[0, s.slot]
                    for t, d, s in zip(track_slots, done, seqs):
                        if d:
                            self.same(
                                f"decode track at {s.pos}",
                                state[0, t],
                                s.state_at(s.pos),
                            )

    def verify(self, eng, state, seqs, ends, nmax, rng, track_slots=None):
        """Chain verify until every request reaches its end: root + nmax - 1 drafts per step, a random number
        accepted, garbage inputs on rejected drafts; the commit only advances the position (and, crossing a
        boundary, closes the chunk into the request's track slot)."""
        while any(s.pos < e for s, e in zip(seqs, ends)):
            x = torch.zeros(len(seqs) * nmax, D, device=dev, dtype=torch.bfloat16)
            a = torch.zeros(len(seqs) * nmax, H, device=dev, dtype=torch.bfloat16)
            b = torch.zeros_like(a)
            accepted = []
            for i, (s, e) in enumerate(zip(seqs, ends)):
                n = min(nmax, e - s.pos)
                window = min(nmax, 64 - s.pos % 64)
                acc = rng.randint(1, max(1, min(n, window))) if s.pos < e else 0
                rows = slice(i * nmax, i * nmax + nmax)
                gen = torch.Generator(device=dev).manual_seed(rng.randint(0, 1 << 30))
                x[rows] = torch.randn(nmax, D, device=dev, generator=gen).to(
                    torch.bfloat16
                )
                a[rows] = torch.randn(nmax, H, device=dev, generator=gen).to(
                    torch.bfloat16
                )
                b[rows] = torch.randn(nmax, H, device=dev, generator=gen).to(
                    torch.bfloat16
                )
                m = min(acc, nmax)
                x[i * nmax : i * nmax + m] = s.x[s.pos : s.pos + m]
                a[i * nmax : i * nmax + m] = s.a[s.pos : s.pos + m]
                b[i * nmax : i * nmax + m] = s.b[s.pos : s.pos + m]
                accepted.append(acc)
            positions = torch.tensor(
                [s.pos + j for s in seqs for j in range(nmax)], device=dev
            )
            step = eng.build_step(
                positions=positions,
                ring=torch.tensor([s.ring for s in seqs], device=dev),
                state=torch.tensor([s.slot for s in seqs], device=dev),
                bs=len(seqs),
                nmax=nmax,
            )
            # SGLang's strided verify split: q, k, v are views of the packed rows.
            q, k, v = x.split([Hg * K, Hg * K, H * V], dim=-1)
            from sglang.srt.true_on_policy.gdn_chunked_engine import strided_mixed

            mixed = strided_mixed(
                q.view(1, -1, Hg, K), k.view(1, -1, Hg, K), v.view(1, -1, H, V)
            )
            self.assertEqual(mixed.data_ptr(), x.data_ptr())
            out = eng.verify(0, step, mixed, a, b, self.A_log, self.dt_bias, state[0])
            dst = []
            for i, (s, acc) in enumerate(zip(seqs, accepted)):
                if acc:
                    self.same(
                        f"verify rows {s.pos}..{s.pos + acc}",
                        out[i * nmax : i * nmax + acc],
                        s.ref[s.pos : s.pos + acc],
                    )
                s.pos += acc
                dst.append(
                    track_slots[i]
                    if track_slots is not None and acc and s.pos % 64 == 0
                    else -1
                )
            if any(d >= 0 for d in dst):
                eng.close_to(
                    state=state,
                    ring=step.ring,
                    slots=step.state,
                    dst=torch.tensor(dst, device=dev),
                )
                for s, d in zip(seqs, dst):
                    if d >= 0:
                        self.same(
                            f"verify track at {s.pos}", state[0, d], s.state_at(s.pos)
                        )

    def test_prefill_segments_and_decode(self):
        eng, state = self.engine(spec=False)
        seqs = [_Seq(self, 600, 1 + i, ring=1 + i, slot=2 + i) for i in range(4)]
        # Fresh prompts of every alignment in one extend, then chunked-prefill segments (64-aligned) with tails.
        self.extend(eng, state, seqs, [100, 64, 1, 130])
        self.extend(eng, state, seqs[1:2], [128])
        self.decode(eng, state, seqs, 140, track_slots=[8, 9, 10, 11])
        # The slot of a decoded request holds E at its open chunk's start.
        for s in seqs:
            self.same(
                f"E at {s.pos // 64 * 64}",
                state[0, s.slot],
                s.state_at(s.pos // 64 * 64),
            )

    def test_radix_track_and_restore(self):
        eng, state = self.engine(spec=False)
        s = _Seq(self, 700, 7, ring=1, slot=2)
        # Track at the extend's last boundary (final state) and at an earlier grid point (inside the prefill).
        tracks = self.extend(
            eng, state, [s], [300], track=[257]
        )  # (257 - 0) % 64 != 0: state at 256
        self.same("track 256", tracks[0], s.state_at(256))
        t = _Seq(self, 700, 8, ring=2, slot=3)
        tracks = self.extend(eng, state, [t], [200], track=[129])
        self.same("track 128", tracks[0], t.state_at(128))
        # A new request matching the radix node at 256: its slot gets the node's state, then extends and decodes.
        r = _Seq(self, 700, 7, ring=3, slot=4)
        state[0, r.slot] = s.state_at(256)
        r.pos = 256
        self.extend(eng, state, [r], [70])
        self.decode(eng, state, [r], 80)

    def test_turn_ends_on_track_boundary(self):
        """A turn whose last token lands on a 128-token track boundary: the decode step after it, still in flight
        when the turn finishes, closes the chunk at row 63 and tracks into the ping-pong slot that is not the radix
        checkpoint at end - 128; the next turn reuses the request row and slot, restores that checkpoint and must
        stay the prefill over the whole sequence."""
        eng, state = self.engine(spec=False)
        s = _Seq(self, 600, 9, ring=1, slot=2)
        self.extend(eng, state, [s], [200])
        self.decode(eng, state, [s], 256 - 200, track_slots=[8])
        # The turn ends at 384: its last forward runs token 382, and the in-flight step token 383.
        self.decode(eng, state, [s], 384 - 256, track_slots=[9])
        self.same("checkpoint at 256", state[0, 8], s.state_at(256))
        r = _Seq(self, 600, 9, ring=1, slot=2)
        state[0, r.slot] = state[0, 8]
        r.pos = 256
        self.extend(eng, state, [r], [430 - 256])
        self.decode(eng, state, [r], 70)

    def test_spec_verify(self):
        for nmax in (4, 2):
            eng, state = self.engine(spec=True, nmax=nmax)
            rng = random.Random(nmax)
            seqs = [_Seq(self, 400, 20 + i, ring=1 + i, slot=1 + i) for i in range(5)]
            self.extend(eng, state, seqs, [1, 13, 63, 64, 200])
            self.verify(
                eng, state, seqs, [300] * 5, nmax, rng, track_slots=[6, 7, 8, 9, 10]
            )
            # Plain decode steps under speculation run chunk_rows2 with one row and the deferred close.
            self.decode(eng, state, seqs, 40, track_slots=[6, 7, 8, 9, 10])

    def test_off_grid_continuation(self):
        """A request extending again from its own open chunk (prefix off the 64-token grid): chunk prefill over the
        cached open-chunk rows plus the new rows from E, then the tail through the row engine."""
        eng, state = self.engine(spec=True, nmax=4)
        rng = random.Random(5)
        seqs = [_Seq(self, 600, 40 + i, ring=1 + i, slot=1 + i) for i in range(3)]
        self.extend(eng, state, seqs, [100, 70, 64])
        self.verify(eng, state, seqs, [128, 110, 90], 4, rng)
        # 110 / 90: off the grid, mixed in one extend with a fresh request.
        fresh = _Seq(self, 600, 50, ring=4, slot=4)
        self.extend(eng, state, seqs[1:] + [fresh], [30, 100, 77])
        self.verify(eng, state, seqs[1:] + [fresh], [300, 300, 300], 4, rng)


class TestEngineWeakDecay(TestEngine):
    A_LOG = (-4.0, -2.0)
    A_STD = 0.5


class TestEngineStrongDecay(TestEngine):
    A_LOG = (1.0, 4.0)
    A_STD = 6.0


if __name__ == "__main__":
    unittest.main()
