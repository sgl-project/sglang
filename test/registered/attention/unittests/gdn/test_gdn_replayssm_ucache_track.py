"""Radix decode-time track snapshots under --enable-gdn-replayssm-ucache.

After a verify's cursor commit, every row whose accepted tokens cross a track
point gets fold(S0, window up to that point) in its track slot, through FlashInfer's
gdn_prefix_materialize. The snapshot is checked against an fp32 token-by-token
recurrence on folded and unfolded rows at every (accept, step, window length, ring
base); the single flattened launch against per-layer launches; CUDA graph replay
against eager; and a request resumed from a snapshot against the original request.
"""

import os

# FlashInfer fixes the materialize kernel's state dtype at the first import of
# flashinfer.gdn_kernels; serving sets this in the server-args hook.
os.environ.setdefault("GDN_UCACHE_STATE_DTYPE", "fp16")

import unittest  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
    (10, 7),
):
    pytest.skip(
        "FlashInfer u-cache verify requires SM100/SM103/SM107", allow_module_level=True
    )
pytest.importorskip("flashinfer.gdn_kernels.gdn_replay_mtp_umma")
pytest.importorskip("flashinfer.gdn_kernels.gdn_prefix_materialize")

from sglang.srt.layers.attention.linear.gdn_replayssm_ucache import (  # noqa: E402
    GdnUcacheTrackMaterializer,
    GdnUcacheVerifier,
    allocate_gdn_ucache,
    commit_gdn_ucache_cursors,
    reset_gdn_ucache_cursors,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402


def setUpModule():
    # GdnUcacheVerifier reads the exec config when it starts; publish a minimal one.
    get_context().set_server_args(ServerArgs(model_path="dummy"))


K = V = 128
T = 4
# FlashInfer's materialize bound against fp32 (tests/gdn/test_gdn_prefix_materialize.py).
STATE_TOL = 2e-2


class _Model:
    """A GDN stack of ``num_layers`` layers over one u-cache pool, driven through
    SGLang's glue: per verify, every layer's FlashInfer verify, then one cursor
    commit and the track materialize; an fp32 recurrence tracks every slot."""

    def __init__(self, *, num_layers, num_slots, h, hv, seed, contiguous=True):
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.h, self.hv, self.num_layers = h, hv, num_layers
        self.buffers = allocate_gdn_ucache(
            num_layers=num_layers,
            num_slots=num_slots,
            num_k_heads=h,
            num_v_heads=hv,
            head_k_dim=K,
            head_v_dim=V,
            max_num_draft_tokens=T,
            device="cuda",
            track_snapshots=True,
        )
        # Poisoned rings: a record read outside the live window turns into NaN.
        for ring in (self.buffers.k_ring, self.buffers.u_ring, self.buffers.g_ring):
            ring.fill_(float("nan"))
        shape = (num_layers, num_slots, hv, V, K)
        init = self.randn(*shape, scale=0.5).to(torch.float16)
        if contiguous:
            self.state = init
        else:
            # A padded layer stride: no single [layers * slots] view exists.
            padded = torch.zeros(
                (num_layers, num_slots + 3, hv, V, K),
                dtype=torch.float16,
                device=self.device,
            )
            self.state = padded[:, :num_slots]
            self.state.copy_(init)
        self.truth = self.state.float().clone()
        self.A_log = [
            torch.full((hv,), -3.0, device=self.device)
            + self.randn(hv, scale=0.3).abs()
            for _ in range(num_layers)
        ]
        self.dt_bias = [
            self.randn(hv, scale=0.5).to(torch.bfloat16) for _ in range(num_layers)
        ]
        self.verifier = GdnUcacheVerifier(device=self.device, first_layer_id=0)
        self.track = GdnUcacheTrackMaterializer(state=self.state, buffers=self.buffers)

    def randn(self, *shape, scale=1.0):
        return torch.randn(*shape, generator=self.gen, device=self.device) * scale

    def draw_inputs(self, batch):
        return [
            dict(
                q=self.randn(batch * T, self.h, K).to(torch.bfloat16),
                k=self.randn(batch * T, self.h, K).to(torch.bfloat16),
                v=self.randn(batch * T, self.hv, V, scale=0.5).to(torch.bfloat16),
                a=self.randn(batch * T, self.hv, scale=0.5).to(torch.bfloat16),
                b=self.randn(batch * T, self.hv).to(torch.bfloat16),
            )
            for _ in range(self.num_layers)
        ]

    def verify(self, slots, inputs):
        """Runs every layer's verify; returns [layers, batch, T, HV, V] outputs."""
        cache_indices = torch.tensor(slots, dtype=torch.int32, device=self.device)
        outputs = []
        for layer, x in enumerate(inputs):
            out = self.verifier.target_verify(
                layer_id=layer,
                query=x["q"],
                key=x["k"],
                value=x["v"],
                a=x["a"],
                b=x["b"],
                A_log=self.A_log[layer],
                dt_bias=self.dt_bias[layer],
                state=self.state[layer],
                k_ring=self.buffers.k_ring[layer],
                u_ring=self.buffers.u_ring[layer],
                g_ring=self.buffers.g_ring[layer],
                buffers=self.buffers,
                cache_indices=cache_indices,
                num_draft_tokens=T,
            )
            outputs.append(out.view(len(slots), T, self.hv, V).clone())
        return torch.stack(outputs)

    def commit(self, slots, accepts, tracks=None, steps=None):
        """Commits the cursors, then materializes rows with ``steps >= 0``."""
        cache_indices = torch.tensor(slots, dtype=torch.int32, device=self.device)
        num_accept = torch.tensor(accepts, dtype=torch.int64, device=self.device)
        commit_gdn_ucache_cursors(
            buffers=self.buffers, slots=cache_indices, num_accept_tokens=num_accept
        )
        if tracks is not None:
            self.track.materialize(
                slots=cache_indices,
                num_accept_tokens=num_accept,
                track_slots=torch.tensor(tracks, dtype=torch.int64, device=self.device),
                steps_to_track=torch.tensor(
                    steps, dtype=torch.int64, device=self.device
                ),
            )

    def fp32_states(self, slots, inputs):
        """[layers, T, batch, HV, V, K] fp32 states after each verify token, from
        each row's truth (rows with slot <= 0 read slot 0 and are ignored)."""
        idx = torch.tensor([max(s, 0) for s in slots], device=self.device)
        return torch.stack(
            [
                _fp32_tokens(
                    self.truth[layer][idx],
                    x,
                    A_log=self.A_log[layer],
                    dt_bias=self.dt_bias[layer],
                    group=self.hv // self.h,
                )[1]
                for layer, x in enumerate(inputs)
            ]
        )


def _fp32_tokens(state, x, *, A_log, dt_bias, group):
    """Batched fp32 recurrence: ``state`` [B, HV, V, K] through each row's T
    tokens; returns outputs [B, T, HV, V] and states [T, B, HV, V, K]."""
    batch = state.shape[0]

    def rows(t):
        return t.float().view(batch, T, *t.shape[1:])

    q, k, v, a, b = (rows(x[n]) for n in ("q", "k", "v", "a", "b"))
    q_hat = F.normalize(q, dim=-1).repeat_interleave(group, dim=2) * K**-0.5
    k_hat = F.normalize(k, dim=-1).repeat_interleave(group, dim=2)
    log_decay = -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())
    beta = torch.sigmoid(b)
    outputs, states = [], []
    for t in range(T):
        state = state * torch.exp(log_decay[:, t])[..., None, None]
        pred = torch.einsum("bhvk,bhk->bhv", state, k_hat[:, t])
        u = (v[:, t] - pred) * beta[:, t][..., None]
        state = state + u[..., None] * k_hat[:, t][:, :, None, :]
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q_hat[:, t]))
        states.append(state)
    return torch.stack(outputs, dim=1), torch.stack(states)


def _prep_accepts(hist):
    # Accept runs that grow an empty window to `hist` without folding (<= 12 before each verify).
    return [T] * (hist // T) + ([hist % T] if hist % T else [])


class TestGdnUcacheTrackMaterialize(CustomTestCase):
    def test_snapshot_matches_fp32_at_every_accept_step_window_and_base(self):
        """Every (accept a, track step s < a) x window length before the verify
        (0..12 unfolded, 13..16 folded in the same verify) x ring base 0..31: the
        track slot holds the fp32 state after s + 1 of the verify's tokens."""
        cases = [
            (a, s, hist, base)
            for a in range(1, T + 1)
            for s in range(a)
            for hist in range(17)
            for base in range(32)
        ]
        batch = len(cases)
        live = list(range(1, batch + 1))
        tracks = list(range(batch + 1, 2 * batch + 1))
        model = _Model(num_layers=1, num_slots=2 * batch + 1, h=4, hv=4, seed=0)
        model.buffers.cache_base[live] = torch.tensor(
            [base for *_, base in cases], dtype=torch.int32, device=model.device
        )
        runs = [_prep_accepts(hist) for _, _, hist, _ in cases]
        for step in range(max(len(r) for r in runs)):
            slots = [s if step < len(r) else -1 for s, r in zip(live, runs)]
            accepts = [r[step] if step < len(r) else 1 for r in runs]
            inputs = model.draw_inputs(batch)
            states = model.fp32_states(slots, inputs)
            model.verify(slots, inputs)
            model.commit(slots, accepts)
            for row, slot in enumerate(slots):
                if slot > 0:
                    model.truth[0, slot] = states[0, accepts[row] - 1, row]
        hist = model.buffers.hist_len[live].tolist()
        self.assertEqual(hist, [h for _, _, h, _ in cases])

        inputs = model.draw_inputs(batch)
        states = model.fp32_states(live, inputs)
        model.verify(live, inputs)
        steps = [s for _, s, _, _ in cases]
        model.commit(live, [a for a, *_ in cases], tracks=tracks, steps=steps)
        want = states[0, steps, torch.arange(batch)]
        err = (model.state[0, tracks].float() - want).abs().amax(dim=(1, 2, 3))
        worst = int(err.argmax())
        report = (
            f"max |track - fp32| {err.max():.2e} at (a, s, hist, base) {cases[worst]}"
        )
        print(report)
        self.assertLess(err.max().item(), STATE_TOL, report)
        self.assertEqual(model.buffers.hist_len[tracks].abs().sum().item(), 0)
        self.assertEqual(model.buffers.cache_base[tracks].abs().sum().item(), 0)

    def test_multistep_snapshots_every_layer_across_folds(self):
        """Many verifies with random accepts and random track steps (incl. none):
        each snapshot of every layer matches the fp32 state at its tracking point,
        rows that do not track leave their track slot untouched, and the live
        windows keep matching the recurrence."""
        num_layers, h, hv, rows = 3, 4, 12, 6
        live = [3, 9, 5, 11, 1, 7]
        tracks = [2, 4, 6, 8, 10, 12]
        model = _Model(num_layers=num_layers, num_slots=13, h=h, hv=hv, seed=1)
        self.assertTrue(model.track.single_launch)
        worst, kinds = 0.0, set()
        for _ in range(40):
            accepts = torch.randint(
                1, T + 1, (rows,), generator=model.gen, device="cuda"
            )
            accepts = accepts.tolist()
            steps = [
                int(torch.randint(-1, a, (1,), generator=model.gen, device="cuda"))
                for a in accepts
            ]
            hist_pre = model.buffers.hist_len[live].tolist()
            inputs = model.draw_inputs(rows)
            states = model.fp32_states(live, inputs)
            track_before = model.state[:, tracks].clone()
            model.verify(live, inputs)
            model.commit(live, accepts, tracks=tracks, steps=steps)
            for r, slot in enumerate(live):
                model.truth[:, slot] = states[:, accepts[r] - 1, r]
                got = model.state[:, tracks[r]]
                if steps[r] < 0:
                    self.assertTrue(torch.equal(got, track_before[:, r]))
                    continue
                err = (got.float() - states[:, steps[r], r]).abs().max().item()
                worst = max(worst, err)
                kinds.add(
                    (
                        hist_pre[r] >= model.buffers.flush_min,
                        steps[r] == 0,
                        steps[r] == accepts[r] - 1,
                    )
                )
        print(f"max |track - fp32| over 40 verifies x {num_layers} layers: {worst:.2e}")
        self.assertLess(worst, STATE_TOL)
        for folded in (False, True):
            self.assertTrue(any(k[0] == folded and k[1] for k in kinds), kinds)
            self.assertTrue(any(k[0] == folded and k[2] for k in kinds), kinds)

    def test_flattened_launch_matches_per_layer_launches_bitwise(self):
        """The one launch over [layers * slots] views must address every layer's
        state and rings exactly as one launch per layer does."""
        models = [
            _Model(num_layers=3, num_slots=10, h=4, hv=8, seed=2, contiguous=c)
            for c in (True, False)
        ]
        self.assertEqual([m.track.single_launch for m in models], [True, False])
        live, tracks = [1, 2, 3, 4], [6, 7, 8, 9]
        for step in range(5):
            accepts = [T, 3, T, 2]
            for m in models:
                m.gen.manual_seed(100 + step)
                m.verify(live, m.draw_inputs(len(live)))
                m.commit(live, accepts, tracks=tracks, steps=[a - 1 for a in accepts])
        flat, per_layer = models
        self.assertTrue(torch.equal(flat.state, per_layer.state))
        self.assertTrue(torch.equal(flat.buffers.hist_len, per_layer.buffers.hist_len))
        # Every layer's track slot was written (none still holds its initial state).
        initial = _Model(num_layers=3, num_slots=10, h=4, hv=8, seed=2).state
        changed = (flat.state[:, tracks] != initial[:, tracks]).flatten(2).any(-1)
        self.assertTrue(bool(changed.all()), changed)

    def test_graph_replay_matches_eager_including_no_active_rows(self):
        """A captured commit + materialize, replayed with other cursors, slots and
        track steps (all rows tracking, some, none), writes exactly what the eager
        calls write; the plan reads its inputs on the device."""
        rows = 5
        eager = _Model(num_layers=2, num_slots=16, h=4, hv=8, seed=3)
        graphed = _Model(num_layers=2, num_slots=16, h=4, hv=8, seed=3)
        rings = (
            eager.randn(*eager.buffers.k_ring.shape).to(torch.bfloat16),
            eager.randn(*eager.buffers.u_ring.shape).to(torch.bfloat16),
            -eager.randn(*eager.buffers.g_ring.shape).abs().cumsum(-1) * 0.05,
        )
        for m in (eager, graphed):
            for ring, value in zip(
                (m.buffers.k_ring, m.buffers.u_ring, m.buffers.g_ring), rings
            ):
                ring.copy_(value)
            m.track.warm_up(max_batch_size=8)
        static = dict(
            slots=torch.zeros(rows, dtype=torch.int32, device="cuda"),
            num_accept_tokens=torch.zeros(rows, dtype=torch.int64, device="cuda"),
            track_slots=torch.zeros(rows, dtype=torch.int64, device="cuda"),
            steps_to_track=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            commit_gdn_ucache_cursors(
                buffers=graphed.buffers,
                slots=static["slots"],
                num_accept_tokens=static["num_accept_tokens"],
            )
            graphed.track.materialize(**static)
        torch.cuda.current_stream().wait_stream(stream)
        # An eager batch past the warm-up size grows the plan after the capture;
        # the captured plan must stay alive (no padding row writes state).
        captured_plan = graphed.track._plan
        padded = torch.full((12,), -1, dtype=torch.int32, device="cuda")
        graphed.track.materialize(
            slots=padded,
            num_accept_tokens=padded,
            track_slots=padded,
            steps_to_track=padded,
        )
        self.assertEqual(graphed.track._plan.rows, 12)
        self.assertIn(captured_plan, graphed.track._retired_plans)

        rounds = [
            (
                [1, 2, 3, 4, 5],
                [T, 1, 2, T, 3],
                [9, 10, 11, 12, 13],
                [0, 0, 1, T - 1, 2],
            ),
            ([5, 4, 0, 2, 1], [2, T, 1, 3, 1], [14, -1, 9, 15, 10], [1, 2, 0, -1, 0]),
            ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [9, 10, 11, 12, 13], [-1] * 5),
        ]
        for slots, accepts, tracks, steps in rounds:
            hist = torch.randint(0, 17, (16,), generator=eager.gen, device="cuda")
            base = torch.randint(0, 32, (16,), generator=eager.gen, device="cuda")
            for m in (eager, graphed):
                m.buffers.hist_len.copy_(hist)
                m.buffers.cache_base.copy_(base)
            eager.commit(slots, accepts, tracks=tracks, steps=steps)
            for name, value in zip(static, (slots, accepts, tracks, steps)):
                static[name].copy_(torch.tensor(value))
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(eager.state, graphed.state), (slots, steps))
            self.assertTrue(
                torch.equal(eager.buffers.hist_len, graphed.buffers.hist_len)
            )
            self.assertTrue(
                torch.equal(eager.buffers.cache_base, graphed.buffers.cache_base)
            )
            self.assertEqual(
                int(captured_plan.num_active),
                sum(
                    s >= 0 and t > 0 and sl > 0
                    for s, t, sl in zip(steps, tracks, slots)
                )
                * 2,
            )

    def test_warm_up_sizes_the_plan_for_eager_batches(self):
        """Serving warms the plan for the largest eager verify, so no batch grows
        it after graph capture."""
        model = _Model(num_layers=2, num_slots=8, h=4, hv=8, seed=5)
        model.track.warm_up(max_batch_size=4, max_eager_batch_size=12)
        self.assertEqual(model.track._plan.rows, 12)
        self.assertEqual(model.track._retired_plans, [])

    def test_request_resumed_from_snapshot_continues_like_the_original(self):
        """A request that decodes past folds snapshots at a track point; a second
        request copy-on-writes that snapshot into a recycled slot (stale cursors,
        reset at its extend) and runs the original's later verifies: its outputs
        track the original's within the snapshot's rounding."""
        model = _Model(num_layers=2, num_slots=8, h=4, hv=8, seed=4)
        orig, track, resumed = 1, 2, 5
        model.buffers.hist_len[resumed] = 11
        model.buffers.cache_base[resumed] = 23
        accepts = [3, T, 2, T, 1, T, 3, T, 2, T]
        folds = 0
        for step, a in enumerate(accepts):
            folds += int(model.buffers.hist_len[orig] >= model.buffers.flush_min)
            inputs = model.draw_inputs(1)
            model.verify([orig], inputs)
            model.commit([orig], [a], tracks=[track], steps=[a - 1])
        self.assertGreaterEqual(folds, 2)
        self.assertGreater(int(model.buffers.hist_len[orig]), 0)
        # COW of the tree slot into the request slot, then the extend-time reset.
        model.state[:, resumed] = model.state[:, track]
        reset_gdn_ucache_cursors(
            buffers=model.buffers,
            slots=torch.tensor([resumed], dtype=torch.int32, device="cuda"),
        )
        worst = 0.0
        for a in [T, 2, T, 3, T, 1, T]:
            inputs = model.draw_inputs(1)
            # Both rows see the same tokens, so the second batch row repeats the first.
            pair = [{n: torch.cat((x[n], x[n])) for n in x} for x in inputs]
            out = model.verify([orig, resumed], pair)
            model.commit([orig, resumed], [a, a])
            worst = max(worst, (out[:, 0] - out[:, 1]).abs().max().item())
        print(f"max |resumed - original| output: {worst:.2e}")
        self.assertLess(worst, STATE_TOL)


if __name__ == "__main__":
    unittest.main()
