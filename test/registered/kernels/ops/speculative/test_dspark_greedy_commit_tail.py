"""SGLANG_OPT_DSV41_DSPARK_GREEDY_COMMIT_TAIL: the greedy static accept tail with the
future relay scatter, the engram history commit and the deferred draft KV staging in its
launch writes exactly what the tail followed by the three separate launches writes
(eager and CUDA graph replay), and the torch reference agrees."""

import itertools
import unittest

import torch

from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history
from sglang.kernels.ops.speculative.dspark.stage_commit import stage_commit
from sglang.kernels.ops.speculative.dspark.static_accept_tail import (
    CommitStage,
    CommitTail,
    static_accept_tail,
    static_accept_tail_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

MAX_BS, POOL, VOCAB = 8, 32, 1000


def _case(bs, T, width, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *shape, hi=VOCAB: torch.randint(0, hi, shape, generator=g)
    candidates = rnd(bs, T)
    target = candidates.clone()
    # random-length matching prefixes so every correct length occurs
    for b in range(bs):
        k = int(rnd(1, hi=T)[0])
        target[b, k:] = rnd(T - k)
    seq_lens = rnd(bs, hi=5000) + 10
    verify_lens = torch.full((bs,), T, dtype=torch.int64)
    if bs > 1:
        verify_lens[-1] = 0  # a padded row
    slots = torch.randperm(POOL - 1, generator=g)[:bs] + 1
    case = dict(
        candidates=candidates,
        target=target.reshape(-1),
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        slots=slots,
        history=rnd(POOL, width) if width else None,
        cache_loc=rnd(MAX_BS * T, hi=1 << 20),
        positions=rnd(MAX_BS * T, hi=1 << 16),
    )
    return {k: (v.to(device) if v is not None else None) for k, v in case.items()}


def _outs(bs, T, device, with_slot):
    z = lambda n, dt: torch.full((n,), -7, dtype=dt, device=device)
    return dict(
        correct_len_out=z(MAX_BS, torch.int64),
        bonus_out=z(MAX_BS, torch.int64),
        cap_trim_lens_out=z(MAX_BS, torch.int32),
        commit_lens_out=z(MAX_BS, torch.int32),
        new_seq_lens_out=z(MAX_BS, torch.int64),
        out_tokens_out=torch.full((MAX_BS, T), -7, dtype=torch.int64, device=device),
        gated_commit_lens_out=z(MAX_BS, torch.int32),
        relay_seq=z(POOL, torch.int64),
        relay_tok=z(POOL, torch.int64),
        loc_dst=z(MAX_BS * T, torch.int64),
        pos_dst=z(MAX_BS * T, torch.int64),
        lens_dst=z(MAX_BS, torch.int32),
        slot_dst=z(MAX_BS * T, torch.int64) if with_slot else None,
    )


_ACCEPT_KEYS = (
    "correct_len_out",
    "bonus_out",
    "cap_trim_lens_out",
    "commit_lens_out",
    "new_seq_lens_out",
    "out_tokens_out",
    "gated_commit_lens_out",
)


def _accept_kw(case, out, gate, sim):
    return dict(
        candidates=case["candidates"],
        target_predict=case["target"],
        seq_lens=case["seq_lens"],
        verify_lens=case["verify_lens"],
        gate=gate,
        simulated_correct_len=sim,
        **{k: out[k] for k in _ACCEPT_KEYS},
    )


def _separate(case, out, history, gate, sim, bs, T, with_history, with_stage):
    """The static tail, then run_static's three launches (relay, history, stage)."""
    static_accept_tail(**_accept_kw(case, out, gate, sim))
    slots = case["slots"]
    if with_history:
        engram_commit_history(
            history,
            case["candidates"],
            slots,
            out["commit_lens_out"][:bs],
            live_rows=case["verify_lens"],
            gate=gate,
        )
    rows = slots[:bs].to(torch.int64)
    out["relay_seq"][rows] = out["new_seq_lens_out"][:bs]
    out["relay_tok"][rows] = out["bonus_out"][:bs]
    if with_stage:
        state_slot = None
        if out["slot_dst"] is not None:
            state_slot = slots.view(-1, 1).expand(bs, T).reshape(-1).contiguous()
        stage_commit(
            cache_loc_dst=out["loc_dst"],
            positions_dst=out["pos_dst"],
            commit_lens_dst=out["lens_dst"],
            state_slot_dst=out["slot_dst"],
            cache_loc=case["cache_loc"][: bs * T],
            positions=case["positions"][: bs * T],
            commit_lens=out["gated_commit_lens_out"][:bs],
            state_slot=state_slot,
            rows=bs * T,
        )


def _tail(case, out, history, gate, sim, bs, T, with_history, with_stage, fn):
    stage = None
    if with_stage:
        stage = CommitStage(
            cache_loc=case["cache_loc"][: bs * T],
            positions=case["positions"][: bs * T],
            cache_loc_dst=out["loc_dst"],
            positions_dst=out["pos_dst"],
            commit_lens_dst=out["lens_dst"],
            state_slot_dst=out["slot_dst"],
        )
    fn(
        **_accept_kw(case, out, gate, sim),
        relay=(case["slots"], out["relay_seq"], out["relay_tok"]),
        commit_tail=CommitTail(
            req_slots=case["slots"],
            history=history if with_history else None,
            stage=stage,
        ),
    )


def _diff(a, b):
    return [k for k in a if a[k] is not None and not torch.equal(a[k], b[k])]


class TestGreedyCommitTail(CustomTestCase):
    def _grid(self):
        return itertools.product(
            (1, 2, 3, 8),  # bs
            (4, 6, 8),  # T
            (0, 3, 7),  # history width
            (False, True),  # ring slots
            (0, 1),  # gate
            (False, True),  # simulated correct length
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fused_equals_separate_eager(self):
        n = 0
        for bs, T, width, with_slot, g, use_sim in self._grid():
            case = _case(bs, T, width, seed=n, device="cuda")
            gate = torch.tensor([g], dtype=torch.int32, device="cuda")
            sim = (
                torch.tensor([min(2, T - 1)], dtype=torch.int64, device="cuda")
                if use_sim
                else None
            )
            with_history = width > 0
            ref, got = _outs(bs, T, "cuda", with_slot), _outs(bs, T, "cuda", with_slot)
            h_ref = case["history"].clone() if with_history else None
            h_got = case["history"].clone() if with_history else None
            _separate(case, ref, h_ref, gate, sim, bs, T, with_history, True)
            _tail(
                case,
                got,
                h_got,
                gate,
                sim,
                bs,
                T,
                with_history,
                True,
                static_accept_tail,
            )
            self.assertEqual(_diff(ref, got), [], (bs, T, width, with_slot, g, use_sim))
            if with_history:
                self.assertTrue(torch.equal(h_ref, h_got))
            # the torch reference of the fused call
            cpu = {k: v.cpu() if v is not None else None for k, v in case.items()}
            out_cpu = _outs(bs, T, "cpu", with_slot)
            h_cpu = cpu["history"].clone() if with_history else None
            _tail(
                cpu,
                out_cpu,
                h_cpu,
                gate.cpu(),
                sim.cpu() if sim is not None else None,
                bs,
                T,
                with_history,
                True,
                static_accept_tail_torch,
            )
            self.assertEqual(
                _diff(
                    {k: v.cpu() if v is not None else None for k, v in got.items()},
                    out_cpu,
                ),
                [],
            )
            if with_history:
                self.assertTrue(torch.equal(h_got.cpu(), h_cpu))
            n += 1
        self.assertGreater(n, 200)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_fused_equals_separate_graph_replay(self):
        bs, T, width = 3, 6, 5
        for with_slot in (False, True):
            case = _case(bs, T, width, seed=99, device="cuda")
            gate = torch.ones((1,), dtype=torch.int32, device="cuda")
            ref, got = _outs(bs, T, "cuda", with_slot), _outs(bs, T, "cuda", with_slot)
            h_ref, h_got = case["history"].clone(), case["history"].clone()
            stream = torch.cuda.Stream()
            graphs = []
            for out, h, fn in ((ref, h_ref, None), (got, h_got, static_accept_tail)):
                with torch.cuda.stream(stream):  # warm up the Triton compile
                    if fn is None:
                        _separate(case, out, h.clone(), gate, None, bs, T, True, True)
                    else:
                        _tail(case, out, h.clone(), gate, None, bs, T, True, True, fn)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    if fn is None:
                        _separate(case, out, h, gate, None, bs, T, True, True)
                    else:
                        _tail(case, out, h, gate, None, bs, T, True, True, fn)
                graphs.append(graph)
            for step in range(20):
                # new inputs each replay, as consecutive verify steps
                fresh = _case(bs, T, width, seed=1000 + step, device="cuda")
                for k in ("candidates", "target", "seq_lens", "cache_loc", "positions"):
                    case[k].copy_(fresh[k])
                gate.fill_(step % 4 != 3)
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(_diff(ref, got), [], (with_slot, step))
                self.assertTrue(torch.equal(h_ref, h_got))


if __name__ == "__main__":
    unittest.main()
