"""Multi-layer EAGLE's draft proposal inside per-step CUDA graphs.

MultiLayerEagleMultiStepDraftExtendCudaGraphRunner captures one graph per draft
step and capture size into a shared memory pool, steps in reverse, and replays
them forward. Under rejection sampling each graph runs the production
_sample_draft_proposal: q = softmax(logits / T) into the step's draft_probs
slot, then X ~ q, seeded under deterministic inference. Here the model forward
is a copy of staged logits; the rest of the graph body is production code.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_utils import fast_sample
from sglang.test.test_utils import CustomTestCase

NUM_STEPS = 3
CAPTURE_BS = (2, 4)
MAX_BS = max(CAPTURE_BS)
# Draft-extend window per request (num_steps + 1 rows, no front rows).
WIDTH = NUM_STEPS + 1
UNTOUCHED = -1.0


def _bucket(raw_bs: int) -> int:
    return min(bs for bs in CAPTURE_BS if bs >= raw_bs)


class _PerStepProposalGraphs:
    """The per-step runner's capture pattern around the production proposal."""

    def __init__(self, vocab: int, *, prune: bool, gumbel: bool, seeded=False):
        self.prune = prune
        device = "cuda"
        rows = MAX_BS if prune else MAX_BS * WIDTH
        # The model's lm_head output for the replayed step.
        self.logits = torch.zeros((rows, vocab), dtype=torch.float32, device=device)
        self.buffers = SimpleNamespace(
            select_index=torch.zeros((MAX_BS,), dtype=torch.int64, device=device),
            temperatures=torch.ones((MAX_BS, 1), dtype=torch.float32, device=device),
            draft_probs=torch.full(
                (MAX_BS, NUM_STEPS, vocab),
                UNTOUCHED,
                dtype=torch.float32,
                device=device,
            ),
            seq_lens=torch.ones((MAX_BS,), dtype=torch.int32, device=device),
            sampling_seed=(
                torch.zeros((MAX_BS,), dtype=torch.int64, device=device)
                if seeded
                else None
            ),
        )
        self.runners = []
        for step in range(NUM_STEPS):
            runner = object.__new__(MultiLayerEagleDraftExtendCudaGraphRunner)
            runner.step = step
            runner.buffers = self.buffers
            runner.prune_draft_extend_logits = prune
            self.runners.append(runner)

        self.graphs, self.outs = {}, {}
        pool = torch.cuda.graph_pool_handle()
        with envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(gumbel):
            for step in reversed(range(NUM_STEPS)):
                for bs in reversed(CAPTURE_BS):
                    self._capture(step, bs, pool)

    def _rows(self, bs: int) -> int:
        return bs if self.prune else bs * WIDTH

    def _step_body(self, step: int, bs: int):
        # As in the real step graph, the logits live in graph memory.
        ret = SimpleNamespace(next_token_logits=self.logits[: self._rows(bs)] * 1.0)
        self.runners[step]._sample_draft_proposal(ret, bs)
        return ret

    def _capture(self, step: int, bs: int, pool):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._step_body(step, bs)  # compile the kernels before capture
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            self.outs[step, bs] = self._step_body(step, bs)
        self.graphs[step, bs] = graph

    def stage(self, raw_bs: int, temperatures, num_correct_drafts) -> int:
        """prepare()'s staging: real rows' temperatures, 1.0 on padded rows,
        and each row's last accepted token. Returns the capture size."""
        bs = _bucket(raw_bs)
        self.buffers.temperatures[:bs].fill_(1.0)
        self.buffers.temperatures[:raw_bs].copy_(temperatures)
        rows = torch.arange(bs, device="cuda")
        self.buffers.select_index[:bs].copy_(rows * WIDTH + num_correct_drafts[:bs])
        return bs

    def replay(self, step: int, bs: int, logits):
        self.logits[: logits.shape[0]].copy_(logits)
        self.graphs[step, bs].replay()
        return self.outs[step, bs]

    def selected(self, logits, bs: int):
        return logits if self.prune else logits[self.buffers.select_index[:bs]]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestPerStepDraftProposalGraphs(CustomTestCase):
    def test_each_step_writes_its_own_proposal(self):
        # 12289 spans four 4096-wide softmax blocks, the last holding one column.
        for vocab in (1000, 12289):
            for prune in (True, False):
                for gumbel in (True, False):
                    with self.subTest(vocab=vocab, prune=prune, gumbel=gumbel):
                        self._check_replays(vocab, prune=prune, gumbel=gumbel)

    def _check_replays(self, vocab: int, *, prune: bool, gumbel: bool):
        graphs = _PerStepProposalGraphs(vocab, prune=prune, gumbel=gumbel)
        gen = torch.Generator(device="cuda").manual_seed(vocab + prune + 2 * gumbel)
        torch.cuda.manual_seed(1234)
        held = []  # (worker-owned snapshot, expected value) after later replays
        for raw_bs in (3, 1, 4, 2):  # shrinks and grows across both sizes
            temperatures = 0.5 + torch.rand((raw_bs, 1), generator=gen, device="cuda")
            num_correct_drafts = torch.randint(
                0, NUM_STEPS + 1, (MAX_BS,), generator=gen, device="cuda"
            )
            bs = graphs.stage(raw_bs, temperatures, num_correct_drafts)
            graphs.buffers.draft_probs.fill_(UNTOUCHED)
            staged_temperatures = graphs.buffers.temperatures[:bs].double()
            expected_q = []
            for step in range(NUM_STEPS):
                logits = 3 * torch.randn(
                    (graphs._rows(bs), vocab), generator=gen, device="cuda"
                )
                out = graphs.replay(step, bs, logits)
                # The worker clones each step's top-k before the next replay.
                topk_p = out.topk_p[:raw_bs].clone()
                topk_index = out.topk_index[:raw_bs].clone()

                ref = torch.softmax(
                    graphs.selected(logits, bs).double() / staged_temperatures, dim=-1
                )
                q = graphs.buffers.draft_probs[:bs, step]
                # Every captured row, padded ones included, holds softmax(logits / T).
                self.assertTrue(torch.isfinite(q).all())
                torch.testing.assert_close(q.double(), ref, rtol=1e-4, atol=1e-7)
                torch.testing.assert_close(
                    q.double().sum(-1),
                    torch.ones(bs, dtype=torch.float64, device="cuda"),
                    rtol=0,
                    atol=1e-5,
                )
                self.assertTrue(((topk_index >= 0) & (topk_index < vocab)).all())
                torch.testing.assert_close(
                    topk_p, q[:raw_bs].gather(1, topk_index), rtol=0, atol=0
                )
                # Later steps' slots and rows past the capture size are untouched.
                draft_probs = graphs.buffers.draft_probs
                self.assertTrue((draft_probs[:bs, step + 1 :] == UNTOUCHED).all())
                self.assertTrue((draft_probs[bs:] == UNTOUCHED).all())
                expected_q.append(ref[:raw_bs])
                held.append((topk_p, topk_p.clone()))
                held.append((topk_index, topk_index.clone()))

            # Earlier steps' slots survive the later steps' replays.
            snapshot = (
                MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.clone_draft_probs(
                    SimpleNamespace(buffers=graphs.buffers, raw_bs=raw_bs)
                )
            )
            expected = torch.stack(expected_q, dim=1)
            torch.testing.assert_close(
                snapshot.double(), expected, rtol=1e-4, atol=1e-7
            )
            held.append((snapshot, snapshot.clone()))

        for snapshot, expected in held:
            torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)

    def test_seeded_draws_repeat_and_match_the_eager_sampler(self):
        """Under deterministic inference each step's graph draws the seeded
        proposal: the same seeds, positions and step give the same token on
        every replay, whatever the Philox state, and the graph's token is the
        one fast_sample draws eagerly from the same q."""
        vocab, raw_bs = 12289, 3
        graphs = _PerStepProposalGraphs(vocab, prune=True, gumbel=True, seeded=True)
        gen = torch.Generator(device="cuda").manual_seed(7)
        bs = graphs.stage(
            raw_bs,
            0.5 + torch.rand((raw_bs, 1), generator=gen, device="cuda"),
            torch.zeros((MAX_BS,), dtype=torch.int64, device="cuda"),
        )
        seeds = torch.tensor([11, 22, 33, 44], device="cuda")
        graphs.buffers.sampling_seed[:bs].copy_(seeds[:bs])
        graphs.buffers.seq_lens[:bs].copy_(torch.tensor([100, 200, 300, 1])[:bs])
        logits = [
            3 * torch.randn((bs, vocab), generator=gen, device="cuda")
            for _ in range(NUM_STEPS)
        ]
        draws = []
        for philox_seed in (1, 2):
            torch.cuda.manual_seed(philox_seed)
            tokens = []
            for step in range(NUM_STEPS):
                out = graphs.replay(step, bs, logits[step])
                tokens.append(out.topk_index[:bs].clone())
                _, eager = fast_sample(
                    graphs.buffers.draft_probs[:bs, step].clone(),
                    num_samples=1,
                    sampling_seed=graphs.buffers.sampling_seed[:bs],
                    positions=graphs.buffers.seq_lens[:bs],
                    draft_step=step,
                )
                self.assertEqual(tokens[-1].tolist(), eager.tolist())
            draws.append(torch.cat(tokens, dim=1))
        self.assertEqual(draws[0].tolist(), draws[1].tolist())

    def test_replays_draw_fresh_samples(self):
        """Seeded once, a replay sequence keeps drawing from q; frozen or
        reused draws pile onto one token or repeat across steps and rows."""
        vocab, raw_bs, num_replays = 12289, 4, 300
        probs = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
        # One hot token per softmax block.
        hot = torch.tensor([7, 4100, 8200, 12288], device="cuda")
        logits = torch.full((raw_bs, vocab), -50.0, device="cuda")
        logits[:, hot] = probs.log().float().cuda()
        for gumbel in (True, False):
            with self.subTest(gumbel=gumbel):
                graphs = _PerStepProposalGraphs(vocab, prune=True, gumbel=gumbel)
                bs = graphs.stage(
                    raw_bs,
                    torch.ones((raw_bs, 1), device="cuda"),
                    torch.zeros((MAX_BS,), dtype=torch.int64, device="cuda"),
                )
                torch.cuda.manual_seed(4321)
                draws = torch.empty(
                    (num_replays, NUM_STEPS, raw_bs), dtype=torch.int64, device="cuda"
                )
                for i in range(num_replays):
                    for step in range(NUM_STEPS):
                        out = graphs.replay(step, bs, logits)
                        draws[i, step] = out.topk_index[:raw_bs, 0]
                category = (draws.unsqueeze(-1) == hot).int().argmax(-1)
                self.assertTrue(torch.isin(draws, hot).all())

                # Per step and row: each token's count within 6 sigma of n * p.
                counts = torch.stack(
                    [(category == c).sum(0) for c in range(len(hot))], dim=-1
                ).cpu()
                expected = num_replays * probs
                bound = 6 * (num_replays * probs * (1 - probs)).sqrt() + 1
                self.assertTrue(((counts - expected).abs() <= bound).all(), counts)

                # Matching draws happen at sum(p^2) between independent draws.
                match = float((probs**2).sum())

                def assert_independent(a, b, what):
                    n = a.numel()
                    rate = (a == b).double().mean().item()
                    slack = 6 * math.sqrt(match * (1 - match) / n)
                    self.assertLess(rate, match + slack, what)

                assert_independent(category[1:], category[:-1], "consecutive replays")
                assert_independent(category[:, 1:], category[:, :-1], "adjacent steps")
                assert_independent(
                    category[:, :, 1:], category[:, :, :-1], "adjacent rows"
                )


if __name__ == "__main__":
    unittest.main()
