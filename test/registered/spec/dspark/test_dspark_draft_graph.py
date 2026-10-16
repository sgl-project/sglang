import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    MarkovGreedyStep,
)
from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    DSparkDraftMixin,
    VanillaMarkov,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    draft_markov_block,
    resolve_draft_params,
    run_draft_proposal,
)
from sglang.srt.speculative.dspark_components.dspark_proposal_graph_tail import (
    DraftProposalGraphTail,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
VOCAB, RANK, BS, GAMMA = 10000, 16, 8, 5


class TestDraftMarkovBlockGraph(CustomTestCase):
    def test_captured_block_matches_eager(self):
        # Same tokens and logits as eager; the noise counter advances per replay.
        torch.manual_seed(0)
        head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(DEVICE)
        head = head.to(torch.bfloat16)
        base = torch.randn(BS, GAMMA, VOCAB, device=DEVICE).to(torch.bfloat16)
        anchor = torch.randint(0, VOCAB, (BS,), device=DEVICE)
        temperatures = torch.rand(BS, device=DEVICE) + 0.3
        temperatures[::2] = 0.0
        noise = DraftNoise(
            key=11, offset=torch.zeros(1, dtype=torch.int64, device=DEVICE)
        )
        logits_out = torch.empty(BS, GAMMA, VOCAB, dtype=torch.bfloat16, device=DEVICE)
        tokens_out = torch.empty(BS, GAMMA, dtype=torch.int64, device=DEVICE)

        def block():
            tokens, _ = draft_markov_block(
                markov_head=head,
                base_logits=base,
                anchor_tokens=anchor,
                draft_hidden=None,
                step_temperatures=temperatures,
                noise=noise,
                tp_sync=SimpleNamespace(sync=lambda site, value: value),
                logits_out=logits_out,
            )
            tokens_out.copy_(tokens)

        block()  # warm up the triton kernels outside the capture
        noise.offset.zero_()
        block()
        eager_tokens, eager_logits = tokens_out.clone(), logits_out.clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        noise.offset.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(tokens_out, eager_tokens))
        self.assertTrue(torch.equal(logits_out, eager_logits))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(noise.offset.item(), 2)
        self.assertFalse(torch.equal(tokens_out[1::2], eager_tokens[1::2]))
        self.assertTrue(torch.equal(tokens_out[::2], eager_tokens[::2]))

    def test_one_greedy_row_matches_eager_and_fp32_reference(self):
        # The fused one-row kernel runs; the graph replays eager; both match FP32.
        torch.manual_seed(2)
        head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(DEVICE)
        head = head.to(torch.bfloat16)
        base = torch.randn(1, GAMMA, VOCAB, device=DEVICE).to(torch.bfloat16)
        anchor = torch.randint(0, VOCAB, (1,), device=DEVICE)
        noise = DraftNoise(
            key=3, offset=torch.zeros(1, dtype=torch.int64, device=DEVICE)
        )
        tokens_out = torch.empty(1, GAMMA, dtype=torch.int64, device=DEVICE)

        def block():
            tokens, logits = draft_markov_block(
                markov_head=head,
                base_logits=base,
                anchor_tokens=anchor,
                draft_hidden=None,
                step_temperatures=torch.zeros(1, device=DEVICE),
                noise=noise,
                tp_sync=SimpleNamespace(sync=lambda site, value: value),
                greedy_draft=True,
            )
            assert logits is None
            tokens_out.copy_(tokens)

        with mock.patch.object(
            MarkovGreedyStep, "execute", wraps=MarkovGreedyStep.execute
        ) as fused:
            block()
        self.assertEqual(fused.call_count, GAMMA)
        eager_tokens = tokens_out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        tokens_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(tokens_out, eager_tokens))

        prev = anchor
        for step in range(GAMMA):
            bias = head.markov_w1(prev).float() @ head.markov_w2.weight.float().T
            prev = (base[:, step].float() + bias).argmax(-1)
            self.assertEqual(eager_tokens[0, step].item(), prev.item())

    def test_graph_tail_matches_eager_proposal(self):
        # The real tail (staging, static buffers, [:bs] views) against eager.
        for sample_from_anchor in (True, False):
            for greedy_draft in (False, True):
                for with_confidence in (True, False):
                    with self.subTest(
                        sample_from_anchor=sample_from_anchor,
                        greedy_draft=greedy_draft,
                        with_confidence=with_confidence,
                    ):
                        self._check_graph_tail_matches_eager(
                            sample_from_anchor=sample_from_anchor,
                            greedy_draft=greedy_draft,
                            with_confidence=with_confidence,
                        )

    def _check_graph_tail_matches_eager(
        self, *, sample_from_anchor, greedy_draft, with_confidence
    ):
        torch.manual_seed(1)
        head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(DEVICE)
        head = head.to(torch.bfloat16)
        weight = torch.randn(VOCAB, 64, device=DEVICE).to(torch.bfloat16)
        model = SimpleNamespace(
            markov_head=head,
            confidence_head=(
                DSparkConfidenceHead(hidden_size=64, markov_rank=RANK).to(DEVICE)
                if with_confidence
                else None
            ),
            sample_from_anchor=sample_from_anchor,
            lm_head=SimpleNamespace(org_vocab_size=VOCAB, weight=weight),
            compute_base_logits=lambda hidden: (hidden @ weight.T, hidden),
        )
        model.compute_confidence = MethodType(
            DSparkDraftMixin.compute_confidence, model
        )
        tp_sync = SimpleNamespace(sync=lambda site, value: value)
        noise = DraftNoise(
            key=5, offset=torch.zeros(1, dtype=torch.int64, device=DEVICE)
        )
        tail = DraftProposalGraphTail(
            model=model,
            gamma=GAMMA,
            max_bs=2 * BS,
            device=DEVICE,
            noise=noise,
            tp_sync=tp_sync,
            greedy_draft=greedy_draft,
        )
        query_token_num = GAMMA if sample_from_anchor else GAMMA + 1
        hidden = torch.randn(BS * query_token_num, 64, device=DEVICE).to(torch.bfloat16)
        input_ids = torch.randint(0, VOCAB, (BS * query_token_num,), device=DEVICE)
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=(torch.rand(BS, 1, device=DEVICE) + 0.3),
            top_ks=torch.tensor([1, 40] * (BS // 2), device=DEVICE),
        )
        tail.stage_sampling_params(bs=BS, sampling_info=sampling_info)
        tail(hidden, input_ids)  # warm up outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tail(hidden, input_ids)
        noise.offset.zero_()
        graph.replay()
        torch.cuda.synchronize()

        eager_noise = DraftNoise(
            key=5, offset=torch.zeros(1, dtype=torch.int64, device=DEVICE)
        )
        params = resolve_draft_params(sampling_info=sampling_info, bs=BS, device=DEVICE)
        eager = run_draft_proposal(
            model=model,
            hidden_states=hidden,
            anchor_tokens=input_ids.view(BS, query_token_num)[:, 0],
            bs=BS,
            gamma=GAMMA,
            step_temperatures=params.step_temperatures,
            noise=eager_noise,
            tp_sync=tp_sync,
            greedy_draft=greedy_draft,
        )
        folded, _ = tail.views(BS)
        self.assertEqual(noise.offset.item(), eager_noise.offset.item())
        if greedy_draft:
            self.assertIsNone(folded.step_logits)
            self.assertIsNone(eager.step_logits)
        if not with_confidence:
            self.assertIsNone(folded.confidence)
            self.assertIsNone(eager.confidence)
        for name, value in eager._asdict().items():
            if value is not None:
                self.assertTrue(torch.equal(getattr(folded, name), value), name)


if __name__ == "__main__":
    unittest.main()
