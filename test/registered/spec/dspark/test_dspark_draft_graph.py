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
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
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

    def test_captured_sampler_matches_eager_block(self):
        # The real tail (staging, static buffers, [:bs] views) against eager.
        torch.manual_seed(1)
        head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(DEVICE)
        head = head.to(torch.bfloat16)
        weight = torch.randn(VOCAB, 64, device=DEVICE).to(torch.bfloat16)
        model = SimpleNamespace(
            markov_head=head,
            confidence_head=DSparkConfidenceHead(hidden_size=64, markov_rank=RANK).to(
                DEVICE
            ),
            sample_from_anchor=True,
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
        sampler = DsparkDraftSampler(
            model=model,
            gamma=GAMMA,
            max_bs=2 * BS,
            device=DEVICE,
            noise=noise,
            tp_sync=tp_sync,
            greedy_draft=False,
        )
        hidden = torch.randn(BS * GAMMA, 64, device=DEVICE).to(torch.bfloat16)
        input_ids = torch.randint(0, VOCAB, (BS * GAMMA,), device=DEVICE)
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=(torch.rand(BS, 1, device=DEVICE) + 0.3),
            top_ks=torch.tensor([1, 40] * (BS // 2), device=DEVICE),
        )
        sampler.stage_sampling_params(bs=BS, sampling_info=sampling_info)
        sampler(hidden, input_ids)  # warm up outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(hidden, input_ids)
        noise.offset.zero_()
        graph.replay()
        torch.cuda.synchronize()

        eager_noise = DraftNoise(
            key=5, offset=torch.zeros(1, dtype=torch.int64, device=DEVICE)
        )
        eager = sample_draft_block(
            base_logits=model.compute_base_logits(hidden)[0].view(BS, GAMMA, -1),
            anchor_tokens=input_ids.view(BS, GAMMA)[:, 0],
            draft_hidden=hidden.view(BS, GAMMA, -1),
            sampling_info=sampling_info,
            markov_head=head,
            noise=eager_noise,
            device=DEVICE,
            tp_sync=tp_sync,
            greedy_draft=False,
        )
        self.assertTrue(
            torch.equal(sampler.out[: BS * GAMMA].view(BS, GAMMA), eager.draft_tokens)
        )
        self.assertTrue(torch.equal(sampler.step_logits[:BS], eager.corrected_logits))
        self.assertEqual(noise.offset.item(), eager_noise.offset.item())
        # Folded steps carry the confidence the eager proposal computes.
        confidence, confidence_raw = model.compute_confidence(
            anchor_tokens=input_ids.view(BS, GAMMA)[:, 0],
            draft_tokens=eager.draft_tokens,
            confidence_features=hidden,
        )
        self.assertTrue(torch.equal(sampler.confidence_out[:BS], confidence))
        self.assertTrue(torch.equal(sampler.confidence_raw_out[:BS], confidence_raw))


if __name__ == "__main__":
    unittest.main()
