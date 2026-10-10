import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    draft_markov_block,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
VOCAB, RANK, BS, GAMMA = 10000, 16, 8, 5


class TestDraftMarkovBlockGraph(CustomTestCase):
    def test_captured_block_matches_eager(self):
        # The graph tail replays the eager markov block: same tokens, same step
        # logits, and the noise counter advances once per replay.
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
        logits_out = torch.empty(GAMMA, BS, VOCAB, dtype=torch.bfloat16, device=DEVICE)
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


if __name__ == "__main__":
    unittest.main()
