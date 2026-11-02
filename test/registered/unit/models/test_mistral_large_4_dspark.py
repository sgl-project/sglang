import unittest

import torch

from sglang.srt.models.dspark import RNNHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, HIDDEN, DEPTH, MARKOV_RANK = 100, 64, 3, 8


def _trainer_rnn_bias(prev_ids, hidden, embedding, joint_proj, output):
    """Mistral's trainer RNNHead (mtp.dspark.autoregressive_head), unrolled."""
    state = None
    biases = []
    for d in range(prev_ids.shape[1]):
        emb = embedding(prev_ids[:, d])
        if state is None:
            state = torch.zeros_like(emb)
        joint = joint_proj(torch.cat([state, emb, hidden[:, d]], dim=-1))
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        biases.append(output(torch.tanh(output_raw)))
    return torch.stack(biases, dim=1)


class TestRNNHeadMatchesTrainer(CustomTestCase):
    def test_step_bias_threads_the_state_like_the_trainer(self):
        torch.manual_seed(0)
        head = RNNHead(vocab_size=VOCAB, markov_rank=MARKOV_RANK, hidden_size=HIDDEN)
        with torch.no_grad():
            for param in head.parameters():
                param.normal_()
        prev_ids = torch.randint(VOCAB, (3, DEPTH))
        hidden = torch.randn(3, DEPTH, HIDDEN)

        state = head.init_state(hidden[:, 0])
        biases = []
        for d in range(DEPTH):
            bias, state = head.step_bias(
                prev_ids[:, d], hidden_states=hidden[:, d], state=state
            )
            biases.append(bias)

        expected = _trainer_rnn_bias(
            prev_ids, hidden, head.markov_w1, head.joint_proj, head.markov_w2
        )
        torch.testing.assert_close(torch.stack(biases, dim=1), expected)


if __name__ == "__main__":
    unittest.main()
