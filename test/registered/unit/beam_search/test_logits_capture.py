"""Sampling may mutate logits, but must not change the beam score inputs."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.logits_capture import capture_pre_sample_logits
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBeamLogitsCapture(unittest.TestCase):
    def test_extend_capture_survives_in_place_sampling(self):
        logits = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        expected = logits[[1, 3]].clone()
        output = SimpleNamespace(next_token_logits=logits, beam=None)
        batch = SimpleNamespace(
            beam_tail=None,
            reqs=[
                SimpleNamespace(beam_group=object() if i in (1, 3) else None)
                for i in range(4)
            ],
        )
        forward = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
        capture_pre_sample_logits(batch, forward, output)
        output.next_token_logits.zero_()
        self.assertEqual(output.beam.leader_rows, [1, 3])
        self.assertTrue(torch.equal(output.beam.leader_logits, expected))

    def test_decode_capture_keeps_group_order_and_member_logits(self):
        logits = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        expected_leaders = logits[[2, 0]].clone()
        expected_members = logits[3:].clone()
        output = SimpleNamespace(
            next_token_logits=logits,
            hidden_states=torch.arange(10).reshape(5, 2),
            beam=None,
        )
        batch = SimpleNamespace(
            beam_tail=SimpleNamespace(
                num_base_rows=3,
                entries=[SimpleNamespace(leader_idx=2), SimpleNamespace(leader_idx=0)],
            )
        )
        forward = SimpleNamespace(positions=torch.arange(5))
        capture_pre_sample_logits(batch, forward, output)
        output.next_token_logits.zero_()
        self.assertTrue(torch.equal(output.beam.leader_logits, expected_leaders))
        self.assertTrue(torch.equal(output.beam.tail_logits, expected_members))
        self.assertEqual(output.hidden_states.shape[0], 3)
        self.assertEqual(forward.positions.tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
