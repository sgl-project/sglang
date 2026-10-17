import unittest

import torch

from sglang.srt.speculative.pp_spec_wire import (
    decode_relay,
    encode_chain,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPSpecWire(unittest.TestCase):
    def test_fixed_capacity_chain_decodes_to_logical_width(self):
        wire_chain = encode_chain(
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            batch_size=2,
            logical_width=4,
            capacity=6,
        )
        relay = decode_relay(
            rids=["a", "b"],
            tensors={"spec_next_chain": wire_chain},
            steps=3,
            logical_width=4,
            fixed_capacity=True,
        )

        self.assertEqual(tuple(wire_chain.shape), (2, 6))
        self.assertEqual(relay.configuration, (3, 4))
        self.assertTrue(relay.tokens.is_contiguous())
        torch.testing.assert_close(
            relay.tokens, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        )

    def test_missing_chain_decodes_to_logical_degenerate_proposal(self):
        relay = decode_relay(
            rids=["a"],
            tensors={"spec_bonus_tokens": torch.tensor([7])},
            steps=1,
            logical_width=2,
            fixed_capacity=True,
        )

        self.assertEqual(relay.configuration, (1, 2))
        torch.testing.assert_close(relay.tokens, torch.tensor([[7, 0]]))


if __name__ == "__main__":
    unittest.main()
