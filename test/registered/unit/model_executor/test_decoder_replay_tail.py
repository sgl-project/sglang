"""The decoder replay tail of DeepSeek-V4.1: the state's round trip
through the graph runner's static buffers. CPU only; capture runs on GPU."""

import unittest

import torch

from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcState
from sglang.srt.models.deepseek_v4_replay_tail import _flatten_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlattenState(CustomTestCase):
    def test_streams_and_pre_round_trip_and_row_caches_drop(self):
        state = HcState(
            torch.randn(3, 4, 8),
            torch.randn(3, 4),
            HcNormed(torch.randn(3, 8)),
            (torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 16)),
        )
        leaves, structure, rebuild = _flatten_state(state)
        self.assertEqual(len(leaves), 2)
        back = rebuild(leaves)
        self.assertIs(back.streams, state.streams)
        self.assertIs(back.pre, state.pre)
        self.assertIsNone(back.input)
        self.assertIsNone(back.stats)
        # The same graphs serve a state with or without the per-row caches.
        self.assertEqual(
            structure, _flatten_state(HcState(state.streams, state.pre))[1]
        )
        bare = _flatten_state(HcState(torch.randn(3, 4, 8)))
        self.assertEqual(len(bare[0]), 1)
        self.assertNotEqual(structure, bare[1])
        self.assertIsNone(bare[2](bare[0]).pre)


if __name__ == "__main__":
    unittest.main()
