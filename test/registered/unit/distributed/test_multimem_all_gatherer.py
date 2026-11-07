"""Unit tests for MultimemAllGatherer's eager state build."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _lazy_gatherer() -> MultimemAllGatherer:
    gatherer = object.__new__(MultimemAllGatherer)
    gatherer._max_tokens = 16
    gatherer._skip_entry_sync = True
    gatherer._state = MultimemAllGatherer._UNINIT
    return gatherer


class TestMultimemAllGathererPrebuild(CustomTestCase):
    def setUp(self):
        self.created = []

        def create_state(**kwargs):
            self.created.append(kwargs)
            return SimpleNamespace(symm_mem_hdl=SimpleNamespace(multicast_ptr=1))

        self.enterContext(
            patch.object(triton_symm_mem_ag, "create_state", side_effect=create_state)
        )
        self.enterContext(
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=SimpleNamespace(
                    world_size=2, device_group=None, rank_in_group=0
                ),
            )
        )

    def test_prebuild_decides_like_the_first_call(self):
        """Without it a prebuild that sizes the buffer for the shard instead of
        the gathered width, rebuilds under capture, or leaves an unsupported
        width or dtype pending (a bf16 buffer for fp32 logits) escapes."""
        with self.subTest(case="aligned bf16 builds once for the gathered width"):
            gatherer = _lazy_gatherer()
            self.assertTrue(gatherer.prebuild_pending)
            self.assertTrue(gatherer.prebuild(hidden_size=128, dtype=torch.bfloat16))
            state = gatherer._state
            gatherer.prebuild(hidden_size=128, dtype=torch.bfloat16)
            self.assertFalse(gatherer.prebuild_pending)
            self.assertIs(gatherer._state, state)
            self.assertEqual(len(self.created), 1)
            self.assertEqual(self.created[0]["hidden_size"], 256)
            self.assertEqual(self.created[0]["max_tokens"], 16)
        self.created.clear()
        with self.subTest(case="unaligned width disables without allocating"):
            gatherer = _lazy_gatherer()
            gatherer.prebuild(hidden_size=129, dtype=torch.bfloat16)
            self.assertFalse(gatherer.prebuild_pending)
            self.assertIsNone(gatherer._state)
            self.assertEqual(self.created, [])
        for dtype in (torch.float32, torch.float16):
            with self.subTest(case="non-bf16 disables without allocating", dtype=dtype):
                gatherer = _lazy_gatherer()
                self.assertFalse(gatherer.prebuild(hidden_size=128, dtype=dtype))
                self.assertFalse(gatherer.prebuild_pending)
                self.assertIsNone(gatherer._state)
                self.assertEqual(self.created, [])


if __name__ == "__main__":
    unittest.main()
