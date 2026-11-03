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

    def test_prebuild_sizes_the_gathered_width_once(self):
        """The state is built once, for the gathered width (shard x TP size);
        repeated prebuilds keep it, so nothing allocates again under capture."""
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

    def test_unaligned_width_disables_like_the_first_call(self):
        """A width the kernel cannot move disables the gatherer (NCCL fallback),
        the same decision the first eager call makes, instead of staying pending."""
        gatherer = _lazy_gatherer()
        gatherer.prebuild(hidden_size=129, dtype=torch.bfloat16)
        self.assertFalse(gatherer.prebuild_pending)
        self.assertIsNone(gatherer._state)
        self.assertEqual(self.created, [])

    def test_non_bf16_logits_disable_like_the_first_call(self):
        """Non-bf16 logits (fp32 lm_head, fp16 models) disable the gatherer without
        allocating, as the first eager call does, instead of building a bf16 buffer."""
        for dtype in (torch.float32, torch.float16):
            gatherer = _lazy_gatherer()
            self.assertFalse(gatherer.prebuild(hidden_size=128, dtype=dtype))
            self.assertFalse(gatherer.prebuild_pending)
            self.assertIsNone(gatherer._state)
        self.assertEqual(self.created, [])


if __name__ == "__main__":
    unittest.main()
