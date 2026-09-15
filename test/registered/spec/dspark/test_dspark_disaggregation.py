"""Regression tests for the DSpark PD decode draft-input handoff."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SEQ_LENS = torch.tensor([11, 23], dtype=torch.int64)
REQ_POOL_INDICES = torch.tensor([3, 7], dtype=torch.int64)
LAST_TOKENS = torch.tensor([101, 202], dtype=torch.int64)


class TestDSparkDisaggDraftInput(CustomTestCase):
    def _make_batch(self, *, enable_overlap: bool) -> SimpleNamespace:
        return SimpleNamespace(
            seq_lens=SEQ_LENS,
            req_pool_indices=REQ_POOL_INDICES,
            enable_overlap=enable_overlap,
        )

    def _build(self, *, enable_overlap: bool) -> tuple[Any, FutureMap]:
        future_map = create_autospec(FutureMap, instance=True)
        spec_info = SpeculativeAlgorithm.DSPARK.build_disagg_draft_input(
            batch=self._make_batch(enable_overlap=enable_overlap),
            server_args=SimpleNamespace(),
            last_tokens_tensor=LAST_TOKENS,
            future_map=future_map,
        )
        return spec_info, future_map

    def test_transferred_state_reaches_the_first_decode(self):
        spec_info, future_map = self._build(enable_overlap=False)

        # Regression: DSPARK must not fall through to None here.
        self.assertIsNotNone(spec_info)
        torch.testing.assert_close(spec_info.bonus_tokens, LAST_TOKENS)
        torch.testing.assert_close(spec_info.new_seq_lens, SEQ_LENS)

        # Without overlap there is no relay to seed.
        future_map.publish.assert_not_called()
        future_map.stash.assert_not_called()

    def test_overlap_seeds_the_future_relay_before_the_first_decode(self):
        spec_info, future_map = self._build(enable_overlap=True)

        self.assertIsNotNone(spec_info)
        torch.testing.assert_close(spec_info.future_indices, REQ_POOL_INDICES)
        # No DSA IndexShare seed survives the transfer.
        self.assertFalse(spec_info.future_dsa_topk_indices_available)

        future_map.publish.assert_called_once()
        published_indices, published_seq_lens = future_map.publish.call_args.args
        torch.testing.assert_close(published_indices, REQ_POOL_INDICES)
        torch.testing.assert_close(published_seq_lens, SEQ_LENS)

        future_map.stash.assert_called_once()
        stashed_indices, payload = future_map.stash.call_args.args
        torch.testing.assert_close(stashed_indices, REQ_POOL_INDICES)
        torch.testing.assert_close(payload.bonus_tokens, LAST_TOKENS)


if __name__ == "__main__":
    unittest.main()
