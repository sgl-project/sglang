"""A PD decode batch under NGRAM speculation carries an NGRAM verify input.

PD prefill hands the decode its first token through req.output_ids, and the
prebuilt batch still has to seed the NGRAM accept state and its relay rows;
without them a decode-only NGRAM server fails while warming up.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation import decode_schedule_batch_mixin
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.managers.overlap_utils import (
    FutureMap,
    RelayPayload,
    resolve_forward_inputs,
)
from sglang.srt.speculative import ngram_disaggregation
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

DRAFT_TOKEN_NUM = 4
POOL_ROWS = 8
CPU = torch.device("cpu")


def _future_map():
    return FutureMap(
        device=CPU,
        spec_algo=SpeculativeAlgorithm.NGRAM,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(POOL_ROWS, 4)),
        needs_cpu_seq_lens=True,
    )


def _prebuilt_batch(pool_indices, enable_overlap):
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(output_ids=[7], swa_branching_seqlen=None, grammar=None)
            for _ in pool_indices
        ],
        tree_cache=None,
        device=CPU,
        enable_overlap=enable_overlap,
        spec_algorithm=SpeculativeAlgorithm.NGRAM,
        req_pool_indices=torch.tensor(pool_indices, dtype=torch.int64),
        seq_lens=torch.full((len(pool_indices),), 10, dtype=torch.int64),
        spec_info=None,
        input_ids=None,
        prefill_input_ids_cpu=None,
    )


def _running_input(future_indices):
    # The verify input a running decode batch holds after its own steps.
    batch_size = len(future_indices)
    return NgramVerifyInput(
        draft_token_num=DRAFT_TOKEN_NUM,
        future_indices=torch.tensor(future_indices, dtype=torch.int64),
        new_seq_lens=torch.full((batch_size,), 20, dtype=torch.int64),
        accept_tokens=torch.ones(batch_size * DRAFT_TOKEN_NUM, dtype=torch.int32),
        accept_lens=torch.ones(batch_size, dtype=torch.int32),
    )


class TestPdNgramDraftInput(CustomTestCase):
    def prebuild(self, batch, future_map):
        spec = SimpleNamespace(speculative_num_draft_tokens=DRAFT_TOKEN_NUM)
        with (
            patch.object(ngram_disaggregation, "get_spec", return_value=spec),
            patch.object(decode_schedule_batch_mixin, "checkpoint_kv_cache"),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, future_map)

    def test_first_decode_reads_no_accepts_from_a_reused_pool_row(self):
        future_map = _future_map()
        # A finished request left accepted tokens in pool row 3's relay slot.
        future_map.stash(
            torch.tensor([3]),
            RelayPayload(
                bonus_tokens=None,
                accept_tokens=torch.full((1, DRAFT_TOKEN_NUM), 9, dtype=torch.int32),
                accept_lens=torch.tensor([2], dtype=torch.int32),
            ),
        )
        batch = _prebuilt_batch([3, 5], enable_overlap=True)

        self.prebuild(batch, future_map)
        resolve_forward_inputs(batch, future_map)

        self.assertIsInstance(batch.spec_info, NgramVerifyInput)
        self.assertEqual(batch.spec_info.accept_lens.tolist(), [0, 0])
        self.assertEqual(
            batch.spec_info.accept_tokens.tolist(), [0] * (2 * DRAFT_TOKEN_NUM)
        )

    def test_prebuilt_input_merges_into_a_running_batch(self):
        for enable_overlap in (True, False):
            with self.subTest(enable_overlap=enable_overlap):
                batch = _prebuilt_batch([3, 5], enable_overlap=enable_overlap)
                self.prebuild(batch, _future_map())
                running = _running_input([1])
                if not enable_overlap:
                    running.future_indices = None

                running.merge_batch(batch.spec_info)

                if enable_overlap:
                    self.assertEqual(running.future_indices.tolist(), [1, 3, 5])
                else:
                    self.assertEqual(running.accept_lens.tolist(), [1, 0, 0])
                    self.assertEqual(running.accept_tokens.numel(), 3 * DRAFT_TOKEN_NUM)
                    self.assertEqual(running.new_seq_lens.tolist(), [20, 10, 10])


if __name__ == "__main__":
    unittest.main()
