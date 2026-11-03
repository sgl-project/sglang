"""Unit tests for the HiSparse staged-spec release path and padded-draft
page-table rebuild.

Two crashes fixed together (hisparse + spec decode + PD under PP prefill):

1. A staged spec request left the relay seeded with bonus tokens only, so the
   first draft decode after staging ran eager; under uneven DP-attention batch
   sizes its pre-pad metadata init crashed (stale page-table rows). The staged
   release must seed the relay with the transferred draft state.
2. When the eager draft did run, ``metadata.page_table_1`` could have fewer
   rows than the DP-padded ``topk_indices``; the page table is rebuilt from
   ``req_to_token`` for the padded batch.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    rebuild_padded_draft_page_table,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _fake_req(idx, n_in, n_out):
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=idx),
        origin_input_ids=list(range(100, 100 + n_in)),
        output_ids=list(range(200, 200 + n_out)),
    )


class TestStagedSpecSeed(CustomTestCase):
    def _run(self, spec_algorithm):
        reqs = [_fake_req(3, 5, 2), _fake_req(7, 8, 3)]
        future_map = Mock()
        future_map.make_staged_spec_input.return_value = "staged-handle"
        seed = SimpleNamespace(
            bonus_tokens=torch.arange(2),
            topk_p=torch.ones(2, 3),
            topk_index=torch.arange(6).reshape(2, 3),
            hidden_states=torch.zeros(2, 4),
            draft_probs=None,
            dsa_topk_indices=torch.full((2, 2), 5),
        )
        fake_batch = SimpleNamespace(
            spec_algorithm=spec_algorithm, return_logprob=False
        )
        self_sched = SimpleNamespace(
            device=torch.device("cpu"),
            future_map=future_map,
            spec_algorithm=spec_algorithm,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(vocab_size=100),
            enable_overlap=False,
        )
        with (
            patch(
                "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
                return_value=fake_batch,
            ),
            patch(
                "sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
                return_value=None,
            ),
            patch(
                "sglang.srt.speculative.eagle_disaggregation.build_eagle_disagg_draft_input",
                return_value=seed,
            ) as build_seed,
        ):
            batch = Scheduler._build_hisparse_decode_batch(self_sched, reqs)
        return batch, future_map, build_seed, seed

    def test_spec_release_seeds_full_draft_state(self):
        batch, future_map, build_seed, seed = self._run(SpeculativeAlgorithm.EAGLE)

        build_seed.assert_called_once()
        self.assertIsNone(batch.input_ids)
        self.assertEqual(batch.spec_info, "staged-handle")

        stash_args = future_map.stash.call_args
        payload = stash_args.args[1]
        self.assertIsInstance(payload, RelayPayload)
        self.assertTrue(
            torch.equal(payload.topk_p, seed.topk_p),
            "relay must carry the transferred topk_p",
        )
        self.assertTrue(
            torch.equal(payload.topk_index, seed.topk_index),
            "relay must carry the transferred topk_index",
        )
        self.assertTrue(
            torch.equal(payload.hidden_states, seed.hidden_states),
            "relay must carry the transferred hidden states",
        )
        self.assertTrue(
            torch.equal(payload.dsa_topk_indices, seed.dsa_topk_indices),
            "relay must carry the dsa seed so the draft stays cuda-graphable",
        )
        self.assertTrue(torch.equal(payload.bonus_tokens, seed.bonus_tokens))

    def test_standalone_release_stashes_bonus_only(self):
        # build_disagg_draft_input returns None for standalone; the release
        # must fall back to the bonus-only relay instead of crashing.
        reqs = [_fake_req(3, 5, 2)]
        future_map = Mock()
        future_map.make_staged_spec_input.return_value = "staged-handle"
        fake_batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.STANDALONE, return_logprob=False
        )
        self_sched = SimpleNamespace(
            device=torch.device("cpu"),
            future_map=future_map,
            spec_algorithm=SpeculativeAlgorithm.STANDALONE,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(vocab_size=100),
            enable_overlap=False,
        )
        with (
            patch(
                "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
                return_value=fake_batch,
            ),
            patch(
                "sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
                return_value=None,
            ),
            patch(
                "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.build_disagg_draft_input",
                return_value=None,
            ),
        ):
            batch = Scheduler._build_hisparse_decode_batch(self_sched, reqs)

        payload = future_map.stash.call_args.args[1]
        self.assertIsInstance(payload, RelayPayload)
        self.assertIsNone(payload.topk_p)
        self.assertIsNone(payload.dsa_topk_indices)
        self.assertEqual(batch.spec_info, "staged-handle")

    def test_non_spec_release_stashes_bonus_only(self):
        batch, future_map, build_seed, _ = self._run(SpeculativeAlgorithm.NONE)

        build_seed.assert_not_called()
        self.assertIsNone(batch.input_ids)
        payload = future_map.stash.call_args.args[1]
        self.assertIsInstance(payload, RelayPayload)
        self.assertIsNone(payload.topk_p)
        self.assertIsNone(payload.dsa_topk_indices)


class TestPaddedDraftPageTable(CustomTestCase):
    def test_rebuild_extends_to_padded_rows(self):
        # Planned metadata for 1 real request; DP padding made topk 2 rows.
        req_to_token = torch.arange(24, dtype=torch.int64).reshape(2, 12)
        pool = SimpleNamespace(req_to_token=req_to_token)
        metadata = SimpleNamespace(page_table_1=torch.ones(1, 4, dtype=torch.int64))
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0]),
            seq_lens=torch.tensor([5, 3]),
        )
        topk_indices = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int64)

        rebuilt = rebuild_padded_draft_page_table(
            metadata, forward_batch, topk_indices, pool
        )

        self.assertEqual(rebuilt.shape, (2, 6))
        self.assertTrue(
            torch.equal(rebuilt[0], req_to_token[1, :6]),
            "row 0 must map request 1's real tokens",
        )
        # The dummy padded row maps request 0's table row; its topk is all -1
        # so the downstream gather is inert for it.
        self.assertTrue(torch.equal(rebuilt[1], req_to_token[0, :6]))
        self.assertIs(metadata.page_table_1, rebuilt)

    def test_rebuild_preserves_planned_width(self):
        # Page table wider than max_seqlen+1 keeps the planned width.
        req_to_token = torch.arange(18, dtype=torch.int64).reshape(1, 18)
        pool = SimpleNamespace(req_to_token=req_to_token)
        metadata = SimpleNamespace(page_table_1=torch.zeros(1, 9, dtype=torch.int64))
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([2]),
        )
        rebuilt = rebuild_padded_draft_page_table(
            metadata, forward_batch, torch.zeros(1, 2, dtype=torch.int64), pool
        )
        self.assertEqual(rebuilt.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
