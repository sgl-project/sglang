"""NGRAM corpus breadth must not select chain-only GDN verification."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNgramTreeMetadata(CustomTestCase):
    def test_backend_selects_tree_metadata_for_ngram_breadth_one(self):
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as backend

        runner = MagicMock()
        runner.device = "cpu"
        runner.is_draft_worker = False
        runner.server_args.enable_mis = False
        runner.req_to_token_pool.get_mamba_indices.side_effect = lambda ids: ids
        runner.runner_ids.state_ids.return_value.ids.side_effect = lambda ids: ids
        runner.token_to_kv_pool.layer_router.part_view.return_value = None

        mode = backend.ForwardMode.TARGET_VERIFY
        children = torch.tensor([[1, -1, 3, -1]])
        siblings = torch.tensor([[-1, 2, -1, -1]])
        forward_batch = SimpleNamespace(
            batch_size=1,
            forward_mode=mode,
            logical_forward_mode=mode,
            req_pool_indices=torch.tensor([1]),
            mamba_track_indices=None,
            mamba_track_mask=None,
            extend_start_loc=None,
            extend_seq_lens=None,
            ragged_verify_layout=None,
            input_ids=torch.tensor([16, 24, 16, 24]),
            spec_info=SimpleNamespace(
                draft_token_num=4,
                retrieve_next_token=children,
                retrieve_next_sibling=siblings,
            ),
        )
        cases = (
            ("NGRAM", 1, True),
            ("EAGLE", 1, False),
            ("EAGLE", 2, True),
            (None, None, False),
        )
        for algorithm, topk, expected in cases:
            with (
                self.subTest(algorithm=algorithm, topk=topk),
                patch.object(
                    backend,
                    "get_spec",
                    return_value=SimpleNamespace(
                        speculative_algorithm=algorithm,
                        speculative_eagle_topk=topk,
                    ),
                ),
                patch.object(
                    backend,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_unified_memory=False,
                    ),
                ),
            ):
                metadata = backend.MambaAttnBackendBase(runner)._forward_metadata(
                    forward_batch
                )
                self.assertEqual(metadata.query_start_loc.tolist(), [0, 4])
                if expected:
                    self.assertIs(metadata.retrieve_next_token, children)
                    self.assertIs(metadata.retrieve_next_sibling, siblings)
                    self.assertEqual(metadata.retrieve_parent_token.shape, (1, 4))
                else:
                    self.assertIsNone(metadata.retrieve_next_token)
                    self.assertIsNone(metadata.retrieve_next_sibling)
                    self.assertIsNone(metadata.retrieve_parent_token)


if __name__ == "__main__":
    unittest.main()
