"""CPU operator/lifecycle contracts; not CUDA Graph or TP parity evidence."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.models.dflash import CandidateSelector
from sglang.srt.speculative.dmtp_graph_tails import DMTPGraphSamplingTail
from sglang.srt.speculative.dmtp_selector_graph import native_proposal
from sglang.srt.speculative.mtp_prefix_contract import MTPPrefixConditioner


def cpu_topk(hidden, head, k):
    values, ids = (
        torch.nn.functional.linear(hidden, head.weight).float().topk(k, dim=-1)
    )
    return ids, values, None


class TailTest(unittest.TestCase):
    def fixture(self, t1):
        head = torch.nn.Linear(8, 32, bias=False)
        head.shard_indices = SimpleNamespace(
            num_org_elements=32,
            num_org_elements_padded=32,
            num_added_elements=0,
            org_vocab_start_index=0,
            added_vocab_start_index=32,
        )
        model = torch.nn.Module()
        model.candidate_selector = CandidateSelector(
            hidden_size=8, vocab_size=32, state_rank=4, top_k=2
        )
        model.draft_config = SimpleNamespace(
            output_multiplier=1.0, final_logit_softcapping=None
        )
        model._transform_unary_logits = lambda value: value
        model.mtp_prefix_conditioner = MTPPrefixConditioner(8, 4, 1e-6)
        embedding = torch.nn.Embedding(32, 8)
        target = SimpleNamespace(lm_head=head, get_input_embeddings=lambda: embedding)
        controller = SimpleNamespace(
            head=head,
            worker=SimpleNamespace(
                draft_model=model,
                target_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=target)
                ),
                server_args=SimpleNamespace(max_running_requests=3),
                _mask_token_id=31,
                _full_embed_gpu=None,
                _noise_embed_scale=1.0,
            ),
        )
        return DMTPGraphSamplingTail(controller, t1=t1)

    def test_t1_tail_preserves_original_q_and_all_eight_embeddings(self):
        group = SimpleNamespace(world_size=1, broadcast=lambda value, src: None)
        with (
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.srt.speculative.dmtp_graph_tails.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.candidate_topk",
                side_effect=cpu_topk,
            ),
        ):
            tail = self.fixture(True)
            tail.block_ids[:2, 0] = torch.tensor([2, 3])
            hidden = torch.randn(2, 8)
            actual = tail.forward(hidden)
            expected = native_proposal(
                tail.model,
                hidden[:, None],
                tail.block_ids[:2, 0],
                tail.temperatures[:2],
                tail.greedy_mask[:2],
                tail.coins[:2],
                t1=True,
                top_k=2,
            )
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            blocks = tail.embedding(tail.block_ids[:2])
            torch.testing.assert_close(
                actual[3], tail.condition(blocks, hidden), rtol=0, atol=0
            )
            torch.testing.assert_close(actual[3][:, 0], blocks[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(actual[3][:, 2:], blocks[:, 2:], rtol=0, atol=0)

    def test_df_tail_uses_next_scored_rows_and_captured_outputs_survive_reference(self):
        with (
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.candidate_topk",
                side_effect=cpu_topk,
            ),
        ):
            tail = self.fixture(False)
            hidden = torch.randn(2, 8, 8)
            blocks = torch.randint(0, 32, (2, 8))
            actual = tail.forward(hidden.flatten(0, 1), blocks.flatten())
            expected = native_proposal(
                tail.model,
                hidden[:, 1:7],
                blocks[:, 1],
                tail.temperatures[:2],
                tail.greedy_mask[:2],
                tail.coins[:2],
                t1=False,
                top_k=2,
            )
            tail.check(actual, expected)
            tail.freeze_outputs()
            tail.forward(torch.randn_like(hidden).flatten(0, 1), blocks.flatten())
            saved = tail.read(2)
            for value, reference in zip(saved, actual):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            actual[0].zero_()
            self.assertFalse(torch.equal(saved[0], actual[0]))

    def test_stage_selects_reordered_sampling_rows_and_rejects_changed_weights(self):
        group = SimpleNamespace(world_size=1, broadcast=lambda value, src: None)
        with (
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.srt.speculative.dmtp_graph_tails.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.srt.speculative.dmtp_graph_tails.resolve_greedy_mask",
                return_value=torch.tensor([False, True, False]),
            ),
        ):
            tail = self.fixture(True)
            info = SimpleNamespace(temperatures=torch.tensor([0.7, 1.0, 1.3]))
            tail.stage(torch.tensor([9, 4]), info, torch.tensor([2, 0]))
            self.assertEqual(
                tail.temperatures[:2].tolist(), info.temperatures[[2, 0]].tolist()
            )
            self.assertEqual(tail.block_ids[:2, 0].tolist(), [9, 4])
            self.assertEqual(tail.block_ids[:2, 2:].unique().tolist(), [31])
            with torch.no_grad():
                tail.model.lm_head.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, "weights/layout changed"):
                tail.stage(torch.tensor([1, 2, 3]), info)

    @patch(
        "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    )
    def test_stage_rejects_replaced_embedding_or_conditioner(self, _group):
        for replacement in ("embedding", "conditioner"):
            with self.subTest(replacement=replacement):
                tail = self.fixture(True)
                if replacement == "embedding":
                    tail.worker.target_worker.model_runner.model.get_input_embeddings = (
                        lambda: torch.nn.Embedding(32, 8)
                    )
                else:
                    tail.worker.draft_model.mtp_prefix_conditioner = (
                        MTPPrefixConditioner(8, 4, 1e-6)
                    )
                with self.assertRaisesRegex(RuntimeError, "weights/layout changed"):
                    tail.stage(
                        torch.tensor([1]),
                        SimpleNamespace(temperatures=torch.ones(1)),
                    )

    @patch(
        "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    )
    def test_stage_refreshes_sampling_state_without_reusing_previous_rows(self, _group):
        group = SimpleNamespace(world_size=1, broadcast=lambda value, src: None)
        with (
            patch(
                "sglang.srt.speculative.dmtp_graph_tails.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.srt.speculative.dmtp_graph_tails.resolve_greedy_mask",
                side_effect=[torch.tensor([False, True]), torch.tensor([True])],
            ),
        ):
            tail = self.fixture(True)
            tail.stage(
                torch.tensor([3, 4]),
                SimpleNamespace(temperatures=torch.tensor([0.7, 1.2])),
            )
            tail.stage(
                torch.tensor([9]), SimpleNamespace(temperatures=torch.tensor([0.5]))
            )
            torch.testing.assert_close(tail.temperatures[:1], torch.tensor([0.5]))
            self.assertTrue(tail.greedy_mask[0])
            self.assertEqual(tail.block_ids[0, 0].item(), 9)
            self.assertTrue((tail.block_ids[0, 1:] == 31).all())

    def test_dense_df_tail_keeps_its_different_six_scored_rows(self):
        with (
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.candidate_topk",
                side_effect=cpu_topk,
            ),
        ):
            tail = self.fixture(False)
            tail.scored_start = 2
            hidden = torch.randn(3, 8, 8)
            blocks = torch.randint(0, 32, (3, 8))
            actual = tail.forward(hidden.flatten(0, 1), blocks.flatten())
            expected = native_proposal(
                tail.model,
                hidden[:, 2:],
                blocks[:, 1],
                tail.temperatures[:3],
                tail.greedy_mask[:3],
                tail.coins[:3],
                t1=False,
                top_k=2,
            )
            tail.check(actual, expected)

    def test_dense_proposal_cache_is_request_indexed_and_seed_invalidates_reused_slots(
        self,
    ):
        from sglang.srt.speculative.native_mtp_bridge import NativeMTPBridge

        bridge = NativeMTPBridge.__new__(NativeMTPBridge)
        bridge.controller = SimpleNamespace(
            two_graphs=True,
            mtp_tail=SimpleNamespace(
                temperatures=torch.tensor([0.7, 1.2]),
                greedy_mask=torch.tensor([False, True]),
            ),
        )
        bridge.hidden_by_req = torch.empty(5, 8)
        bridge.anchor_by_req = torch.full((5,), -1, dtype=torch.int64)
        bridge.token_by_req = torch.full((5,), -1, dtype=torch.int64)
        bridge.proposal_valid = torch.zeros(5, dtype=torch.bool)
        values = (
            torch.tensor([[7], [11]]),
            torch.arange(4).view(2, 1, 2),
            torch.tensor([[[0.2, 0.8]], [[0.0, 1.0]]]),
            torch.randn(2, 8, 8),
        )
        bridge.proposal_buffers = tuple(
            torch.empty((5,) + v.shape[1:], dtype=v.dtype) for v in values
        )
        bridge.proposal_temperatures = torch.empty(5)
        bridge.proposal_greedy = torch.empty(5, dtype=torch.bool)
        reqs = torch.tensor([4, 1])
        bridge.seed(
            reqs, torch.randn(2, 8), torch.tensor([20, 30]), torch.tensor([2, 3])
        )
        bridge.seed_proposal(reqs, values)
        for buffer, value in zip(bridge.proposal_buffers, values):
            torch.testing.assert_close(buffer[[1, 4]], value[[1, 0]], rtol=0, atol=0)
        self.assertTrue(bridge.proposal_valid[reqs].all())
        bridge.seed(
            torch.tensor([1]), torch.randn(1, 8), torch.tensor([8]), torch.tensor([9])
        )
        self.assertFalse(bridge.proposal_valid[1])
        self.assertTrue(bridge.proposal_valid[4])


if __name__ == "__main__":
    unittest.main()
