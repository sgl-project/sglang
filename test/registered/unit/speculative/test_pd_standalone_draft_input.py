"""CPU contracts for the first STANDALONE draft input after PD admission.

These tests exercise the public algorithm dispatch and real tensor conversion.
The overlap boundary is recorded; device events and KV readiness need GPU tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.speculative import standalone_disaggregation
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB_SIZE = 11


def _request(rid, probability, token):
    # Metadata has fixed capacity; only the configured top-k is the seed.
    # Deliberately omit hidden_states_tensor: STANDALONE must not read it.
    return SimpleNamespace(
        rid=rid,
        output_topk_p=[probability] + [0.0] * 15,
        output_topk_index=[token] + [0] * 15,
    )


def _batch(reqs, *, overlap=False):
    return SimpleNamespace(
        reqs=reqs,
        device=torch.device("cpu"),
        enable_overlap=overlap,
        model_config=SimpleNamespace(vocab_size=VOCAB_SIZE),
        req_pool_indices=torch.tensor([5, 2][: len(reqs)]),
        seq_lens=torch.tensor([13, 7][: len(reqs)]),
    )


class TestPdStandaloneDraftInput(unittest.TestCase):
    def build(self, batch, future_map=None):
        bonus_tokens = torch.tensor([4, 8][: len(batch.reqs)])
        with patch.object(
            standalone_disaggregation,
            "get_spec",
            return_value=SimpleNamespace(speculative_eagle_topk=1),
        ):
            result = SpeculativeAlgorithm.STANDALONE.build_disagg_draft_input(
                batch, bonus_tokens, future_map
            )
        self.assertIsInstance(result, EagleDraftInput)
        self.assertIsNone(result.hidden_states)
        self.assertEqual(result.capture_hidden_mode, CaptureHiddenMode.NULL)
        torch.testing.assert_close(result.bonus_tokens, bonus_tokens)
        return result

    def test_sync_recovers_real_seed_without_hidden_states(self):
        future_map = Mock()
        result = self.build(_batch([_request("a", 0.75, 3)]), future_map)
        torch.testing.assert_close(result.topk_p, torch.tensor([[0.75]]))
        torch.testing.assert_close(result.topk_index, torch.tensor([[3]]))
        self.assertEqual(result.topk_p.dtype, torch.float32)
        self.assertEqual(result.topk_index.dtype, torch.int64)
        self.assertEqual(result.topk_p.device.type, "cpu")
        self.assertEqual(future_map.mock_calls, [])

    def test_request_order_is_preserved(self):
        reqs = [_request("b", 0.25, 7), _request("a", 0.75, 3)]
        result = self.build(_batch(reqs))
        torch.testing.assert_close(result.topk_p, torch.tensor([[0.25], [0.75]]))
        torch.testing.assert_close(result.topk_index, torch.tensor([[7], [3]]))

    def test_overlap_hands_seed_to_pool_indexed_relay(self):
        batch = _batch([_request("a", 0.75, 3), _request("b", 0.25, 7)], overlap=True)
        future_map = Mock()
        result = self.build(batch, future_map)
        self.assertEqual(
            [call[0] for call in future_map.mock_calls], ["publish", "stash"]
        )
        indices, seq_lens = future_map.publish.call_args.args
        torch.testing.assert_close(indices, torch.tensor([5, 2]))
        torch.testing.assert_close(seq_lens, torch.tensor([13, 7]))
        indices, payload = future_map.stash.call_args.args
        torch.testing.assert_close(indices, torch.tensor([5, 2]))
        torch.testing.assert_close(result.future_indices, torch.tensor([5, 2]))
        torch.testing.assert_close(payload.topk_p, torch.tensor([[0.75], [0.25]]))
        torch.testing.assert_close(payload.topk_index, torch.tensor([[3], [7]]))
        torch.testing.assert_close(payload.bonus_tokens, torch.tensor([4, 8]))
        self.assertIsNone(payload.hidden_states)

    def test_missing_seed_rejects_batch_before_relay_publication(self):
        for field in ("output_topk_p", "output_topk_index"):
            with self.subTest(field=field):
                bad = _request("bad", 0.5, 2)
                setattr(bad, field, None)
                future_map = Mock()
                batch = _batch([_request("good", 0.75, 3), bad], overlap=True)
                with self.assertRaisesRegex(ValueError, "Missing.*bad"):
                    self.build(batch, future_map)
                self.assertEqual(future_map.mock_calls, [])

    def test_malformed_seed_shapes_are_rejected(self):
        for probabilities, indices in (([], []), ([[0.5]], [[2]]), ([0.5], [2, 3])):
            with self.subTest(probabilities=probabilities, indices=indices):
                req = SimpleNamespace(
                    rid="bad", output_topk_p=probabilities, output_topk_index=indices
                )
                with self.assertRaisesRegex(ValueError, "seed shape"):
                    self.build(_batch([req]))

    def test_unwritten_or_invalid_seed_is_rejected(self):
        cases = (
            (0.0, 0),
            (-0.5, 2),
            (float("nan"), 2),
            (float("inf"), 2),
            (0.5, -1),
            (0.5, VOCAB_SIZE),
        )
        for probability, token in cases:
            with self.subTest(probability=probability, token=token):
                with self.assertRaisesRegex(ValueError, "Invalid or unwritten"):
                    self.build(_batch([_request("bad", probability, token)]))


if __name__ == "__main__":
    unittest.main()
