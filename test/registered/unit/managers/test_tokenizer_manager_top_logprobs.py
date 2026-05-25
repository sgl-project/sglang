"""Unit tests for ``TokenizerManager.detokenize_top_logprobs_tokens``.

Regression for https://github.com/sgl-project/sglang/issues/26286 — in
PD / disaggregated mode the sampler runs with ``no_copy_to_cpu=True``,
so per-position slots in both ``token_logprobs_val`` and
``token_logprobs_idx`` arrive as ``torch.Tensor`` rather than
``List[float]`` / ``List[int]``. The previous truthiness check
(``if token_logprobs_val[i]:``) raised ``RuntimeError: Boolean value
of Tensor with more than one value is ambiguous``, took down
``TokenizerManager``, and cascaded to a detokenizer SIGQUIT.

These tests pin five contracts:
  - PD-shape tensor val (only) does not crash and round-trips.
  - PD-shape tensor val + tensor idx normalizes both before
    delegating to the tokenizer.
  - Non-PD list slots keep the legacy behavior byte-for-byte.
  - Empty / None slots resolve to ``None`` in the output.
  - ``decode_to_text=True`` path forwards plain Python ints to
    ``tokenizer.batch_decode`` (catches the index-half of the bug).
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _bare_tokenizer_manager() -> TokenizerManager:
    """Bypass ``__init__`` so we can call the pure detokenize helper
    without spinning a real tokenizer / scheduler. Mirrors the minimal
    repro from issue #26286.
    """
    return TokenizerManager.__new__(TokenizerManager)


class TestDetokenizeTopLogprobsTokensTensorTruthiness(CustomTestCase):
    def test_pd_tensor_val_does_not_raise_and_round_trips_values(self) -> None:
        """The minimal repro from the issue: per-position values arrive
        as a multi-element tensor. Before the fix this raised
        ``RuntimeError: Boolean value of Tensor with more than one
        value is ambiguous`` at the truthiness check.
        """
        manager = _bare_tokenizer_manager()
        token_logprobs_val = [torch.tensor([-0.1, -0.2])]
        token_logprobs_idx = [[1, 2]]

        result = manager.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )

        self.assertEqual(len(result), 1)
        slot = result[0]
        self.assertIsNotNone(slot)
        # Tensor values must round-trip into plain Python floats so the
        # downstream JSON serializer doesn't choke on tensor instances.
        self.assertEqual(len(slot), 2)
        self.assertAlmostEqual(slot[0][0], -0.1, places=5)
        self.assertEqual(slot[0][1], 1)
        self.assertIsNone(slot[0][2])
        self.assertAlmostEqual(slot[1][0], -0.2, places=5)
        self.assertEqual(slot[1][1], 2)
        self.assertIsNone(slot[1][2])

    def test_pd_tensor_val_and_tensor_idx_both_normalized(self) -> None:
        """In PD mode the producer (``get_top_logprobs(..., no_copy_to_cpu=True)``)
        returns *both* values and indices as tensors. Normalizing only
        ``token_logprobs_val`` would still yield ``(float, tensor(1), None)``
        for ``decode_to_text=False`` and ``batch_decode([[tensor(1)]...])``
        for ``decode_to_text=True``. Pin that both halves are normalized
        to plain Python types before delegation.
        """
        manager = _bare_tokenizer_manager()
        token_logprobs_val = [torch.tensor([-0.1, -0.2])]
        token_logprobs_idx = [torch.tensor([1, 2])]

        result = manager.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )

        self.assertEqual(len(result), 1)
        slot = result[0]
        self.assertEqual(len(slot), 2)
        # Both halves are plain Python types -- assert the idx is a
        # raw int, not a 0-d tensor that quietly serializes wrong.
        self.assertIsInstance(slot[0][1], int)
        self.assertIsInstance(slot[1][1], int)
        self.assertEqual(slot[0][1], 1)
        self.assertEqual(slot[1][1], 2)

    def test_non_pd_list_slot_keeps_legacy_behavior(self) -> None:
        """Non-PD callers still pass ``List[float]`` per slot. The fix
        must not alter that path: a non-empty list still flows through
        ``detokenize_logprob_tokens``, and an empty list still resolves
        to ``None``.
        """
        manager = _bare_tokenizer_manager()
        token_logprobs_val = [[-0.5, -0.7], []]
        token_logprobs_idx = [[3, 4], []]

        result = manager.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )

        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertEqual(len(result[0]), 2)
        self.assertAlmostEqual(result[0][0][0], -0.5, places=5)
        self.assertEqual(result[0][0][1], 3)
        self.assertIsNone(result[1])

    def test_empty_tensor_slot_resolves_to_none(self) -> None:
        """A zero-element tensor (no top logprobs at this position)
        must resolve to ``None`` in the output, matching the empty-list
        signal in the non-PD path. Without ``.tolist()`` the truthiness
        on a zero-element tensor is also ambiguous.
        """
        manager = _bare_tokenizer_manager()
        token_logprobs_val = [torch.tensor([], dtype=torch.float32)]
        token_logprobs_idx = [torch.tensor([], dtype=torch.int64)]

        result = manager.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )

        self.assertEqual(result, [None])

    def test_mixed_tensor_and_none_slots(self) -> None:
        """The output preserves slot order: a real tensor slot
        produces a populated list, a ``None`` slot stays ``None``, an
        empty-tensor slot also stays ``None``.
        """
        manager = _bare_tokenizer_manager()
        token_logprobs_val = [
            torch.tensor([-0.05]),
            None,
            torch.tensor([], dtype=torch.float32),
            torch.tensor([-0.9, -1.1]),
        ]
        token_logprobs_idx = [
            torch.tensor([7]),
            None,
            torch.tensor([], dtype=torch.int64),
            torch.tensor([10, 11]),
        ]

        result = manager.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )

        self.assertEqual(len(result), 4)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertIsNotNone(result[3])
        self.assertEqual(len(result[3]), 2)

    def test_decode_to_text_path_forwards_plain_ints_to_tokenizer(self) -> None:
        """When ``decode_to_text=True`` the helper calls
        ``self.tokenizer.batch_decode`` once per slot. Pin that the
        idx values reaching the tokenizer are plain Python ints, not
        0-d tensors -- otherwise a real tokenizer would either raise
        or silently mis-decode. This catches the index-half of the
        bug that value-only normalization would miss.
        """
        manager = _bare_tokenizer_manager()
        # ``side_effect`` so we can assert the per-slot batch_decode
        # call args are plain Python ints (not torch scalars).
        manager.tokenizer = MagicMock()
        manager.tokenizer.batch_decode = MagicMock(return_value=["foo", "bar"])

        token_logprobs_val = [torch.tensor([-0.1, -0.2])]
        token_logprobs_idx = [torch.tensor([1, 2])]

        result = manager.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )

        # Exactly one batch_decode call (one slot), with plain-int ids
        # wrapped in single-element lists per upstream convention.
        manager.tokenizer.batch_decode.assert_called_once_with([[1], [2]])
        # And the returned shape is the full per-token tuple.
        self.assertEqual(len(result), 1)
        slot = result[0]
        self.assertEqual(len(slot), 2)
        self.assertAlmostEqual(slot[0][0], -0.1, places=5)
        self.assertEqual(slot[0][1], 1)
        self.assertEqual(slot[0][2], "foo")
        self.assertAlmostEqual(slot[1][0], -0.2, places=5)
        self.assertEqual(slot[1][1], 2)
        self.assertEqual(slot[1][2], "bar")


if __name__ == "__main__":
    unittest.main()
