"""
Unit tests for VocabIntersectionMapper used in heterogeneous vocabulary speculative decoding.
"""

import unittest

import torch

from sglang.srt.speculative.vocab_intersection import (
    VocabIntersectionMapper,
    create_vocab_mapper,
)


class MockTokenizer:
    """Mock tokenizer for testing vocabulary intersection."""

    def __init__(self, vocab: dict):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab

    def __len__(self):
        return len(self._vocab)


class TestVocabIntersectionMapper(unittest.TestCase):
    """Test VocabIntersectionMapper functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Draft vocab: tokens A, B, C, D with IDs 0, 1, 2, 3
        self.draft_vocab = {"A": 0, "B": 1, "C": 2, "D": 3}
        # Target vocab: tokens A, B, E, F with IDs 10, 11, 12, 13
        # Intersection: A, B (draft 0->target 10, draft 1->target 11)
        self.target_vocab = {"A": 10, "B": 11, "E": 12, "F": 13}

        self.draft_tokenizer = MockTokenizer(self.draft_vocab)
        self.target_tokenizer = MockTokenizer(self.target_vocab)

        self.device = "cpu"  # Use CPU for unit tests

    def test_intersection_computation(self):
        """Test that vocabulary intersection is computed correctly."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Check intersection size
        self.assertEqual(mapper.intersection_size, 2)  # A and B
        self.assertEqual(mapper.draft_vocab_size, 4)
        self.assertEqual(mapper.target_vocab_size, 4)

        # Check shared tokens
        self.assertEqual(mapper.shared_tokens, {"A", "B"})

    def test_draft_to_target_mapping(self):
        """Test mapping from draft vocab to target vocab."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Map draft tokens to target tokens
        draft_tokens = torch.tensor([0, 1], device=self.device)  # A, B
        target_tokens = mapper.map_draft_to_target(draft_tokens)

        self.assertEqual(target_tokens[0].item(), 10)  # A: 0 -> 10
        self.assertEqual(target_tokens[1].item(), 11)  # B: 1 -> 11

    def test_target_to_draft_mapping(self):
        """Test mapping from target vocab to draft vocab."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Map target tokens to draft tokens
        target_tokens = torch.tensor([10, 11], device=self.device)  # A, B
        draft_tokens = mapper.map_target_to_draft(target_tokens)

        self.assertEqual(draft_tokens[0].item(), 0)  # A: 10 -> 0
        self.assertEqual(draft_tokens[1].item(), 1)  # B: 11 -> 1

    def test_target_to_draft_fallback(self):
        """Test fallback for tokens not in intersection."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Token E (12) is not in intersection, should return fallback_id (default 0)
        target_tokens = torch.tensor([12], device=self.device)  # E
        draft_tokens = mapper.map_target_to_draft(target_tokens, fallback_id=0)

        self.assertEqual(draft_tokens[0].item(), 0)

    def test_draft_vocab_mask(self):
        """Test that draft vocab mask correctly identifies intersection tokens."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        mask = mapper.draft_vocab_mask
        self.assertEqual(mask.shape[0], 4)  # draft_vocab_size

        # A (0) and B (1) are in intersection
        self.assertTrue(mask[0].item())  # A
        self.assertTrue(mask[1].item())  # B
        # C (2) and D (3) are not in intersection
        self.assertFalse(mask[2].item())  # C
        self.assertFalse(mask[3].item())  # D

    def test_mask_draft_logits(self):
        """Test masking draft logits to intersection tokens."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Create fake logits
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device)
        masked_logits = mapper.mask_draft_logits(logits)

        # A and B should be unchanged, C and D should be -inf
        self.assertEqual(masked_logits[0, 0].item(), 1.0)  # A
        self.assertEqual(masked_logits[0, 1].item(), 2.0)  # B
        self.assertEqual(masked_logits[0, 2].item(), float("-inf"))  # C
        self.assertEqual(masked_logits[0, 3].item(), float("-inf"))  # D

    def test_mask_draft_logits_inplace(self):
        """Test in-place masking of draft logits."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device)
        mapper.mask_draft_logits_inplace(logits)

        self.assertEqual(logits[0, 2].item(), float("-inf"))
        self.assertEqual(logits[0, 3].item(), float("-inf"))

    def test_intersection_ratio(self):
        """Test intersection ratio calculation."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        ratio = mapper.get_intersection_ratio()
        self.assertEqual(ratio, 0.5)  # 2 out of 4 draft tokens

    def test_create_vocab_mapper_warning(self):
        """Test that create_vocab_mapper logs warning for small intersection."""
        # Create vocabs with small intersection
        draft_vocab = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9,
        }
        target_vocab = {"A": 0}  # Only A in common = 10% intersection

        draft_tokenizer = MockTokenizer(draft_vocab)
        target_tokenizer = MockTokenizer(target_vocab)

        # Should still create mapper but log warning
        mapper = create_vocab_mapper(
            draft_tokenizer,
            target_tokenizer,
            device=self.device,
            min_intersection_ratio=0.1,
        )

        self.assertIsNotNone(mapper)
        self.assertEqual(mapper.intersection_size, 1)


class TestVocabIntersectionMapperBatched(unittest.TestCase):
    """Test VocabIntersectionMapper with batched inputs."""

    def setUp(self):
        """Set up test fixtures with larger vocab."""
        # Simulate more realistic vocab sizes
        self.draft_vocab = {f"token_{i}": i for i in range(100)}
        self.target_vocab = {f"token_{i}": i + 1000 for i in range(50)}  # 50% overlap

        self.draft_tokenizer = MockTokenizer(self.draft_vocab)
        self.target_tokenizer = MockTokenizer(self.target_vocab)
        self.device = "cpu"

    def test_batched_mapping(self):
        """Test mapping with batched tensor inputs."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Batch of draft tokens
        draft_tokens = torch.tensor([[0, 1, 2], [3, 4, 5]], device=self.device)
        target_tokens = mapper.map_draft_to_target(draft_tokens)

        self.assertEqual(target_tokens.shape, draft_tokens.shape)
        # token_0: 0 -> 1000, token_1: 1 -> 1001, etc.
        self.assertEqual(target_tokens[0, 0].item(), 1000)
        self.assertEqual(target_tokens[0, 1].item(), 1001)

    def test_batched_logits_masking(self):
        """Test logits masking with batched inputs."""
        mapper = VocabIntersectionMapper(
            self.draft_tokenizer, self.target_tokenizer, device=self.device
        )

        # Batch of logits [batch_size, vocab_size]
        batch_size = 4
        logits = torch.randn(batch_size, 100, device=self.device)
        masked_logits = mapper.mask_draft_logits(logits)

        # Check that non-intersection tokens are masked
        for i in range(batch_size):
            for j in range(50, 100):  # tokens 50-99 are not in intersection
                self.assertEqual(masked_logits[i, j].item(), float("-inf"))


if __name__ == "__main__":
    unittest.main()
