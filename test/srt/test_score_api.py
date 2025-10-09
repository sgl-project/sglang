import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestScoreAPI(CustomTestCase):
    """Test the scoring API functionality."""

    def setUp(self):
        """Set up each test case."""
        self.engine = Engine(model_path=TEST_MODEL_NAME)

    def tearDown(self):
        """Clean up after each test case."""
        if self.engine is not None:
            self.engine.shutdown()
            torch.cuda.empty_cache()

    def compute_hf_scores(
        self, query, items, label_token_ids, apply_softmax=False, item_first=False
    ):
        """Compute scores using direct HuggingFace model inference.
        Returns probabilities for each token ID, optionally normalized with softmax.

        Args:
            query: The query text
            items: List of item texts
            label_token_ids: List of token IDs to compute probabilities for
            apply_softmax: Whether to normalize probabilities using softmax
            item_first: If True, prepend items to query. Otherwise append items to query.
        """
        # Initialize HF model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            TEST_MODEL_NAME, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            TEST_MODEL_NAME, trust_remote_code=True
        )

        try:
            scores = []
            for item in items:
                # Construct full text based on item_first parameter
                full_text = f"{item}{query}" if item_first else f"{query}{item}"
                inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

                # Get logits for the last token
                with torch.no_grad():
                    outputs = model(**inputs)
                    last_token_logits = outputs.logits[0, -1]

                # Get logits for just our target tokens
                target_logits = last_token_logits[label_token_ids]

                # Apply softmax over just the target tokens
                target_probs = torch.softmax(target_logits, dim=-1)

                # Convert to list of probabilities in order of label_token_ids
                probs = [target_probs[i].item() for i in range(len(label_token_ids))]

                scores.append(probs)

            return scores
        finally:
            # Clean up HF resources
            model.cpu()
            del model
            del tokenizer
            torch.cuda.empty_cache()

    def _get_token_ids(self, tokens):
        """Helper method to get token IDs for a list of tokens."""
        tokenizer = AutoTokenizer.from_pretrained(
            TEST_MODEL_NAME, trust_remote_code=True
        )
        try:
            label_token_ids = []
            for token in tokens:
                encoding = tokenizer.encode_plus(token, add_special_tokens=False)
                token_ids = encoding["input_ids"]
                label_token_ids.append(token_ids[0])
            return label_token_ids
        finally:
            del tokenizer

    def _compare_scores(self, hf_scores, sglang_scores, label_token_ids, case_name=""):
        """Helper method to compare scores between HF and SGLang using relative tolerance."""
        self.assertEqual(
            len(hf_scores),
            len(sglang_scores),
            f"Score lengths don't match for {case_name}",
        )

        # Use a relative tolerance of 1%
        TOLERANCE = 0.01

        for hf_score_list, sglang_score_list in zip(hf_scores, sglang_scores):
            self.assertEqual(
                len(hf_score_list),
                len(sglang_score_list),
                f"Score list lengths don't match for {case_name}",
            )

            for hf_score, sglang_score in zip(hf_score_list, sglang_score_list):
                diff = abs(hf_score - sglang_score)
                self.assertLessEqual(
                    diff,
                    TOLERANCE,
                    msg=f"Scores differ by {diff:.2%} ({case_name}): "
                    f"HF={hf_score:.6f}, SGLang={sglang_score:.6f}",
                )

                self.assertGreaterEqual(
                    sglang_score, 0, f"SGLang score {sglang_score:.6f} not in [0,1]"
                )
                self.assertLessEqual(
                    sglang_score, 1, f"SGLang score {sglang_score:.6f} not in [0,1]"
                )

            self.assertAlmostEqual(
                sum(sglang_score_list),
                1.0,
                places=6,
                msg=f"SGLang scores don't sum to 1 ({case_name}): {sum(sglang_score_list):.6f}",
            )

    def test_score_consistency(self):
        """Test that SGLang scoring matches direct HuggingFace model scoring."""
        # Define test cases
        test_cases = [
            {
                "name": "default case",
                "query": "I pledge allegiance",
                "items": ["", " to"],
                "item_first": False,
            },
            {
                "name": "item_first case",
                "query": " is a city",
                "items": ["Tokyo", "Japan"],
                "item_first": True,
            },
        ]

        # Common tokens to test for all cases
        tokens = [" to", " the"]
        label_token_ids = self._get_token_ids(tokens)

        # Run each test case
        for case in test_cases:
            # Get scores from SGLang
            sglang_scores = self.engine.score(
                query=case["query"],
                items=case["items"],
                label_token_ids=label_token_ids,
                apply_softmax=True,
                item_first=case["item_first"],
            )

            # Get scores from HuggingFace using the same parameters
            hf_scores = self.compute_hf_scores(
                query=case["query"],
                items=case["items"],
                label_token_ids=label_token_ids,
                apply_softmax=True,
                item_first=case["item_first"],
            )

            # Compare scores
            self._compare_scores(
                hf_scores, sglang_scores, label_token_ids, case["name"]
            )

    def test_score_batch_handling(self):
        """Test that batch scoring works correctly."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        label_token_ids = [1, 2, 3]

        for batch_size in batch_sizes:
            texts = [f"test {i}" for i in range(batch_size)]
            scores = self.engine.score(
                query="The test was",
                items=texts,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )

            self.assertEqual(
                len(scores),
                batch_size,
                f"Expected {batch_size} scores, got {len(scores)}",
            )

            # Verify each score list has the correct length
            for score_list in scores:
                self.assertEqual(
                    len(score_list),
                    len(label_token_ids),
                    f"Score list length {len(score_list)} doesn't match label_token_ids length {len(label_token_ids)}",
                )
                self.assertTrue(
                    all(isinstance(v, float) for v in score_list),
                    "All scores should be floats",
                )
                self.assertAlmostEqual(
                    1.0, sum(score_list), 6, "Scores should sum to 1"
                )

    def test_score_request_construction(self):
        """Test that scoring requests are constructed to avoid decode phase."""
        from unittest.mock import patch

        # Capture the internal request to verify optimization
        captured_requests = []
        original_gen = self.engine.tokenizer_manager.generate_request

        async def mock_generate_request(req, request=None):
            captured_requests.append(req)
            async for result in original_gen(req, request):
                yield result

        # Patch the generate_request method
        with patch.object(
            self.engine.tokenizer_manager,
            "generate_request",
            side_effect=mock_generate_request,
        ):
            # Run a scoring request
            query = "What is the capital of"
            items = ["France", "Germany"]
            label_token_ids = [1, 2, 3]

            scores = self.engine.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )

            # Verify we got results
            self.assertEqual(len(scores), len(items))

            # Verify the captured request has decode-avoiding properties
            self.assertEqual(len(captured_requests), 1)
            request = captured_requests[0]

            # Key assertions for decode phase avoidance:
            # 1. max_new_tokens should be 0 (prevents token generation)
            # Handle both single and batch request cases
            if isinstance(request.sampling_params, dict):
                max_new_tokens = request.sampling_params.get("max_new_tokens", 0)
            elif isinstance(request.sampling_params, list):
                # For batch requests, check the first item
                max_new_tokens = request.sampling_params[0].get("max_new_tokens", 0)
            else:
                max_new_tokens = getattr(request.sampling_params, "max_new_tokens", 0)

            self.assertEqual(
                max_new_tokens, 0, "max_new_tokens should be 0 to avoid decode phase"
            )

            # 2. Should have token_ids_logprob for scoring
            # Handle both single and batch request cases
            if (
                isinstance(request.token_ids_logprob, list)
                and len(request.token_ids_logprob) > 0
                and isinstance(request.token_ids_logprob[0], list)
            ):
                # Batch case: token_ids_logprob is a list of lists
                # Each item in the batch should have the same label_token_ids
                for item_token_ids in request.token_ids_logprob:
                    self.assertEqual(
                        item_token_ids,
                        label_token_ids,
                        "Each batch item should have label_token_ids for scoring",
                    )
            else:
                # Single request case
                self.assertEqual(
                    request.token_ids_logprob,
                    label_token_ids,
                    "Should have label_token_ids for scoring",
                )

            # 3. Should request logprobs but not stream
            self.assertTrue(
                request.return_logprob, "Should request logprobs for scoring"
            )
            self.assertFalse(request.stream, "Scoring requests should not stream")

    def test_multi_item_scoring_basic(self):
        """Test basic multi-item scoring functionality."""
        # Test with a simple query and items
        query = "What is the capital of California? Answer Yes or No for each of the following options:"
        items = ["Sacramento", "San Jose", "San Francisco"]
        label_token_ids = [9454, 2753]  # "Yes" and "No" tokens

        # Get scores using SGLang
        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        # Verify we get the expected number of scores
        self.assertEqual(len(scores), len(items), "Should get one score list per item")

        # Verify each score list has the correct length
        for i, score_list in enumerate(scores):
            self.assertEqual(
                len(score_list),
                len(label_token_ids),
                f"Item {i} should have {len(label_token_ids)} scores",
            )
            # Verify scores are probabilities (sum to 1)
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Scores for item {i} should sum to 1",
            )
            # Verify all scores are non-negative
            for j, score in enumerate(score_list):
                self.assertGreaterEqual(
                    score, 0, f"Score {j} for item {i} should be non-negative"
                )

    def test_multi_item_scoring_consistency(self):
        """Test that multi-item scoring gives consistent results."""
        query = "Choose the best option:"
        items = ["Option A", "Option B", "Option C"]
        label_token_ids = [1, 2, 3]

        # Run the same test multiple times
        scores1 = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        scores2 = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        # Results should be identical (deterministic)
        self.assertEqual(len(scores1), len(scores2), "Should get same number of items")
        for i, (s1, s2) in enumerate(zip(scores1, scores2)):
            self.assertEqual(
                len(s1), len(s2), f"Item {i} should have same number of scores"
            )
            for j, (score1, score2) in enumerate(zip(s1, s2)):
                self.assertAlmostEqual(
                    score1,
                    score2,
                    places=6,
                    msg=f"Score {j} for item {i} should be identical",
                )

    def test_multi_item_scoring_different_sizes(self):
        """Test multi-item scoring with different numbers of items."""
        query = "Rate each option:"
        label_token_ids = [1, 2, 3, 4, 5]

        # Test with different numbers of items
        test_cases = [
            ["Single item"],
            ["Item 1", "Item 2"],
            ["A", "B", "C", "D"],
            ["X", "Y", "Z", "W", "V", "U"],
        ]

        for items in test_cases:
            with self.subTest(items=items):
                scores = self.engine.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                )

                self.assertEqual(
                    len(scores), len(items), f"Should get {len(items)} score lists"
                )

                for i, score_list in enumerate(scores):
                    self.assertEqual(
                        len(score_list),
                        len(label_token_ids),
                        f"Item {i} should have {len(label_token_ids)} scores",
                    )
                    self.assertAlmostEqual(sum(score_list), 1.0, places=6)

    def test_multi_item_scoring_empty_items(self):
        """Test multi-item scoring with empty items list."""
        query = "Test query"
        items = []
        label_token_ids = [1, 2]

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), 0, "Should return empty list for empty items")

    def test_multi_item_scoring_single_item(self):
        """Test multi-item scoring with single item (should work like regular scoring)."""
        query = "Complete this sentence: The capital of France is"
        items = ["Paris"]
        label_token_ids = [1, 2, 3]

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), 1, "Should get one score list")
        self.assertEqual(
            len(scores[0]), len(label_token_ids), "Should have correct number of scores"
        )
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=6)

    def test_multi_item_scoring_different_queries(self):
        """Test multi-item scoring with different types of queries."""
        items = ["Yes", "No"]
        label_token_ids = [1, 2]

        test_queries = [
            "Is this true?",
            "Choose the correct answer:",
            "What is the best option?",
            "Select all that apply:",
            "",  # Empty query
        ]

        for query in test_queries:
            with self.subTest(query=query):
                scores = self.engine.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                )

                self.assertEqual(
                    len(scores),
                    len(items),
                    f"Should get {len(items)} score lists for query: '{query}'",
                )

                for i, score_list in enumerate(scores):
                    self.assertEqual(len(score_list), len(label_token_ids))
                    self.assertAlmostEqual(sum(score_list), 1.0, places=6)

    def test_multi_item_scoring_different_label_tokens(self):
        """Test multi-item scoring with different label token sets."""
        query = "Choose the best option:"
        items = ["Option A", "Option B"]

        test_label_tokens = [
            [1, 2],  # Two tokens
            [1, 2, 3, 4],  # Four tokens
            [1],  # Single token
            [1, 2, 3, 4, 5, 6, 7, 8],  # Many tokens
        ]

        for label_token_ids in test_label_tokens:
            with self.subTest(label_tokens=label_token_ids):
                scores = self.engine.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                )

                self.assertEqual(len(scores), len(items))

                for i, score_list in enumerate(scores):
                    self.assertEqual(
                        len(score_list),
                        len(label_token_ids),
                        f"Item {i} should have {len(label_token_ids)} scores",
                    )
                    self.assertAlmostEqual(sum(score_list), 1.0, places=6)

    def test_multi_item_scoring_without_softmax(self):
        """Test multi-item scoring without softmax normalization."""
        query = "Rate each option:"
        items = ["Good", "Bad", "Neutral"]
        label_token_ids = [1, 2, 3]

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=False,  # No softmax
        )

        self.assertEqual(len(scores), len(items))

        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(label_token_ids))
            # Without softmax, scores don't need to sum to 1
            # But they should still be valid logits/probabilities
            for j, score in enumerate(score_list):
                self.assertIsInstance(
                    score, (int, float), f"Score {j} for item {i} should be numeric"
                )

    def test_multi_item_scoring_large_batch(self):
        """Test multi-item scoring with a large number of items."""
        query = "Classify each item:"
        items = [f"Item {i}" for i in range(20)]  # 20 items
        label_token_ids = [1, 2, 3]

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), len(items), "Should handle large batches")

        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=6)

    def test_multi_item_scoring_unicode(self):
        """Test multi-item scoring with unicode characters."""
        query = "选择最佳选项："
        items = ["选项A", "选项B", "选项C"]
        label_token_ids = [1, 2, 3]

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), len(items))

        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=6)

    def test_multi_item_scoring_error_handling(self):
        """Test multi-item scoring error handling."""
        query = "Test query"
        items = ["Item 1", "Item 2"]
        label_token_ids = [1, 2]

        # Test with invalid label_token_ids
        with self.assertRaises((ValueError, TypeError)):
            self.engine.score(
                query=query,
                items=items,
                label_token_ids="invalid",  # Should be list of ints
                apply_softmax=True,
            )

        # Test with None items
        with self.assertRaises((ValueError, TypeError)):
            self.engine.score(
                query=query,
                items=None,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )


if __name__ == "__main__":
    unittest.main()
