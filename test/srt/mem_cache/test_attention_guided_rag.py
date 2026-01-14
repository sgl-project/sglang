"""
Unit tests for Attention-Guided RAG.

Tests:
1. Query analysis and retrieval need detection
2. Chunk scoring with fingerprint similarity
3. Target quality computation
4. Retrieval decision making
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestQueryAnalysis(unittest.TestCase):
    """Test query analysis for retrieval needs."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

        self.retriever = AttentionGuidedRetriever()

    def test_high_retrieval_need_query(self):
        """Test that factual queries have high retrieval need."""
        # Semantic bridge fingerprints with high entropy = searching for info
        fps = [
            np.array([0.15, 0.45, 0.40, 0.75] + [0.0] * 16),  # anchor
            np.array([0.70, 0.20, 0.10, 0.20] + [0.0] * 16),  # syntax
            np.array([0.15, 0.45, 0.40, 0.80] + [0.0] * 16),  # anchor
        ]
        zones = ["semantic_bridge", "syntax_floor", "semantic_bridge"]
        tokens = ["What", "is", "capital"]

        analysis = self.retriever.analyze_query(
            query="What is capital",
            fingerprints=fps,
            zones=zones,
            tokens=tokens,
        )

        # Should have high retrieval need
        self.assertGreater(analysis.retrieval_need_score, 0.4)
        # Should identify anchors
        self.assertGreater(len(analysis.anchor_tokens), 0)

    def test_low_retrieval_need_query(self):
        """Test that simple queries have low retrieval need."""
        # Syntax floor fingerprints with low entropy = self-contained
        fps = [
            np.array([0.70, 0.20, 0.10, 0.20] + [0.0] * 16),
            np.array([0.70, 0.20, 0.10, 0.20] + [0.0] * 16),
        ]
        zones = ["syntax_floor", "syntax_floor"]
        tokens = ["Hello", "!"]

        analysis = self.retriever.analyze_query(
            query="Hello!",
            fingerprints=fps,
            zones=zones,
            tokens=tokens,
        )

        # Should have low retrieval need
        self.assertLess(analysis.retrieval_need_score, 0.4)

    def test_empty_fingerprints_fallback(self):
        """Test fallback when no fingerprints available."""
        analysis = self.retriever.analyze_query(
            query="Test query",
            fingerprints=[],
            zones=[],
            tokens=[],
        )

        # Should have moderate default score
        self.assertEqual(analysis.retrieval_need_score, 0.5)


class TestChunkScoring(unittest.TestCase):
    """Test chunk scoring with fingerprint similarity."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

        self.retriever = AttentionGuidedRetriever()

    def test_relevant_chunk_scores_higher(self):
        """Test that relevant chunks score higher than irrelevant."""
        query = "What is the capital of France?"
        chunks = ["Paris is the capital of France.", "The weather is nice."]

        # Query fingerprints - semantic bridge (searching for info)
        query_fps = [np.array([0.15, 0.45, 0.40, 0.70] + [0.0] * 16)] * 5

        # Chunk fingerprints
        # Relevant chunk - good target (focused, low entropy)
        relevant_fps = [np.array([0.20, 0.50, 0.30, 0.25] + [0.0] * 16)] * 3
        # Irrelevant chunk - poor target (diffuse, high entropy)
        irrelevant_fps = [np.array([0.35, 0.30, 0.25, 0.80] + [0.0] * 16)] * 3

        analysis = self.retriever.analyze_query(query, query_fps)
        scores = self.retriever.score_chunks(
            query=query,
            chunks=chunks,
            query_analysis=analysis,
            chunk_fingerprints=[relevant_fps, irrelevant_fps],
        )

        # Relevant chunk should score higher (be first after sorting)
        self.assertEqual(scores[0].chunk, chunks[0])
        self.assertGreater(scores[0].final_score, scores[1].final_score)

    def test_target_quality_affects_score(self):
        """Test that target quality contributes to final score."""
        query = "Test query"
        chunks = ["Chunk 1", "Chunk 2"]

        query_fps = [np.array([0.20, 0.40, 0.40, 0.50] + [0.0] * 16)]

        # Same similarity, different target quality
        high_quality_fps = [np.array([0.20, 0.50, 0.30, 0.20] + [0.0] * 16)]  # Low entropy
        low_quality_fps = [np.array([0.35, 0.30, 0.25, 0.90] + [0.0] * 16)]  # High entropy

        analysis = self.retriever.analyze_query(query, query_fps)
        scores = self.retriever.score_chunks(
            query=query,
            chunks=chunks,
            query_analysis=analysis,
            chunk_fingerprints=[high_quality_fps, low_quality_fps],
        )

        # High quality target should have higher target_quality score
        high_q_score = next(s for s in scores if s.chunk == "Chunk 1")
        low_q_score = next(s for s in scores if s.chunk == "Chunk 2")

        self.assertGreater(high_q_score.target_quality, low_q_score.target_quality)


class TestZoneClassification(unittest.TestCase):
    """Test zone classification from fingerprints."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

        self.retriever = AttentionGuidedRetriever()

    def test_syntax_floor_classification(self):
        """Test syntax_floor zone classification."""
        # High local mass, low entropy
        fp = np.array([0.75, 0.15, 0.10, 0.15] + [0.0] * 16)
        zone = self.retriever._classify_zone(fp)
        self.assertEqual(zone, "syntax_floor")

    def test_semantic_bridge_classification(self):
        """Test semantic_bridge zone classification."""
        # High mid mass
        fp = np.array([0.15, 0.55, 0.30, 0.50] + [0.0] * 16)
        zone = self.retriever._classify_zone(fp)
        self.assertEqual(zone, "semantic_bridge")

    def test_long_range_classification(self):
        """Test long_range zone classification."""
        # High long mass
        fp = np.array([0.10, 0.25, 0.65, 0.40] + [0.0] * 16)
        zone = self.retriever._classify_zone(fp)
        self.assertEqual(zone, "long_range")

    def test_diffuse_classification(self):
        """Test diffuse zone classification."""
        # Low total mass
        fp = np.array([0.10, 0.15, 0.10, 0.80] + [0.0] * 16)
        zone = self.retriever._classify_zone(fp)
        self.assertEqual(zone, "diffuse")


class TestRetrievalDecision(unittest.TestCase):
    """Test retrieval decision making."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

        self.retriever = AttentionGuidedRetriever(min_retrieval_need=0.4)

    def test_should_retrieve_high_need(self):
        """Test that high need queries trigger retrieval."""
        # High retrieval need fingerprints
        fps = [np.array([0.15, 0.45, 0.40, 0.80] + [0.0] * 16)] * 5

        analysis = self.retriever.analyze_query(
            "What is the GDP of Japan?",
            fingerprints=fps,
            zones=["semantic_bridge"] * 5,
        )

        self.assertGreaterEqual(analysis.retrieval_need_score, 0.4)

    def test_should_not_retrieve_low_need(self):
        """Test that low need queries skip retrieval."""
        # Low retrieval need fingerprints
        fps = [np.array([0.70, 0.20, 0.10, 0.20] + [0.0] * 16)] * 3

        analysis = self.retriever.analyze_query(
            "Hello!",
            fingerprints=fps,
            zones=["syntax_floor"] * 3,
        )

        self.assertLess(analysis.retrieval_need_score, 0.4)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity computation."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

        self.retriever = AttentionGuidedRetriever()

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        v = np.array([1.0, 2.0, 3.0])
        sim = self.retriever._cosine_similarity(v, v)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = self.retriever._cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        sim = self.retriever._cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_different_length_vectors(self):
        """Test similarity with different length vectors (should pad)."""
        v1 = np.array([1.0, 2.0])
        v2 = np.array([1.0, 2.0, 0.0])
        sim = self.retriever._cosine_similarity(v1, v2)
        # After padding v1 to [1, 2, 0], should be identical
        self.assertAlmostEqual(sim, 1.0, places=5)


class TestHelperFunctions(unittest.TestCase):
    """Test module helper functions."""

    def test_analyze_retrieval_need(self):
        """Test convenience function for retrieval need analysis."""
        from sglang.srt.mem_cache.attention_guided_rag import analyze_retrieval_need

        fps = [
            np.array([0.15, 0.45, 0.40, 0.75] + [0.0] * 16),
            np.array([0.70, 0.20, 0.10, 0.20] + [0.0] * 16),
        ]
        zones = ["semantic_bridge", "syntax_floor"]

        score, anchors = analyze_retrieval_need("Test", fps, zones)

        self.assertIsInstance(score, float)
        self.assertIsInstance(anchors, list)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_chunk_relevance(self):
        """Test convenience function for chunk relevance scoring."""
        from sglang.srt.mem_cache.attention_guided_rag import score_chunk_relevance

        query_fps = [np.array([0.20, 0.40, 0.40, 0.50] + [0.0] * 16)]
        chunk_fps = [np.array([0.20, 0.40, 0.40, 0.50] + [0.0] * 16)]

        score = score_chunk_relevance(query_fps, chunk_fps)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)


class TestPhraseExtraction(unittest.TestCase):
    """Test phrase extraction from anchor tokens."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

        self.retriever = AttentionGuidedRetriever()

    def test_consecutive_anchors_form_phrase(self):
        """Test that consecutive anchors are grouped into phrases."""
        anchor_tokens = [(0, "machine", 0.8), (1, "learning", 0.9), (3, "algorithms", 0.7)]
        tokens = ["machine", "learning", "and", "algorithms"]

        phrases = self.retriever._extract_phrases(anchor_tokens, tokens)

        self.assertEqual(len(phrases), 2)  # "machine learning" and "algorithms"
        self.assertIn("machine learning", phrases)
        self.assertIn("algorithms", phrases)

    def test_empty_anchors(self):
        """Test phrase extraction with no anchors."""
        phrases = self.retriever._extract_phrases([], ["hello", "world"])
        self.assertEqual(phrases, [])


if __name__ == "__main__":
    unittest.main()
