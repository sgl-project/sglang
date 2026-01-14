"""
Attention-Guided RAG: Use attention fingerprints to improve retrieval.

Key Insight: Tokens classified as "semantic_bridge" in the attention manifold
are retrieval anchors - they're looking for information that isn't in the
immediate context. We can use this to:

1. **Identify retrieval need**: High-entropy tokens in semantic_bridge zone
   indicate the model is "searching" for information it doesn't have.

2. **Score chunks by attention coherence**: Retrieved chunks that would serve
   as good attention targets (low entropy, high long-range mass) are better.

3. **Re-rank by fingerprint similarity**: Chunks whose fingerprints align with
   the query's retrieval anchors are more relevant.

Usage:
    from sglang.srt.mem_cache.attention_guided_rag import AttentionGuidedRetriever

    retriever = AttentionGuidedRetriever(base_url="http://localhost:30000")

    # Analyze query to find retrieval-needing tokens
    analysis = retriever.analyze_query("What is the capital of France?")
    print(f"Retrieval anchors: {analysis.anchor_tokens}")
    print(f"Retrieval score: {analysis.retrieval_need_score}")

    # Score candidate chunks
    chunks = ["Paris is the capital of France.", "France is in Europe."]
    scores = retriever.score_chunks("What is the capital?", chunks)

    # Full RAG pipeline with attention-based re-ranking
    result = retriever.retrieve_and_generate(
        query="What is the capital of France?",
        documents=["Paris is the capital...", "London is...", ...],
        top_k=3
    )

Author: SGLang Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Fingerprint feature indices (from spectral_eviction.py)
FP_LOCAL_MASS = 0
FP_MID_MASS = 1
FP_LONG_MASS = 2
FP_ENTROPY = 3
FP_HISTOGRAM_START = 4

# Zone importance for retrieval (inverted from eviction - high entropy = needs retrieval)
ZONE_RETRIEVAL_NEED = {
    "semantic_bridge": 0.9,  # Looking for distant connections - high need
    "long_range": 0.7,  # Has some long-range - moderate need
    "structure_ripple": 0.5,  # Structural - low need
    "syntax_floor": 0.2,  # Local syntax - minimal need
    "diffuse": 0.8,  # Unfocused attention - needs grounding
    "unknown": 0.5,
}

# Zone scores for being a good retrieval TARGET (what makes a chunk useful)
ZONE_TARGET_QUALITY = {
    "semantic_bridge": 0.95,  # Great anchor point
    "long_range": 0.85,  # Good for connecting
    "structure_ripple": 0.6,  # Provides structure
    "syntax_floor": 0.4,  # Less useful for retrieval
    "diffuse": 0.3,  # Scattered attention
    "unknown": 0.5,
}


@dataclass
class QueryAnalysis:
    """Analysis of a query's retrieval needs based on attention patterns."""

    query: str
    tokens: List[str]
    fingerprints: List[np.ndarray]
    zones: List[str]

    # Tokens that are "searching" for information
    anchor_tokens: List[Tuple[int, str, float]] = field(default_factory=list)

    # Overall retrieval need score (0 = self-contained, 1 = needs retrieval)
    retrieval_need_score: float = 0.0

    # Key phrases extracted from anchor tokens
    retrieval_phrases: List[str] = field(default_factory=list)


@dataclass
class ChunkScore:
    """Score for a candidate chunk."""

    chunk: str
    chunk_idx: int

    # Component scores
    fingerprint_similarity: float = 0.0  # Similarity to query anchors
    target_quality: float = 0.0  # How good as retrieval target
    semantic_score: float = 0.0  # From embedding similarity (if available)

    # Combined score
    final_score: float = 0.0

    # Explanation
    explanation: str = ""


@dataclass
class RAGResult:
    """Result of attention-guided RAG."""

    query: str
    answer: str
    retrieved_chunks: List[ChunkScore]
    query_analysis: QueryAnalysis
    generation_fingerprints: Optional[List[np.ndarray]] = None


class AttentionGuidedRetriever:
    """
    Retriever that uses attention fingerprints to improve RAG quality.

    The key insight is that attention patterns reveal what information
    the model is "looking for" but doesn't have. By analyzing the query's
    attention fingerprints, we can:

    1. Identify which tokens are "retrieval anchors" (semantic_bridge zone)
    2. Score candidate chunks by how well they'd serve those anchors
    3. Re-rank retrieved documents for better relevance
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        fingerprint_weight: float = 0.4,
        embedding_weight: float = 0.4,
        target_quality_weight: float = 0.2,
        min_retrieval_need: float = 0.3,
    ):
        """
        Args:
            base_url: SGLang server URL
            fingerprint_weight: Weight for fingerprint similarity in scoring
            embedding_weight: Weight for embedding similarity (requires embeddings)
            target_quality_weight: Weight for chunk's quality as retrieval target
            min_retrieval_need: Minimum retrieval need score to trigger retrieval
        """
        self.base_url = base_url.rstrip("/")
        self.fingerprint_weight = fingerprint_weight
        self.embedding_weight = embedding_weight
        self.target_quality_weight = target_quality_weight
        self.min_retrieval_need = min_retrieval_need

        # Normalize weights
        total = fingerprint_weight + embedding_weight + target_quality_weight
        self.fingerprint_weight /= total
        self.embedding_weight /= total
        self.target_quality_weight /= total

    def analyze_query(
        self,
        query: str,
        fingerprints: Optional[List[np.ndarray]] = None,
        zones: Optional[List[str]] = None,
        tokens: Optional[List[str]] = None,
    ) -> QueryAnalysis:
        """
        Analyze a query to determine its retrieval needs.

        If fingerprints are not provided, will attempt to get them from server.

        Args:
            query: The query text
            fingerprints: Pre-computed fingerprints (optional)
            zones: Pre-computed zone classifications (optional)
            tokens: Tokenized query (optional)

        Returns:
            QueryAnalysis with retrieval need assessment
        """
        if fingerprints is None:
            # Get fingerprints from server
            fingerprints, zones, tokens = self._get_query_fingerprints(query)

        if fingerprints is None or len(fingerprints) == 0:
            # Fallback: assume moderate retrieval need
            return QueryAnalysis(
                query=query,
                tokens=tokens or [],
                fingerprints=[],
                zones=[],
                retrieval_need_score=0.5,
            )

        # Classify zones if not provided
        if zones is None:
            zones = [self._classify_zone(fp) for fp in fingerprints]

        # Find anchor tokens (high retrieval need)
        anchor_tokens = []
        retrieval_scores = []

        for i, (fp, zone) in enumerate(zip(fingerprints, zones)):
            # Retrieval need based on zone and entropy
            zone_need = ZONE_RETRIEVAL_NEED.get(zone, 0.5)
            entropy = fp[FP_ENTROPY] if len(fp) > FP_ENTROPY else 0.5

            # High entropy in semantic_bridge = searching for info
            retrieval_need = zone_need * (0.5 + 0.5 * min(entropy, 1.0))
            retrieval_scores.append(retrieval_need)

            if retrieval_need > 0.6:
                token_text = tokens[i] if tokens and i < len(tokens) else f"[{i}]"
                anchor_tokens.append((i, token_text, retrieval_need))

        # Overall retrieval need
        overall_need = np.mean(retrieval_scores) if retrieval_scores else 0.5

        # Extract phrases from anchor tokens
        phrases = self._extract_phrases(anchor_tokens, tokens)

        return QueryAnalysis(
            query=query,
            tokens=tokens or [],
            fingerprints=fingerprints,
            zones=zones,
            anchor_tokens=anchor_tokens,
            retrieval_need_score=float(overall_need),
            retrieval_phrases=phrases,
        )

    def score_chunks(
        self,
        query: str,
        chunks: List[str],
        query_analysis: Optional[QueryAnalysis] = None,
        chunk_fingerprints: Optional[List[List[np.ndarray]]] = None,
        chunk_embeddings: Optional[np.ndarray] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[ChunkScore]:
        """
        Score candidate chunks for relevance to query.

        Args:
            query: The query text
            chunks: List of candidate chunks
            query_analysis: Pre-computed query analysis (optional)
            chunk_fingerprints: Pre-computed fingerprints for each chunk (optional)
            chunk_embeddings: Embedding vectors for chunks (optional)
            query_embedding: Embedding vector for query (optional)

        Returns:
            List of ChunkScore, sorted by final_score descending
        """
        if query_analysis is None:
            query_analysis = self.analyze_query(query)

        scores = []

        for i, chunk in enumerate(chunks):
            # 1. Fingerprint similarity (if available)
            fp_sim = 0.0
            if chunk_fingerprints and i < len(chunk_fingerprints):
                fp_sim = self._compute_fingerprint_similarity(
                    query_analysis.fingerprints,
                    chunk_fingerprints[i],
                    query_analysis.anchor_tokens,
                )

            # 2. Target quality (how good is this chunk as a retrieval target)
            target_quality = 0.5  # Default
            if chunk_fingerprints and i < len(chunk_fingerprints):
                target_quality = self._compute_target_quality(chunk_fingerprints[i])

            # 3. Semantic similarity (if embeddings provided)
            semantic_sim = 0.0
            if chunk_embeddings is not None and query_embedding is not None:
                semantic_sim = self._cosine_similarity(
                    query_embedding, chunk_embeddings[i]
                )

            # Combined score
            final_score = (
                self.fingerprint_weight * fp_sim
                + self.embedding_weight * semantic_sim
                + self.target_quality_weight * target_quality
            )

            # Explanation
            explanation = self._generate_explanation(
                fp_sim, semantic_sim, target_quality
            )

            scores.append(
                ChunkScore(
                    chunk=chunk,
                    chunk_idx=i,
                    fingerprint_similarity=fp_sim,
                    target_quality=target_quality,
                    semantic_score=semantic_sim,
                    final_score=final_score,
                    explanation=explanation,
                )
            )

        # Sort by score descending
        scores.sort(key=lambda x: x.final_score, reverse=True)
        return scores

    def retrieve_and_generate(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
        max_tokens: int = 500,
        include_fingerprints: bool = False,
    ) -> RAGResult:
        """
        Full RAG pipeline with attention-guided retrieval.

        Args:
            query: User query
            documents: Candidate documents/chunks to retrieve from
            top_k: Number of chunks to include in context
            max_tokens: Max tokens for generation
            include_fingerprints: Whether to include generation fingerprints

        Returns:
            RAGResult with answer and retrieval details
        """
        # 1. Analyze query
        query_analysis = self.analyze_query(query)

        # 2. Check if retrieval is needed
        if query_analysis.retrieval_need_score < self.min_retrieval_need:
            logger.info(
                f"Low retrieval need ({query_analysis.retrieval_need_score:.2f}), "
                "generating without retrieval"
            )
            answer = self._generate(query, context=None, max_tokens=max_tokens)
            return RAGResult(
                query=query,
                answer=answer,
                retrieved_chunks=[],
                query_analysis=query_analysis,
            )

        # 3. Score and rank chunks
        chunk_scores = self.score_chunks(query, documents, query_analysis)

        # 4. Select top chunks
        top_chunks = chunk_scores[:top_k]

        # 5. Build context
        context = "\n\n".join(
            [f"[Source {i+1}]: {cs.chunk}" for i, cs in enumerate(top_chunks)]
        )

        # 6. Generate with context
        answer, gen_fingerprints = self._generate_with_fingerprints(
            query, context, max_tokens, include_fingerprints
        )

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_chunks=top_chunks,
            query_analysis=query_analysis,
            generation_fingerprints=gen_fingerprints if include_fingerprints else None,
        )

    def should_retrieve(self, query: str) -> Tuple[bool, float, str]:
        """
        Determine if retrieval is needed for a query.

        Returns:
            (should_retrieve, confidence, reason)
        """
        analysis = self.analyze_query(query)

        should = analysis.retrieval_need_score >= self.min_retrieval_need

        if should:
            reason = (
                f"Query has {len(analysis.anchor_tokens)} retrieval anchors "
                f"with need score {analysis.retrieval_need_score:.2f}"
            )
        else:
            reason = (
                f"Query is self-contained (need score {analysis.retrieval_need_score:.2f})"
            )

        return should, analysis.retrieval_need_score, reason

    # =========================================================================
    # Private methods
    # =========================================================================

    def _get_query_fingerprints(
        self, query: str
    ) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Get fingerprints for query from server."""
        try:
            import requests

            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1,  # Just need prefill fingerprints
                    "extra_body": {"return_attention_tokens": True},
                },
                timeout=30,
            )
            data = resp.json()

            # Extract fingerprints from response
            attention_tokens = data.get("attention_tokens", [])
            fingerprints = []
            zones = []
            tokens = []

            for token_info in attention_tokens:
                if isinstance(token_info, dict):
                    fp = token_info.get("fingerprint")
                    if fp:
                        fingerprints.append(np.array(fp))
                        zones.append(token_info.get("manifold", "unknown"))
                        tokens.append(token_info.get("token", ""))

            return fingerprints, zones, tokens

        except Exception as e:
            logger.warning(f"Failed to get fingerprints from server: {e}")
            return [], [], []

    def _classify_zone(self, fp: np.ndarray) -> str:
        """Classify attention zone from fingerprint."""
        if len(fp) < 4:
            return "unknown"

        local_mass = fp[FP_LOCAL_MASS]
        mid_mass = fp[FP_MID_MASS]
        long_mass = fp[FP_LONG_MASS]
        entropy = fp[FP_ENTROPY]

        if local_mass > 0.6 and entropy < 0.3:
            return "syntax_floor"
        elif long_mass > 0.4:
            return "long_range"
        elif mid_mass > 0.4:
            return "semantic_bridge"
        elif local_mass + mid_mass + long_mass < 0.5:
            return "diffuse"
        else:
            return "structure_ripple"

    def _compute_fingerprint_similarity(
        self,
        query_fps: List[np.ndarray],
        chunk_fps: List[np.ndarray],
        anchor_tokens: List[Tuple[int, str, float]],
    ) -> float:
        """
        Compute similarity between query and chunk fingerprints.

        Weights similarity by anchor token importance.
        """
        if not query_fps or not chunk_fps:
            return 0.0

        # Get anchor indices and weights
        anchor_indices = {idx: weight for idx, _, weight in anchor_tokens}

        total_sim = 0.0
        total_weight = 0.0

        for i, q_fp in enumerate(query_fps):
            # Weight by anchor importance
            weight = anchor_indices.get(i, 0.3)

            # Find best matching chunk fingerprint
            best_sim = 0.0
            for c_fp in chunk_fps:
                sim = self._cosine_similarity(q_fp, c_fp)
                best_sim = max(best_sim, sim)

            total_sim += weight * best_sim
            total_weight += weight

        return total_sim / total_weight if total_weight > 0 else 0.0

    def _compute_target_quality(self, chunk_fps: List[np.ndarray]) -> float:
        """Compute how good a chunk is as a retrieval target."""
        if not chunk_fps:
            return 0.5

        qualities = []
        for fp in chunk_fps:
            zone = self._classify_zone(fp)
            zone_quality = ZONE_TARGET_QUALITY.get(zone, 0.5)

            # Low entropy = focused = good target
            entropy = fp[FP_ENTROPY] if len(fp) > FP_ENTROPY else 0.5
            entropy_quality = 1.0 - min(entropy, 1.0)

            # High long-range mass = good anchor
            long_mass = fp[FP_LONG_MASS] if len(fp) > FP_LONG_MASS else 0.5

            quality = 0.4 * zone_quality + 0.3 * entropy_quality + 0.3 * long_mass
            qualities.append(quality)

        return float(np.mean(qualities))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()

        # Pad to same length if needed
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _extract_phrases(
        self, anchor_tokens: List[Tuple[int, str, float]], tokens: List[str]
    ) -> List[str]:
        """Extract key phrases from anchor tokens."""
        if not tokens or not anchor_tokens:
            return []

        # Group consecutive anchors into phrases
        phrases = []
        current_phrase = []
        last_idx = -2

        for idx, token_text, score in sorted(anchor_tokens):
            if idx == last_idx + 1:
                current_phrase.append(token_text)
            else:
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                current_phrase = [token_text]
            last_idx = idx

        if current_phrase:
            phrases.append(" ".join(current_phrase))

        return phrases

    def _generate_explanation(
        self, fp_sim: float, semantic_sim: float, target_quality: float
    ) -> str:
        """Generate human-readable explanation of score."""
        parts = []

        if fp_sim > 0.7:
            parts.append("strong attention alignment")
        elif fp_sim > 0.4:
            parts.append("moderate attention alignment")

        if semantic_sim > 0.7:
            parts.append("high semantic similarity")
        elif semantic_sim > 0.4:
            parts.append("moderate semantic similarity")

        if target_quality > 0.7:
            parts.append("excellent retrieval target")
        elif target_quality > 0.5:
            parts.append("good retrieval target")

        return ", ".join(parts) if parts else "baseline score"

    def _generate(
        self, query: str, context: Optional[str], max_tokens: int
    ) -> str:
        """Generate response with optional context."""
        try:
            import requests

            if context:
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            else:
                prompt = query

            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {e}"

    def _generate_with_fingerprints(
        self, query: str, context: str, max_tokens: int, include_fingerprints: bool
    ) -> Tuple[str, Optional[List[np.ndarray]]]:
        """Generate with optional fingerprint capture."""
        try:
            import requests

            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "extra_body": {
                        "return_attention_tokens": include_fingerprints,
                    },
                },
                timeout=60,
            )
            data = resp.json()

            answer = data["choices"][0]["message"]["content"]

            fingerprints = None
            if include_fingerprints:
                attention_tokens = data.get("attention_tokens", [])
                fingerprints = []
                for token_info in attention_tokens:
                    if isinstance(token_info, dict):
                        fp = token_info.get("fingerprint")
                        if fp:
                            fingerprints.append(np.array(fp))

            return answer, fingerprints

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {e}", None


# =============================================================================
# Convenience functions
# =============================================================================


def analyze_retrieval_need(
    query: str,
    fingerprints: List[np.ndarray],
    zones: Optional[List[str]] = None,
) -> Tuple[float, List[int]]:
    """
    Quick analysis of retrieval need from fingerprints.

    Args:
        query: Query text
        fingerprints: Attention fingerprints
        zones: Optional zone classifications

    Returns:
        (retrieval_need_score, anchor_token_indices)
    """
    retriever = AttentionGuidedRetriever()
    analysis = retriever.analyze_query(query, fingerprints, zones)
    anchor_indices = [idx for idx, _, _ in analysis.anchor_tokens]
    return analysis.retrieval_need_score, anchor_indices


def score_chunk_relevance(
    query_fingerprints: List[np.ndarray],
    chunk_fingerprints: List[np.ndarray],
) -> float:
    """
    Quick scoring of chunk relevance based on fingerprint similarity.

    Args:
        query_fingerprints: Query token fingerprints
        chunk_fingerprints: Chunk token fingerprints

    Returns:
        Relevance score (0-1)
    """
    retriever = AttentionGuidedRetriever()
    return retriever._compute_fingerprint_similarity(
        query_fingerprints, chunk_fingerprints, []
    )


# =============================================================================
# Demo / Testing
# =============================================================================


def _demo():
    """Demo of attention-guided RAG."""
    print("=" * 60)
    print("Attention-Guided RAG Demo")
    print("=" * 60)

    # Create synthetic fingerprints for demo
    np.random.seed(42)

    # Query fingerprints (some are retrieval anchors)
    query_fps = [
        np.array([0.1, 0.3, 0.6, 0.8] + [0.0] * 16),  # semantic_bridge, high entropy
        np.array([0.7, 0.2, 0.1, 0.2] + [0.0] * 16),  # syntax_floor
        np.array([0.2, 0.5, 0.3, 0.7] + [0.0] * 16),  # semantic_bridge
        np.array([0.1, 0.4, 0.5, 0.6] + [0.0] * 16),  # long_range
    ]

    # Chunk fingerprints
    chunk1_fps = [  # Good target (focused, low entropy)
        np.array([0.2, 0.4, 0.4, 0.2] + [0.0] * 16),
        np.array([0.1, 0.5, 0.4, 0.3] + [0.0] * 16),
    ]
    chunk2_fps = [  # Poor target (diffuse, high entropy)
        np.array([0.3, 0.3, 0.2, 0.9] + [0.0] * 16),
        np.array([0.4, 0.3, 0.2, 0.8] + [0.0] * 16),
    ]

    retriever = AttentionGuidedRetriever()

    # Analyze query
    print("\n1. Query Analysis")
    print("-" * 40)
    analysis = retriever.analyze_query(
        query="What is the capital of France?",
        fingerprints=query_fps,
        tokens=["What", "is", "the", "capital"],
    )
    print(f"Retrieval need score: {analysis.retrieval_need_score:.2f}")
    print(f"Anchor tokens: {analysis.anchor_tokens}")

    # Score chunks
    print("\n2. Chunk Scoring")
    print("-" * 40)

    chunks = [
        "Paris is the capital of France, located on the Seine River.",
        "France is a country in Western Europe with diverse landscapes.",
    ]

    scores = retriever.score_chunks(
        query="What is the capital?",
        chunks=chunks,
        query_analysis=analysis,
        chunk_fingerprints=[chunk1_fps, chunk2_fps],
    )

    for score in scores:
        print(f"Chunk {score.chunk_idx}: {score.final_score:.3f}")
        print(f"  FP similarity: {score.fingerprint_similarity:.3f}")
        print(f"  Target quality: {score.target_quality:.3f}")
        print(f"  Explanation: {score.explanation}")
        print(f"  Text: {score.chunk[:50]}...")
        print()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    _demo()
