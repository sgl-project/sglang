"""Embedding correctness tests.

Tests that embeddings from the router match HuggingFace reference embeddings.
Validates numerical correctness including tokenization and inference.

Source: Migrated from e2e_grpc/basic/test_embedding_correctness.py

Usage:
    pytest e2e_test/embeddings/test_correctness.py -v
    pytest e2e_test/embeddings/test_correctness.py -v -k "grpc"

Requirements:
    - sentence-transformers (for reference embeddings)
    - torch
    - numpy
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Thread-safe storage for HF reference embeddings
_hf_embeddings_cache: dict[str, Any] | None = None
_hf_embeddings_lock = threading.Lock()


# Test data for semantic similarity checks
SEMANTIC_TEST_SETS: list[list[str]] = [
    [
        "The cat sat on the mat.",
        "A feline was resting on a rug.",
        "Bright stars illuminate the night sky.",  # Unrelated sentence
    ],
    [
        "The quick brown fox jumps over the lazy dog.",
        "A fast, dark-colored fox leaps above a sluggish canine.",
        "Ocean waves gently lap against the shore.",  # Unrelated sentence
    ],
    [
        "An apple a day keeps the doctor away.",
        "Eating a daily apple can prevent medical visits.",
        "Mountains are vast and often snow-capped.",  # Unrelated sentence
    ],
]

# Test data for relevance scoring
RELEVANCE_TEST_DATA: dict[str, Any] = {
    "sample_query": "Why is Oracle launching Cloud Lift Services?",
    "sample_reference": [
        {
            "docid": 466,
            "body": "What are some extended benefits of using Oracle Cloud Infrastructure?  \nWhen customers migrate their on-premises Oracle applications to Oracle Cloud Infrastructure, they realize the benefits \nof the cloud without needing to rearchitect those applications. Customers can lower total cost of ownership, improve \nagility and increase workload performance. Additional benefits include:  \nConsistently low global pricing and lack of hidden charges \nAutomated migration support, leveraging cloud managers and tools for key applications \nFlexible universal credits applied towards any IaaS or PaaS service \nBring Your Own License (BYOL) capabilities \nIs Oracle Cloud Lift available for PAYGO customers?  \nOracle Cloud Lift Services are designed for customers who use the UCM credits (Monthly Flex). PAYGO customers can \ncontact their sales representative or cloud engineer to evaluate their eligibility.  \nAre any countries excluded from Oracle Cloud Lift Services? \nAmong the countries that Oracle operates in, only China is excluded from the Oracle Cloud Lift Services program.",
        },
        {
            "docid": 636,
            "body": "Cloud Lift Services as needed to make our joint customers more successful.  Public Sector accounts and partner \nengagements are not currently eligible to participate in this program. \n          How can I get started with Oracle Cloud?  \nYou can use the Oracle Cloud Free Tier for a free trial and Contact Us for more information.",
        },
        {
            "docid": 545,
            "body": "Frequently Asked Questions (FAQs) for  \nOracle Cloud Lift Services \n \nWhy is Oracle launching Cloud Lift Services? \n \n \n  \nThis program underscores Oracle's intent to better serve its customer base. Cloud Lift Services provide new and \nexisting customers expanded access to cloud engineering tools and resources to quickly migrate workloads at no \nadditional cost.",
        },
        {
            "docid": 716,
            "body": "as part of their existing contract. \nWhat happens if I already have a paid services engagement? \nPlease keep proceeding with your existing engagement. Oracle will work with you to identify expansion opportunities \nto leverage Cloud Lift Services for other projects.",
        },
    ],
}


def get_openai_embeddings(
    texts: str | list[str],
    client,
    model: str,
) -> list[list[float]]:
    """Get embeddings from the gateway via OpenAI-compatible API."""
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        embeddings.append(response.data[0].embedding)

    return embeddings


def get_hf_st_embeddings(texts: str | list[str], model_path: str) -> np.ndarray:
    """Get embeddings using sentence-transformers library.

    This handles the correct pooling strategy for each model automatically.
    For e5-mistral, it uses last-token pooling (not mean pooling).

    Uses CPU to compute reference embeddings to avoid GPU memory conflicts
    with the worker being tested.
    """
    from sentence_transformers import SentenceTransformer

    if isinstance(texts, str):
        texts = [texts]

    # Force CPU to avoid GPU memory conflicts in CI
    model = SentenceTransformer(model_path, trust_remote_code=True, device="cpu")
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


def compare_embeddings(
    embeddings1: list[list[float]], embeddings2: list[list[float]]
) -> list[float]:
    """Compare two sets of embeddings using cosine similarity."""
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0).item()
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


def get_input_texts(test_json: dict) -> list[str]:
    """Extract document bodies from test JSON."""
    return [doc["body"] for doc in test_json["sample_reference"]]


@pytest.fixture(scope="session")
def hf_reference_embeddings(request):
    """Pre-compute HuggingFace reference embeddings on CPU.

    This is done once per session with thread-safe initialization to support
    pytest-parallel execution. Uses CPU to avoid GPU memory conflicts.
    """
    global _hf_embeddings_cache

    # Thread-safe initialization - only one thread computes embeddings
    with _hf_embeddings_lock:
        if _hf_embeddings_cache is not None:
            return _hf_embeddings_cache

        from infra.model_specs import MODEL_SPECS

        # Get model path from MODEL_SPECS for the embedding model
        model_path = MODEL_SPECS.get("embedding", {}).get("model")
        if model_path is None:
            pytest.skip("Embedding model not found in MODEL_SPECS")

        logger.info(
            "Pre-computing HuggingFace reference embeddings (CPU) for %s", model_path
        )

        # Flatten all test texts for semantic similarity
        all_semantic_texts = []
        for text_set in SEMANTIC_TEST_SETS:
            all_semantic_texts.extend(text_set)

        # Get relevance test texts
        query = f"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {RELEVANCE_TEST_DATA['sample_query']}"
        docs = get_input_texts(RELEVANCE_TEST_DATA)

        # Compute all reference embeddings at once
        hf_semantic = get_hf_st_embeddings(all_semantic_texts, model_path)
        hf_query = get_hf_st_embeddings(query, model_path)
        hf_docs = get_hf_st_embeddings(docs, model_path)

        logger.info("Reference embeddings computed on CPU")

        _hf_embeddings_cache = {
            "semantic": hf_semantic,
            "query": hf_query,
            "docs": hf_docs,
        }

        return _hf_embeddings_cache


@pytest.mark.e2e
@pytest.mark.model("embedding")
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestEmbeddingCorrectness:
    """Test embedding correctness by comparing gateway output against HuggingFace reference.

    Strategy: Pre-compute HuggingFace reference embeddings on CPU, then launch the
    worker on GPU and compare. Using CPU for reference avoids GPU memory conflicts.
    """

    def test_semantic_similarity(self, setup_backend, hf_reference_embeddings):
        """Check if gateway and HF embeddings give similar results.

        For each text in the semantic test sets, the gateway embedding should
        have >0.98 cosine similarity with the HuggingFace reference embedding.
        """
        backend, model_path, client, gateway = setup_backend
        tolerance = 1e-2

        # Track position in pre-computed embeddings
        embed_idx = 0

        for i, input_texts in enumerate(SEMANTIC_TEST_SETS):
            logger.info("Processing semantic similarity test set %d", i + 1)

            embedding_gateway = get_openai_embeddings(input_texts, client, model_path)

            # Get pre-computed HF embeddings for this set
            num_texts = len(input_texts)
            embedding_hf = hf_reference_embeddings["semantic"][
                embed_idx : embed_idx + num_texts
            ].tolist()
            embed_idx += num_texts

            similarities = compare_embeddings(embedding_gateway, embedding_hf)
            logger.info("Similarities: %s", similarities)

            # Verify all similarities are close to 1.0
            for j, sim in enumerate(similarities):
                assert (
                    abs(sim - 1.0) < tolerance
                ), f"Set {i+1}, text {j+1}: similarity {sim:.4f} not close to 1.0"

            logger.info("Semantic similarity test set %d passed", i + 1)

    def test_relevance_scores(self, setup_backend, hf_reference_embeddings):
        """Compare relevance scores between gateway and HF implementations.

        The relevance scores (query @ docs) should match between the gateway
        and HuggingFace implementations within tolerance.
        """
        backend, model_path, client, gateway = setup_backend
        tolerance = 0.05

        # Format query with instruction (for e5-mistral)
        query = f"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {RELEVANCE_TEST_DATA['sample_query']}"
        docs = get_input_texts(RELEVANCE_TEST_DATA)

        # Get gateway scores
        query_embeddings_gateway = get_openai_embeddings(query, client, model_path)
        docs_embeddings_gateway = get_openai_embeddings(docs, client, model_path)
        scores_gateway = (
            np.array(query_embeddings_gateway) @ np.array(docs_embeddings_gateway).T
        ) * 100

        # Use pre-computed HF scores
        scores_hf = (
            hf_reference_embeddings["query"] @ hf_reference_embeddings["docs"].T
        ) * 100

        logger.info("Gateway relevance scores: %s", scores_gateway)
        logger.info("HF relevance scores: %s", scores_hf)

        assert np.allclose(
            scores_gateway, scores_hf, atol=tolerance
        ), f"Scores differ beyond tolerance:\nGateway: {scores_gateway}\nHF: {scores_hf}"

        logger.info("Relevance scores comparison passed")
