"""
gRPC Router E2E Test - Embedding Correctness

Test that embeddings from the gRPC router match HuggingFace reference embeddings.
Validates numerical correctness including tokenization (BOS/EOS handling) and inference.
"""

import logging
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Test data for semantic similarity checks
SEMANTIC_TEST_SETS: List[List[str]] = [
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
RELEVANCE_TEST_DATA: Dict = {
    "sample_query": "Why is Oracle launching Cloud Lift Services?",
    "sample_reference": [
        {
            "docid": 466,
            "body": "What are some extended benefits of using Oracle Cloud Infrastructure?  \nWhen customers migrate their on-premises Oracle applications to Oracle Cloud Infrastructure, they realize the benefits \nof the cloud without needing to rearchitect those applications. Customers can lower total cost of ownership, improve \nagility and increase workload performance. Additional benefits include:  \nConsistently low global pricing and lack of hidden charges \nAutomated migration support, leveraging cloud managers and tools for key applications \nFlexible universal credits applied towards any IaaS or PaaS service \nBring Your Own License (BYOL) capabilities \nIs Oracle Cloud Lift available for PAYGO customers?  \nOracle Cloud Lift Services are designed for customers who use the UCM credits (Monthly Flex). PAYGO customers can \ncontact their sales representative or cloud engineer to evaluate their eligibility.  \nAre any countries excluded from Oracle Cloud Lift Services? \nAmong the countries that Oracle operates in, only China is excluded from the Oracle Cloud Lift Services program. \nHow does Oracle Cloud Lift Services impact Oracle partners?  \nThe Cloud Lift Services program has been socialized with select partners to both complement and grow their existing \nbusinesses. Oracle is committed to working with different partner business models, from services partners to resellers, \nISVs, CSPs, etc. For additional details, partners should contact their Oracle representative.  \nHow will a partner benefit from Oracle Cloud Lift Services model? \nThe purpose of Oracle Cloud Lift Services is to serve our collective customer base and accelerate growth in our \necosystem. Oracle Cloud Engineering will now provide guidance on planning, architecting, prototyping, and managing \ncloud migrations. When partners are leading an opportunity, Oracle will work with and through our partners to offer \nCloud Lift Services as needed to make our joint customers more successful.  Public Sector accounts and partner \nengagements are not currently eligible to participate in this program.",
        },
        {
            "docid": 636,
            "body": "Cloud Lift Services as needed to make our joint customers more successful.  Public Sector accounts and partner \nengagements are not currently eligible to participate in this program. \n          How can I get started with Oracle Cloud?  \nYou can use the Oracle Cloud Free Tier for a free trial and Contact Us for more information.  \n \n  \n blogs.oracle.com             \n facebook.com/OracleCloud/             \n twitter.com/OracleCloud/            \n linkedin.com/showcase/oracle-cloud/\u2028 \n \n \n2 \nFrequently Asked Questions  / Oracle Cloud Lift Services /  Version 1.2 \n \n \nCopyright \u00a9 2021, Oracle and/or its affiliates  /  Public",
        },
        {
            "docid": 545,
            "body": "Frequently Asked Questions (FAQs) for  \nOracle Cloud Lift Services \n \nWhy is Oracle launching Cloud Lift Services? \n \n \n  \nThis program underscores Oracle\u2019s intent to better serve its customer base. Cloud Lift Services provide new and \nexisting customers expanded access to cloud engineering tools and resources to quickly migrate workloads at no \nadditional cost. \nHow are Oracle Cloud Lift Services different from pre-sales activities such as Proof-of-Concepts (POCs)? \nWhile POCs and other presales help are available from Oracle, Cloud Lift Services are post-sales and part of the \nenterprise contract.  Migration and go-live support for eligible workloads mean that our experts can engage during and \nafter the sales process to help bring workloads into production faster. \nWhat\u2019s included \u2013 and excluded \u2013 from Oracle Cloud Lift Services? \nThe Oracle Cloud Lift Services web page provides details on included vs. excluded services.  We encourage you to work \nwith your Oracle sales representative to talk through the details of your plans.  In general terms, migrating up to ten \nOracle Databases, Oracle applications, Cloud Native or HPC applications can be included, while more than 10 \nmigrations, complex migrations involving new business logic, platform upgrades, or custom development are not. \nHow do new customers get access to Oracle Cloud Lift Services? \nNew customers work with cloud engineering during the contract process to create and agree on a documented work \nplan which lays out the specific eligible workloads, timelines, and other details.   \nHow do existing customers get access to Oracle Cloud Lift Services?  \nExisting customers work with cloud engineering or sales to have an addendum for Oracle Cloud Lift Services included \nas part of their existing contract. \nWhat happens if I already have a paid services engagement?",
        },
        {
            "docid": 716,
            "body": "as part of their existing contract. \nWhat happens if I already have a paid services engagement? \nPlease keep proceeding with your existing engagement. Oracle will work with you to identify expansion opportunities \nto leverage Cloud Lift Services for other projects. \nHow do I decide whether to use Oracle Cloud Lift Services?  \nOracle Cloud Lift Services can meet customer needs when cloud engineers can help migrate a few applications to OCI; \nmigrate applications without updating software versions; configure OCI tenancies, compartments, quotas and \nidentities; perform basic reviews of network configurations and security, FastConnect setup, auditing, and assessing \nregulatory compliance; and train in-house resources on OCI.  \nHow does Oracle offer support post go-live solutions as a part of Oracle Lift Services? \nFor cloud-based continuous optimization services,  customers can use Oracle\u2019s partners, Oracle Consulting, and Oracle \nAdvanced Customer Services (ACS),  which provides joint and fully managed 24/7 lifecycle services for database, \napplications and security. \n \n \n \n1 \nFrequently Asked Questions  / Oracle Cloud Lift Services /  Version 1.2 \n \n \nCopyright \u00a9 2021, Oracle and/or its affiliates  /  Public \n \nAre Oracle Cloud Lift Services being offered to public sector? \nAligning to the rules and regulations that govern our public sector customers, Oracle will make Cloud Lift Services \navailable to all North American public sector customers in the near-term. Oracle is committed to making these services \navailable to public sector customers globally.  \nWhat are some extended benefits of using Oracle Cloud Infrastructure?  \nWhen customers migrate their on-premises Oracle applications to Oracle Cloud Infrastructure, they realize the benefits",
        },
    ],
}


def get_openai_embeddings(
    texts: Union[str, List[str]], config: Dict
) -> List[List[float]]:
    """Get embeddings from the gateway via OpenAI-compatible API."""
    import openai

    client = openai.Client(api_key=config["api_key"], base_url=config["base_url"])

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=config["model_name"],
            input=text,
        )
        embeddings.append(response.data[0].embedding)

    return embeddings


def get_hf_st_embeddings(texts: Union[str, List[str]], model_path: str) -> np.ndarray:
    """Get embeddings using sentence-transformers library.

    This handles the correct pooling strategy for each model automatically.
    For e5-mistral, it uses last-token pooling (not mean pooling).

    Uses CPU to compute reference embeddings to avoid GPU memory conflicts
    with the worker being tested. This is acceptable since reference embeddings
    only need to be accurate, not fast.
    """
    from sentence_transformers import SentenceTransformer

    if isinstance(texts, str):
        texts = [texts]

    # Force CPU to avoid GPU memory conflicts in CI where GPUs may be
    # used by other workers. Reference embeddings just need accuracy, not speed.
    model = SentenceTransformer(model_path, trust_remote_code=True, device="cpu")
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


def get_input_texts(test_json: Dict) -> List[str]:
    """Extract document bodies from test JSON."""
    return [doc["body"] for doc in test_json["sample_reference"]]


def compare_embeddings(
    embeddings1: List[List[float]], embeddings2: List[List[float]]
) -> List[float]:
    """Compare two sets of embeddings using cosine similarity."""
    logging.info("Comparing embeddings")
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0).item()
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


class TestEmbeddingCorrectness(CustomTestCase):
    """Test embedding correctness by comparing gateway output against HuggingFace reference.

    Strategy: Pre-compute HuggingFace reference embeddings on CPU, then launch the
    worker on GPU and compare. Using CPU for reference avoids GPU memory conflicts
    in CI where multiple workers may share limited GPU resources.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_EMBEDDING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Pre-compute all reference embeddings on CPU before launching the worker
        # This avoids GPU memory conflicts in CI environments with limited GPUs
        logging.info(
            f"Pre-computing HuggingFace reference embeddings (CPU) for {cls.model}"
        )

        # Flatten all test texts for semantic similarity
        all_semantic_texts = []
        for text_set in SEMANTIC_TEST_SETS:
            all_semantic_texts.extend(text_set)

        # Get relevance test texts
        query = f"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {RELEVANCE_TEST_DATA['sample_query']}"
        docs = get_input_texts(RELEVANCE_TEST_DATA)

        # Compute all reference embeddings at once
        cls.hf_semantic_embeddings = get_hf_st_embeddings(all_semantic_texts, cls.model)
        cls.hf_query_embedding = get_hf_st_embeddings(query, cls.model)
        cls.hf_docs_embeddings = get_hf_st_embeddings(docs, cls.model)

        logging.info("Reference embeddings computed on CPU")

        # Now launch workers with --is-embedding flag
        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            num_workers=1,
            tp_size=1,
            policy="round_robin",
            api_key=cls.api_key,
            worker_args=["--is-embedding"],
        )

        cls.config = {
            "server_engine": "sgl-model-gateway",
            "base_url": cls.base_url + "/v1",
            "model_name": cls.model,
            "model_path": cls.model,
            "api_key": cls.api_key,
        }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_semantic_similarity(self, tolerance: float = 1e-2):
        """Check if gateway and HF embeddings give similar results."""
        # Track position in pre-computed embeddings
        embed_idx = 0

        for i, input_texts in enumerate(SEMANTIC_TEST_SETS):
            logging.info(f"Processing semantic similarity test set {i + 1}")

            embedding_gateway = get_openai_embeddings(input_texts, self.config)

            # Get pre-computed HF embeddings for this set
            num_texts = len(input_texts)
            embedding_hf = self.hf_semantic_embeddings[
                embed_idx : embed_idx + num_texts
            ].tolist()
            embed_idx += num_texts

            logging.info(f'Comparing {self.config["server_engine"]} and HF embeddings')
            similarities = compare_embeddings(embedding_gateway, embedding_hf)

            logging.info(f"Similarities between embeddings: {similarities}")

            # Verify similarities
            for sim in similarities:
                self.assertLess(
                    abs(sim - 1.0), tolerance, f"Similarity {sim} is not close to 1"
                )

            logging.info(f"Semantic similarity test set {i + 1} passed\n")

    def test_relevance_scores(self, tolerance: float = 0.05):
        """Compare relevance scores between gateway and HF implementations."""
        logging.info(
            f'Comparing relevance scores between {self.config["server_engine"]} and HF'
        )

        # Format query with instruction (for e5-mistral)
        query = f"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {RELEVANCE_TEST_DATA['sample_query']}"
        docs = get_input_texts(RELEVANCE_TEST_DATA)

        # Get gateway scores
        query_embeddings_gateway = get_openai_embeddings(query, self.config)
        docs_embeddings_gateway = get_openai_embeddings(docs, self.config)
        scores_gateway = (
            np.array(query_embeddings_gateway) @ np.array(docs_embeddings_gateway).T
        ) * 100

        # Use pre-computed HF scores
        scores_hf = (self.hf_query_embedding @ self.hf_docs_embeddings.T) * 100

        logging.info(
            f'{self.config["server_engine"]} relevance scores: {scores_gateway}'
        )
        logging.info(f"HF relevance scores: {scores_hf}")

        self.assertTrue(
            np.allclose(scores_gateway, scores_hf, atol=tolerance),
            f'Scores differ beyond tolerance: \n{self.config["server_engine"]}: {scores_gateway}\nHF: {scores_hf}',
        )

        logging.info("Relevance scores comparison completed successfully")


if __name__ == "__main__":
    unittest.main()
