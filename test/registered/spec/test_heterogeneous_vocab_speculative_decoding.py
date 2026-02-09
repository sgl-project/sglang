"""
Integration tests for Heterogeneous Vocabulary Speculative Decoding.

This tests the TLI (Token-Level Intersection) algorithm from the paper:
"Lossless Speculative Decoding for Heterogeneous Vocabularies" (ICML 2025 Oral)
https://arxiv.org/abs/2502.05202

The key feature is enabling speculative decoding with draft and target models
that have different vocabularies.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Heterogeneous vocab: target and draft use different tokenizers
TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_HETERO = "HuggingFaceTB/SmolLM-135M"
# Same-vocab baseline: both use the Qwen tokenizer
DRAFT_MODEL_SAME_VOCAB = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Write a short poem about the ocean:",
    "Explain briefly what machine learning is:",
    "List three benefits of open source software:",
    "What is the difference between a compiler and an interpreter?",
]


class TestHeterogeneousVocabSpeculativeDecoding(CustomTestCase):
    """Test speculative decoding with heterogeneous vocabularies."""

    model = TARGET_MODEL
    draft_model = DRAFT_MODEL_HETERO
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--enable-heterogeneous-vocab",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-fraction-static",
                "0.3",
                "--disable-cuda-graph",
                "--max-running-requests",
                "4",
            ],
            env_override={
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up server process."""
        kill_process_tree(cls.process.pid)

    def test_server_health(self):
        """Test that server is running and healthy."""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)

    def test_basic_completion(self):
        """Test basic text completion with heterogeneous vocab."""
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        text = result["choices"][0]["text"]

        # Verify output is readable text (not garbled)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

        # Check that output contains mostly printable characters
        printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
        self.assertGreater(
            printable_ratio, 0.9, f"Output seems garbled: {text[:100]}..."
        )

    def test_speculative_acceptance(self):
        """Test that speculative decoding is working (some tokens accepted)."""
        # First, flush cache for clean state
        requests.get(f"{self.base_url}/flush_cache")

        # Generate some text
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Write a short poem about the ocean:",
                "max_tokens": 100,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)

        # Check server info for speculative decoding stats
        server_info = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(server_info.status_code, 200)

        info = server_info.json()
        if "internal_states" in info and len(info["internal_states"]) > 0:
            internal_state = info["internal_states"][0]
            if "avg_spec_accept_length" in internal_state:
                avg_accept_length = internal_state["avg_spec_accept_length"]
                print(f"Average speculative accept length: {avg_accept_length}")
                # With heterogeneous vocab, accept rate may be lower but should be > 1
                self.assertGreater(
                    avg_accept_length,
                    1.0,
                    "Speculative decoding should accept at least some tokens",
                )

    def test_output_consistency(self):
        """Test that output is consistent with temperature=0."""
        prompt = "The capital of France is"

        # Generate twice with same settings
        responses = []
        for _ in range(2):
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": 20,
                    "temperature": 0,
                },
            )
            self.assertEqual(response.status_code, 200)
            responses.append(response.json()["choices"][0]["text"])

        # With temperature=0, outputs should be identical
        self.assertEqual(
            responses[0],
            responses[1],
            "Deterministic outputs should match with temperature=0",
        )

    def test_chat_completion(self):
        """Test chat completion endpoint."""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                "max_tokens": 50,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Should mention "4" somewhere in the response
        self.assertIn("4", content, f"Expected '4' in response: {content}")


def _get_spec_metrics(base_url):
    """Fetch accept_length and accept_rate from server info."""
    info = requests.get(f"{base_url}/get_server_info").json()
    state = info.get("internal_states", [{}])[0]
    return {
        "avg_accept_length": state.get("avg_spec_accept_length", 0),
    }


def _generate(base_url, prompt, max_tokens=100):
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    tokens = data["usage"]["completion_tokens"]
    return text, tokens


SPEC_ARGS_COMMON = [
    "--speculative-algorithm", "STANDALONE",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--mem-fraction-static", "0.15",
    "--disable-cuda-graph",
    "--max-running-requests", "4",
]


class TestHeterogeneousVsSameVocab(CustomTestCase):
    """Compare heterogeneous-vocab spec decoding against same-vocab spec decoding.

    Three servers:
      - port 30000: hetero vocab  (Qwen-1.5B target + SmolLM-135M draft)
      - port 30001: same vocab    (Qwen-1.5B target + Qwen-0.5B draft)
      - port 30002: no spec       (Qwen-1.5B only, baseline)
    """

    url_hetero = DEFAULT_URL_FOR_TEST          # :30000
    url_same = "http://127.0.0.1:30001"
    url_baseline = "http://127.0.0.1:30002"

    @classmethod
    def setUpClass(cls):
        # 1) heterogeneous vocab
        cls.proc_hetero = popen_launch_server(
            TARGET_MODEL,
            cls.url_hetero,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *SPEC_ARGS_COMMON,
                "--speculative-draft-model-path", DRAFT_MODEL_HETERO,
                "--enable-heterogeneous-vocab",
            ],
            env_override={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
        )

        # 2) same-vocab
        cls.proc_same = popen_launch_server(
            TARGET_MODEL,
            cls.url_same,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *SPEC_ARGS_COMMON,
                "--speculative-draft-model-path", DRAFT_MODEL_SAME_VOCAB,
            ],
        )

        # 3) no spec baseline
        cls.proc_baseline = popen_launch_server(
            TARGET_MODEL,
            cls.url_baseline,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static", "0.15",
                "--disable-cuda-graph",
                "--max-running-requests", "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.proc_hetero.pid)
        kill_process_tree(cls.proc_same.pid)
        kill_process_tree(cls.proc_baseline.pid)

    def test_compare_accept_and_tokens(self):
        """Run the same prompts on all three servers, compare metrics."""
        for url in (self.url_hetero, self.url_same, self.url_baseline):
            requests.get(f"{url}/flush_cache")

        results = {"hetero": [], "same": [], "baseline": []}
        for prompt in PROMPTS:
            text_h, tok_h = _generate(self.url_hetero, prompt)
            text_s, tok_s = _generate(self.url_same, prompt)
            text_b, tok_b = _generate(self.url_baseline, prompt)
            results["hetero"].append(tok_h)
            results["same"].append(tok_s)
            results["baseline"].append(tok_b)

        metrics_hetero = _get_spec_metrics(self.url_hetero)
        metrics_same = _get_spec_metrics(self.url_same)

        total_h = sum(results["hetero"])
        total_s = sum(results["same"])
        total_b = sum(results["baseline"])

        print("\n===== Heterogeneous vs Same-Vocab Speculative Decoding =====")
        print(f"Prompts: {len(PROMPTS)}")
        print(f"")
        print(f"  {'Config':<25} {'Tokens generated':>18} {'Avg accept len':>16}")
        print(f"  {'-'*25} {'-'*18} {'-'*16}")
        print(f"  {'hetero (SmolLM draft)':<25} {total_h:>18} {metrics_hetero['avg_accept_length']:>16.2f}")
        print(f"  {'same-vocab (Qwen draft)':<25} {total_s:>18} {metrics_same['avg_accept_length']:>16.2f}")
        print(f"  {'no spec (baseline)':<25} {total_b:>18} {'n/a':>16}")
        print(f"=========================================================\n")

        # Both spec configs should actually produce tokens
        self.assertGreater(total_h, 0)
        self.assertGreater(total_s, 0)
        # Both should have accept_length > 1 (meaning spec decoding is doing something)
        self.assertGreater(metrics_hetero["avg_accept_length"], 1.0)
        self.assertGreater(metrics_same["avg_accept_length"], 1.0)

    def test_hetero_output_matches_baseline(self):
        """Hetero-vocab spec decoding output should match non-spec baseline (lossless)."""
        prompt = "The capital of France is"
        text_h, _ = _generate(self.url_hetero, prompt, max_tokens=30)
        text_b, _ = _generate(self.url_baseline, prompt, max_tokens=30)
        self.assertEqual(
            text_h, text_b,
            f"Hetero spec output should match baseline.\n"
            f"Hetero:   {text_h}\n"
            f"Baseline: {text_b}",
        )


if __name__ == "__main__":
    unittest.main()
