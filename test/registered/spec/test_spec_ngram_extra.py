import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

# Extra: Triton + Flashinfer NGRAM backends. Sibling per-commit file
# (test_spec_ngram.py) keeps the Paged variant.
register_cuda_ci(est_time=254, stage="extra-a", runner_config="1-gpu-large")


class TestNgramSpeculativeDecodingTriton(NgramServerBase, GSM8KMixin):
    attention_backend = "triton"


class TestNgramSpeculativeDecodingFlashinfer(NgramServerBase, GSM8KMixin):
    attention_backend = "flashinfer"
    extra_args = ["--speculative-ngram-external-sam-budget", "8"]

    def test_output_as_corpus_boosts_accept_length(self):
        """Baseline → HTTP add corpus → verify accept length boost."""
        prompts = [
            "The capital of France is",
            "In mathematics, the Pythagorean theorem states that",
            "The speed of light in a vacuum is approximately",
            "Water boils at a temperature of",
            "The largest planet in our solar system is",
        ]
        max_new_tokens = 128
        num_rounds = 3

        def generate_batch():
            outputs = []
            for prompt in prompts:
                resp = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                    timeout=120,
                )
                self.assertEqual(resp.status_code, 200, resp.text)
                outputs.append(resp.json()["text"])
            return outputs

        def get_accept_length():
            info = requests.get(self.base_url + "/server_info").json()
            return info["internal_states"][0]["avg_spec_accept_length"]

        # Phase 1: baseline — no SAM corpus loaded, only trie
        generated_outputs = []
        for _ in range(num_rounds):
            generated_outputs = generate_batch()
        baseline_accept_len = get_accept_length()
        print(f"\n  Baseline accept length (no SAM): {baseline_accept_len:.2f}")

        # Flush cache so phase 2 starts clean
        requests.post(self.base_url + "/flush_cache", timeout=30)

        # Phase 2: add generated outputs as corpus via HTTP API
        resp = requests.post(
            self.base_url + "/add_external_corpus",
            json={"corpus_id": "bench", "documents": generated_outputs},
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["success"], resp.json().get("message"))

        for _ in range(num_rounds):
            generate_batch()
        sam_accept_len = get_accept_length()
        print(f"  SAM accept length (output as corpus): {sam_accept_len:.2f}")
        print(f"  Speedup: {sam_accept_len / baseline_accept_len:.2f}x")

        self.assertGreater(
            sam_accept_len,
            baseline_accept_len * 2.0,
            f"SAM accept length ({sam_accept_len:.2f}) should be at least 2x "
            f"baseline ({baseline_accept_len:.2f}) when corpus matches output",
        )


if __name__ == "__main__":
    unittest.main()
