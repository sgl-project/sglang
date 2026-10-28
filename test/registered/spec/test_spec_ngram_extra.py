import unittest

import requests

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

# Extra: Triton + Flashinfer NGRAM backends + non-overlap (sync V2) variant.
# Sibling per-commit file (test_spec_ngram.py) keeps the Paged variant.
register_cuda_ci(est_time=216, stage="extra-a", runner_config="1-gpu-large")
# AMD: flashinfer is not built in the ROCm sgl_kernel, so only the
# triton-backend class runs on ROCm.
register_amd_ci(est_time=130, suite="extra-a-test-1-gpu-large-amd")

_AMD_SKIP_BACKEND = (
    "flashinfer attention backend is CUDA-only (not in the ROCm sgl_kernel build)"
)


class TestNgramSpeculativeDecodingTriton(NgramServerBase, GSM8KMixin):
    attention_backend = "triton"


@unittest.skipIf(is_hip(), _AMD_SKIP_BACKEND)
class TestNgramSpeculativeDecodingNoOverlap(NgramServerBase, GSM8KMixin):
    """Non-overlap path: the scheduler drives the V2 worker synchronously
    (seq_lens advance via GenerationBatchResult.new_seq_lens, accepted tokens
    come from req.output_ids instead of the spec_info splice)."""

    attention_backend = "flashinfer"
    extra_args = ["--disable-overlap-schedule"]


@unittest.skipIf(is_hip(), _AMD_SKIP_BACKEND)
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

    def test_request_corpus_ids_selects_corpus(self):
        """Per-request ngram_corpus_ids: drafting from the matching corpus
        beats an unrelated one loaded on the same server."""
        prompts = [
            "The capital of France is",
            "In mathematics, the Pythagorean theorem states that",
            "Water boils at a temperature of",
        ]

        def generate(prompt, corpus_ids):
            sampling_params = {"temperature": 0, "max_new_tokens": 128}
            if corpus_ids is not None:
                sampling_params["ngram_corpus_ids"] = corpus_ids
            resp = requests.post(
                self.base_url + "/generate",
                json={"text": prompt, "sampling_params": sampling_params},
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            out = resp.json()
            return out["text"], out["meta_info"]["spec_accept_length"]

        def run(corpus_ids):
            # Flush so the online trie does not carry over between phases.
            requests.post(self.base_url + "/flush_cache", timeout=30)
            results = [generate(p, corpus_ids) for p in prompts]
            texts = [t for t, _ in results]
            return texts, sum(a for _, a in results) / len(results)

        baseline_texts, _ = run([])
        for corpus_id, documents in (
            ("matching", baseline_texts),
            ("unrelated", ["Lorem ipsum dolor sit amet, consectetur adipiscing elit."]),
        ):
            resp = requests.post(
                self.base_url + "/add_external_corpus",
                json={"corpus_id": corpus_id, "documents": documents},
                timeout=120,
            )
            self.assertTrue(resp.json()["success"], resp.text)
        try:
            _, unrelated_len = run(["unrelated"])
            _, matching_len = run(["matching"])
            print(
                f"\n  accept length unrelated={unrelated_len:.2f} matching={matching_len:.2f}"
            )
            self.assertGreater(matching_len, unrelated_len * 1.5)
        finally:
            for corpus_id in ("matching", "unrelated"):
                requests.post(
                    self.base_url + "/remove_external_corpus",
                    json={"corpus_id": corpus_id},
                    timeout=30,
                )


# Copy-heavy prompts for the prompt SAM: the answer reuses most of the prompt.
_PROMPT_SAM_CODE = [
    """def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged""",
    """def word_count(path):
    counts = {}
    with open(path) as f:
        for line in f:
            for word in line.split():
                word = word.lower().strip(".,!?")
                counts[word] = counts.get(word, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])""",
]


@unittest.skipIf(is_hip(), _AMD_SKIP_BACKEND)
class TestNgramPromptSam(NgramServerBase):
    attention_backend = "flashinfer"
    extra_args = ["--speculative-ngram-prompt-sam-budget", "8"]

    def test_copy_task_drafts_from_prompt(self):
        """A request whose answer copies its prompt drafts from the prompt."""
        accept_lens = []
        for code in _PROMPT_SAM_CODE:
            requests.post(self.base_url + "/flush_cache", timeout=30)
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Add type hints to the following Python code. Keep "
                    "everything else exactly the same and output only the code."
                    f"\n\n{code}\n\nCode with type hints:\n",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 256},
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            accept_lens.append(resp.json()["meta_info"]["spec_accept_length"])
        avg = sum(accept_lens) / len(accept_lens)
        print(f"\n  prompt SAM accept length: {avg:.2f}")
        self.assertGreater(avg, 2.5)


if __name__ == "__main__":
    unittest.main()
