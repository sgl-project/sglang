"""Exercise all eight NGRAM corpus options through real NPU generation."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

PROMPTS = [
    "Continue the sequence: 1, 2, 3, 4, 5,",
    "Repeat the word hello ten times:",
    "Write the first ten positive integers separated by commas:",
]


class TestNpuNgram(unittest.TestCase):
    prompts = PROMPTS

    def generate(self, **options):
        import sglang as sgl

        engine_options = dict(
            model_path=os.environ.get(
                "SGLANG_TEST_NGRAM_MODEL", QWEN3_0_6B_WEIGHTS_PATH
            ),
            device="npu",
            trust_remote_code=True,
            attention_backend="ascend",
            dtype="bfloat16",
            mem_fraction_static=0.75,
            max_running_requests=4,
            max_total_tokens=2048,
            context_length=1024,
            chunked_prefill_size=512,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            log_level="warning",
        )
        engine_options.update(options)
        engine = sgl.Engine(**engine_options)
        try:
            if options.get("disable_cuda_graph") is False:
                info = engine.get_server_info()
                self.assertFalse(info["disable_cuda_graph"])
                self.assertGreater(
                    info["startup_time"]["cuda_graph"]["target_verify"], 0
                )
            # The second batch reuses the CPU trie learned from the first one.
            outputs = [
                engine.generate(
                    self.prompts,
                    sampling_params={"temperature": 0, "max_new_tokens": 48},
                    return_logprob=True,
                )
                for _ in range(2)
            ]
            if options:
                sampled = engine.generate(
                    self.prompts[1],
                    sampling_params={
                        "temperature": 0.7,
                        "top_k": 10,
                        "top_p": 0.9,
                        "max_new_tokens": 24,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                )
                self.assertEqual(len(sampled["output_ids"]), 24)
                self.assertEqual(len(sampled["meta_info"]["output_token_logprobs"]), 24)
            return outputs
        finally:
            engine.shutdown()

    def test_graph_matches_eager_with_long_prefix(self):
        self.prompts = PROMPTS + [
            "Context only: "
            + "The integer list is 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. " * 6
            + "\nContinue the sequence: 1, 2, 3, 4, 5,"
        ]
        options = dict(
            speculative_algorithm="NGRAM",
            speculative_num_draft_tokens=12,
            speculative_ngram_capacity=10000,
            cuda_graph_bs_decode=[1, 2, 4],
            cuda_graph_max_bs_decode=4,
        )
        expected = self.generate(**options, disable_cuda_graph=True)
        actual = self.generate(**options, disable_cuda_graph=False)
        for expected_batch, actual_batch in zip(expected, actual):
            for reference, output in zip(expected_batch, actual_batch):
                self.assertEqual(output["output_ids"], reference["output_ids"])

    def test_eight_options_match_greedy_baseline(self):
        from transformers import AutoTokenizer

        baseline = self.generate()
        model = os.environ.get("SGLANG_TEST_NGRAM_MODEL", QWEN3_0_6B_WEIGHTS_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model)
        documents = [
            prompt + output["text"] for prompt, output in zip(self.prompts, baseline[0])
        ]
        token_count = (
            sum(
                len(tokenizer.encode(doc, add_special_tokens=False))
                for doc in documents
            )
            + len(documents)
            - 1
        )
        evidence = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(
                "".join(json.dumps(doc) + "\n" for doc in documents), encoding="utf-8"
            )
            external = dict(
                speculative_ngram_external_corpus_path=str(path),
                speculative_ngram_external_sam_budget=4,
            )
            cases = {
                "min_bfs_breadth": dict(speculative_ngram_min_bfs_breadth=2),
                "max_bfs_breadth": dict(speculative_ngram_max_bfs_breadth=4),
                "match_type": dict(speculative_ngram_match_type="PROB"),
                "max_trie_depth": dict(speculative_ngram_max_trie_depth=8),
                "capacity": dict(speculative_ngram_capacity=64),
                "external_corpus_path": external,
                "external_sam_budget": dict(
                    external, speculative_ngram_external_sam_budget=11
                ),
                "external_corpus_max_tokens": dict(
                    external, speculative_ngram_external_corpus_max_tokens=token_count
                ),
            }
            for name, options in cases.items():
                with self.subTest(option=name):
                    actual = self.generate(
                        **dict(
                            {
                                "speculative_algorithm": "NGRAM",
                                "speculative_ngram_capacity": 10000,
                            },
                            **options,
                        )
                    )
                    accepts = []
                    max_logprob_diff = 0.0
                    for expected_batch, actual_batch in zip(baseline, actual):
                        for expected, output in zip(expected_batch, actual_batch):
                            self.assertEqual(
                                output["output_ids"], expected["output_ids"]
                            )
                            logprobs = output["meta_info"]["output_token_logprobs"]
                            self.assertEqual(len(logprobs), 48)
                            self.assertEqual(
                                [row[1] for row in logprobs], output["output_ids"]
                            )
                            reference = expected["meta_info"]["output_token_logprobs"]
                            max_logprob_diff = max(
                                max_logprob_diff,
                                max(
                                    abs(a[0] - b[0])
                                    for a, b in zip(logprobs, reference)
                                ),
                            )
                            accepts.append(output["meta_info"]["spec_accept_length"])
                    self.assertGreater(max(accepts), 1)
                    evidence[name] = {
                        "prompts": self.prompts,
                        "options": options,
                        "greedy_output_ids_equal": True,
                        "checked_tokens": 288,
                        "accept_lengths": accepts,
                        "max_logprob_difference": max_logprob_diff,
                        "stochastic_generation": True,
                    }
                    print(json.dumps({name: evidence[name]}), flush=True)
                    if result_path := os.environ.get("SGLANG_TEST_NGRAM_RESULT"):
                        Path(result_path).write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    unittest.main()
