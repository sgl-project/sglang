"""
Integration test for Fast-dLLM v2 (PR #17577) running under sglang.

Both test cases apply the tokenizer's Qwen2 chat template to every prompt
before sending it to the sglang server, so the model sees the same
chat-formatted input a production chat client would send. This mirrors the
HuggingFace reference evaluation in ``official/eval/eval_gsm8k.py`` and
makes the two results directly comparable.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import (
    INVALID,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
STOP_STRINGS = ["Question", "Assistant:", "<|separator|>"]


def build_chat_prompt(tokenizer, user_content: str) -> str:
    """Wrap `user_content` in the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


class TestFastDLLMv2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Efficient-Large-Model/Fast_dLLM_v2_7B"  # Fast dLLM v2 model
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "1",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "HierarchyBlock",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # Tokenizer is used client-side for chat template formatting, so that
        # both this test and the HF reference script see byte-identical input.
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------ #
    def _post_generate(self, prompt: str, max_new_tokens: int) -> dict:
        r = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "stop": STOP_STRINGS,
                },
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ #
    def test_gsm8k(self):
        num_shots = 5
        num_questions = 200
        max_new_tokens = 512
        parallel = 128

        # Load and format GSM8K the same way sglang.test.few_shot_gsm8k does,
        # then wrap each (few-shot + question) in the chat template.
        filename = download_and_cache_file(GSM8K_URL)
        lines = list(read_jsonl(filename))
        few_shot_examples = get_few_shot_examples(lines, num_shots)
        questions = [
            get_one_example(lines, i, include_answer=False)
            for i in range(num_questions)
        ]
        labels = [
            get_answer_value(lines[i]["answer"]) for i in range(num_questions)
        ]
        assert all(l != INVALID for l in labels), "GSM8K label extraction failed"

        prompts = [
            build_chat_prompt(self.tokenizer, few_shot_examples + q)
            for q in questions
        ]
        print(
            f"[test_gsm8k] {len(prompts)} prompts, first prompt len = "
            f"{len(self.tokenizer(prompts[0])['input_ids'])} tokens"
        )

        results: list = [None] * len(prompts)
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {
                pool.submit(self._post_generate, prompts[i], max_new_tokens): i
                for i in range(len(prompts))
            }
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                results[idx] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(prompts):
                    print(f"[test_gsm8k] {done}/{len(prompts)} done")
        total_latency = time.perf_counter() - t_start

        # Compute accuracy + throughput
        correct = 0
        invalid = 0
        total_new_tokens = 0
        for idx, r in enumerate(results):
            text = r.get("text", "")
            meta = r.get("meta_info", {}) or {}
            total_new_tokens += int(meta.get("completion_tokens", 0) or 0)
            pred = get_answer_value(text)
            if pred == INVALID:
                invalid += 1
                continue
            if pred == labels[idx]:
                correct += 1

        accuracy = correct / num_questions
        invalid_rate = invalid / num_questions
        output_throughput = (
            total_new_tokens / total_latency if total_latency > 0 else 0.0
        )
        metrics = {
            "accuracy": accuracy,
            "invalid": invalid_rate,
            "latency": total_latency,
            "output_throughput": output_throughput,
            "total_new_tokens": total_new_tokens,
        }
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (fast-dllm-v2, chat template) with tp1\n"
                f"accuracy={accuracy:.3f}, "
                f"throughput={output_throughput:.2f} token/s\n"
            )

        self.assertGreater(metrics["accuracy"], 0.00)
        self.assertGreater(metrics["output_throughput"], 10)

    # ------------------------------------------------------------------ #
    def test_bs_1_speed(self):
        max_new_tokens = 2048
        user_content = (
            "Please explain the idea of diffusion language models in detail, "
            "including how hierarchical block decoding can accelerate inference."
        )
        prompt = build_chat_prompt(self.tokenizer, user_content)

        t0 = time.perf_counter()
        out = self._post_generate(prompt, max_new_tokens=max_new_tokens)
        dt = time.perf_counter() - t0

        meta = out.get("meta_info", {}) or {}
        new_tokens = int(meta.get("completion_tokens", 0) or 0)
        speed = new_tokens / dt if dt > 0 else 0.0

        print(f"[test_bs_1_speed] new_tokens={new_tokens} dt={dt:.2f}s {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (fast-dllm-v2, chat template) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 10)
            else:
                # Fast dLLM v2 should be faster than standard AR decoding
                self.assertGreater(speed, 100)


if __name__ == "__main__":
    unittest.main()
