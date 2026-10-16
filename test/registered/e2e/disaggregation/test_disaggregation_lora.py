"""PD LoRA requests remain correct under adapter-slot pressure."""

import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=400, stage="base-b", runner_config="2-gpu-large")

ADAPTERS = {
    "fact": "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
    "guard": "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "sql": "philschmid/code-llama-3-1-8b-text-to-sql-lora",
}
LORA_ARGS = [
    "--enable-deterministic-inference",
    "--cuda-graph-max-bs-prefill",
    "1024",
    "--enable-lora",
    "--lora-paths",
    *[f"{name}={path}" for name, path in ADAPTERS.items()],
    "--max-loras-per-batch",
    "2",
]
PROMPTS = [
    "Give three facts about the planet Mars.",
    "Translate this SQL request into a query: list the names of all customers.",
    "Is the following on topic for a cooking assistant? 'How do I fix my car?'",
    "Write one sentence about the sea.",
]


def generate(url: str, prompt: str, lora_path=None) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": 128},
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    response = requests.post(f"{url}/generate", json=payload, timeout=600)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["meta_info"]["finish_reason"]["type"] != "abort", result
    assert result["text"].strip(), result
    return result["text"]


class TestDisaggregationLoRA(PDDisaggregationServerBase):
    extra_prefill_args = LORA_ARGS
    extra_decode_args = LORA_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

    def test_more_adapters_than_slots_match_single_server(self):
        """Excess adapters must wait without aborting or changing outputs."""
        names = [None, *ADAPTERS]
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=LORA_ARGS,
        )
        try:
            expected = {
                (prompt, name): generate(self.base_url, prompt, name)
                for prompt in PROMPTS
                for name in names
            }
        finally:
            terminate_and_kill_process_tree(process)

        for name in ADAPTERS:
            self.assertTrue(
                any(
                    expected[prompt, name] != expected[prompt, None]
                    for prompt in PROMPTS
                ),
                name,
            )

        self.launch_all()
        jobs = list(zip(PROMPTS, names)) * 2
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            texts = list(pool.map(lambda job: generate(self.lb_url, *job), jobs))
        for job, text in zip(jobs, texts):
            self.assertEqual(text, expected[job], job)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
