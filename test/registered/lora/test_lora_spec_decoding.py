"""E2E test for multi-adapter LoRA + EAGLE-family speculative decoding.

Adapters apply to the target model only; one shared draft runs unadapted.

Deliberately asserts serving properties rather than exact output text.
Greedy decoding is not bitwise reproducible across batch shapes or server
restarts here -- reduction order changes flip a token and greedy amplifies
it -- so text equality between configurations is a flaky assertion, not a
correctness oracle. Losslessness is verified out of band by
test/manual/lora/run_spec_lora_matrix.py, which measures that noise floor
first. What this guards is what CI can hold stable:

- the server starts at all with multiple adapters + speculation (it used to
  crash loading the target's adapters into the draft model);
- adapters are actually applied during target-verify (adapter output differs
  from base output);
- a mixed-adapter batch, and a batch wider than the cuda-graph capture, are
  served without error (the eager verify path used to crash on
  extend_seq_lens_cpu=None);
- speculation is really running (accept length above 1).
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List three primary colors.",
    "Write a one-sentence story about a brave detective on Mars.",
]
# Ranks 8 and 64: a mixed-rank batch is what the per-request lora_ranks
# indexing has to get right, and uniform ranks would hide a mixup.
ADAPTERS = [
    ("fact", "algoprog/fact-generation-llama-3.1-8b-instruct-lora"),
    ("guard", "nvidia/llama-3.1-nemoguard-8b-topic-control"),
]
SAMPLING = {"temperature": 0, "max_new_tokens": 32}


class TestEagle3MultiLoRA(CustomTestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                # Canonical EAGLE3 sglang config, as in
                # test/registered/core/test_basic_sanity_eagle3.py: the draft
                # checkpoint is fp16, and bf16 + flashinfer cutlass RMSNorm
                # hits a dtype mismatch on the draft's input_layernorm.
                "--dtype=float16",
                "--attention-backend=triton",
                "--speculative-algorithm=EAGLE3",
                f"--speculative-draft-model-path={DEFAULT_DRAFT_MODEL_EAGLE3}",
                "--speculative-num-steps=5",
                "--speculative-eagle-topk=8",
                "--speculative-num-draft-tokens=32",
                "--enable-lora",
                "--lora-backend=triton",
                "--max-lora-rank=64",
                # +1: the base model occupies a memory-pool slot too, and the
                # batches below co-batch base requests with adapter requests.
                f"--max-loras-per-batch={len(ADAPTERS) + 1}",
                "--mem-fraction-static=0.7",
                "--lora-paths",
            ]
            + [f"{name}={path}" for name, path in ADAPTERS],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _generate(self, texts, lora_paths):
        response = requests.post(
            self.base_url + "/generate",
            json={"text": texts, "lora_path": lora_paths, "sampling_params": SAMPLING},
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), len(texts))
        for item in results:
            self.assertTrue(item["text"].strip(), f"empty output in {results}")
        return [item["text"] for item in results]

    def test_adapters_are_applied_under_speculation(self):
        base = self._generate(PROMPTS, [None] * len(PROMPTS))
        for name, _ in ADAPTERS:
            adapted = self._generate(PROMPTS, [name] * len(PROMPTS))
            self.assertNotEqual(
                adapted,
                base,
                f"adapter {name} matched the base model on every prompt; LoRA "
                "was likely not applied during target-verify",
            )

    def test_mixed_adapter_and_wide_batches_are_served(self):
        routes = [None] + [name for name, _ in ADAPTERS]
        self._generate(
            [p for p in PROMPTS for _ in routes],
            [r for _ in PROMPTS for r in routes],
        )
        # Wider than the default cuda-graph capture, so target-verify falls
        # back to the eager path.
        wide = 24
        self._generate(
            [PROMPTS[i % len(PROMPTS)] for i in range(wide)],
            [routes[i % len(routes)] for i in range(wide)],
        )

    def test_speculation_is_active(self):
        self._generate(PROMPTS, [ADAPTERS[0][0]] * len(PROMPTS))
        info = requests.get(self.base_url + "/get_server_info").json()
        accept_length = info["internal_states"][0]["avg_spec_accept_length"]
        self.assertGreater(
            accept_length,
            1.0,
            f"no drafts accepted with LoRA enabled: {accept_length}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
