import json
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-c", runner_config="4-gpu-h100")


class TestDPSpecPrefillCoordinationMixed(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN)
        cls.counting_prompt = (
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Count from 1 to 1000 in order, separated by commas. Output only the numbers.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "1, 2, 3,"
        )
        with (
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(True),
        ):
            cls.process = popen_launch_server(
                DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--enable-deterministic-inference",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-num-steps",
                    "6",
                    "--speculative-eagle-topk",
                    "10",
                    "--speculative-num-draft-tokens",
                    "32",
                    "--speculative-draft-model-path",
                    DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
                    "--tp-size",
                    "2",
                    "--dp-size",
                    "2",
                    "--enable-dp-attention",
                    "--enable-dp-lm-head",
                    "--moe-dense-tp-size",
                    "1",
                    "--attention-backend",
                    "fa3",
                    "--mem-fraction-static",
                    "0.75",
                    "--cuda-graph-max-bs-decode",
                    "4",
                    "--disable-prefill-cuda-graph",
                    "--enable-mixed-chunk",
                    "--chunked-prefill-size",
                    "16384",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt, rank, tokens, started=None):
        streaming = started is not None
        payload = {
            "text": prompt,
            "routed_dp_rank": rank,
            "stream": streaming,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
                "sampling_seed": 42,
            },
        }
        with requests.post(
            self.base_url + "/generate", json=payload, stream=streaming, timeout=240
        ) as response:
            response.raise_for_status()
            if streaming:
                result = None
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    result = json.loads(data)
                    if result.get("meta_info", {}).get("completion_tokens", 0) > 0:
                        started.set()
                self.assertIsNotNone(result)
            else:
                result = response.json()
        self.assertEqual(result["meta_info"]["completion_tokens"], tokens)
        self.assertTrue(result["text"])
        if prompt == self.counting_prompt:
            numbers = [int(value.strip()) for value in result["text"].split(",")[:-1]]
            self.assertGreater(len(numbers), 128)
            self.assertEqual(numbers, list(range(4, 4 + len(numbers))))
        return result["text"], time.monotonic()

    def test_mixed_rank_and_speculative_peer_match_isolated_outputs(self):
        prompt = self.counting_prompt
        expected = {rank: self._generate(prompt, rank, 1024)[0] for rank in range(2)}
        for mixed_rank in range(2):
            long_prompt = (
                f"Fresh context {uuid.uuid4().hex}:\n"
                + "\n".join(
                    f"Entry {i}: apples are fruit; water is liquid; the sky appears blue."
                    for i in range(1536)
                )
                + "\nSummarize these entries briefly."
            )
            started = [threading.Event(), threading.Event()]
            with self.subTest(mixed_rank=mixed_rank), ThreadPoolExecutor(3) as pool:
                decodes = [
                    pool.submit(self._generate, prompt, rank, 1024, started[rank])
                    for rank in range(2)
                ]
                for event in started:
                    self.assertTrue(event.wait(60), "Decode did not start")
                self.assertTrue(
                    all(not f.done() for f in decodes), "Decode finished before prefill"
                )
                prefill_started = time.monotonic()
                prefill = pool.submit(self._generate, long_prompt, mixed_rank, 32)
                results = [future.result(timeout=240) for future in decodes]
                prefill_output, _ = prefill.result(timeout=240)
                for rank, (actual, completed) in enumerate(results):
                    self.assertGreater(completed, prefill_started)
                    self.assertEqual(actual, expected[rank])
                self.assertEqual(
                    prefill_output, self._generate(long_prompt, mixed_rank, 32)[0]
                )
                for rank in range(2):
                    self.assertEqual(
                        self._generate(prompt, rank, 1024)[0], expected[rank]
                    )


if __name__ == "__main__":
    unittest.main()
