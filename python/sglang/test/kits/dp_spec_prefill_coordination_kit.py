"""Request-level correctness checks for DP attention with speculation."""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests


class DPSpecPrefillCoordinationKit:
    dp_size = 2
    dp_prefill_lines = 1536

    def _dp_generate(self, text, rank, tokens=256, rid=None):
        payload = {
            "text": text,
            "routed_dp_rank": rank,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
                "sampling_seed": 42,
            },
        }
        if rid is not None:
            payload["rid"] = rid
        response = requests.post(self.base_url + "/generate", json=payload, timeout=240)
        response.raise_for_status()
        return response.json()

    def _dp_output(self, result, tokens=256):
        self.assertEqual(result["meta_info"]["completion_tokens"], tokens)
        output = result.get("output_ids") or result["text"]
        self.assertTrue(output)
        return output

    def _dp_long_prompt(self):
        return (
            f"Fresh context {uuid.uuid4().hex}:\n"
            + "\n".join(
                f"Entry {i}: apples are fruit; water is liquid; the sky appears blue."
                for i in range(self.dp_prefill_lines)
            )
            + "\nSummarize these entries briefly."
        )

    def test_dp_spec_prefill_overlap(self):
        prompts = {
            rank: f"Example {rank}. Continue the positive integers separated by commas:\n1, 2, 3,"
            for rank in range(1, self.dp_size)
        }
        expected = {
            rank: self._dp_output(self._dp_generate(prompt, rank))
            for rank, prompt in prompts.items()
        }
        for round_id in range(3):
            with (
                self.subTest(round=round_id),
                ThreadPoolExecutor(max_workers=self.dp_size) as pool,
            ):
                futures = {
                    rank: pool.submit(self._dp_generate, prompt, rank)
                    for rank, prompt in prompts.items()
                }
                time.sleep(0.5)
                prefill = pool.submit(self._dp_generate, self._dp_long_prompt(), 0, 32)
                for rank, future in futures.items():
                    self.assertEqual(self._dp_output(future.result()), expected[rank])
                self._dp_output(prefill.result(), 32)
        for rank, prompt in prompts.items():
            self.assertEqual(
                self._dp_output(self._dp_generate(prompt, rank)), expected[rank]
            )

    def test_dp_spec_idle_rank_transition(self):
        prompt = "Continue this number sequence: 1, 2, 3, 4, 5,"
        expected = self._dp_output(self._dp_generate(prompt, 1))
        with ThreadPoolExecutor(max_workers=2) as pool:
            decode = pool.submit(self._dp_generate, prompt, 1)
            time.sleep(0.5)
            prefill = pool.submit(self._dp_generate, self._dp_long_prompt(), 0, 32)
            self.assertEqual(self._dp_output(decode.result()), expected)
            self._dp_output(prefill.result(), 32)
        self.assertEqual(self._dp_output(self._dp_generate(prompt, 1)), expected)

    def test_dp_spec_abort_during_prefill(self):
        prompt = "Continue this number sequence: 1, 2, 3, 4, 5,"
        expected = self._dp_output(self._dp_generate(prompt, 1))
        rid = "dp-spec-abort-" + uuid.uuid4().hex
        with ThreadPoolExecutor(max_workers=2) as pool:
            decode = pool.submit(self._dp_generate, prompt, 1)
            time.sleep(0.5)
            prefill = pool.submit(
                self._dp_generate, self._dp_long_prompt(), 0, 2048, rid
            )
            time.sleep(0.5)
            response = requests.post(
                self.base_url + "/abort_request", json={"rid": rid}, timeout=30
            )
            response.raise_for_status()
            self.assertEqual(self._dp_output(decode.result()), expected)
            cancelled = prefill.result()
            self.assertEqual(cancelled["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(self._dp_output(self._dp_generate(prompt, 1)), expected)
        self._dp_output(self._dp_generate("Count: 1, 2, 3,", 0, 32), 32)
