"""Compare the exported region with the standard Torch-owned MPS server."""

import math
import os
import tempfile
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_mps_ci(est_time=240, suite="stage-b-e2e-mps")


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
class TestQwen3MlxRegion(CustomTestCase):
    def _run_server(self, enabled):
        model = try_cached_model(
            os.environ.get("SGLANG_MPS_TEST_MODEL", "Qwen/Qwen3-0.6B")
        )
        env = dict(
            os.environ,
            SGLANG_USE_MLX="0",
            SGLANG_ENABLE_MLX_WHOLE_REGION=str(int(enabled)),
        )
        with tempfile.TemporaryFile(mode="w+") as log:
            process = popen_launch_server(
                model,
                DEFAULT_URL_FOR_TEST,
                timeout=300,
                device="mps",
                env=env,
                return_stdout_stderr=(log, log),
                other_args=[
                    "--device",
                    "mps",
                    "--disable-overlap-schedule",
                    "--attention-backend",
                    "torch_native",
                    "--sampling-backend",
                    "pytorch",
                    "--dtype",
                    "bfloat16",
                    "--mem-fraction-static",
                    "0.6",
                    "--max-total-tokens",
                    "4096",
                    "--context-length",
                    "2048",
                    "--chunked-prefill-size",
                    "512",
                    "--decode-log-interval",
                    "1",
                    "--cuda-graph-bs-decode",
                    "1",
                    "2",
                    "4",
                    "--cuda-graph-bs-prefill",
                    "128",
                    "512",
                    "--cuda-graph-config",
                    '{"prefill":{"full_prefill_max_req":4}}',
                ],
            )
            try:

                def generate(text, *, input_ids=False, logprob_start=None):
                    payload = {
                        "input_ids" if input_ids else "text": text,
                        "sampling_params": {
                            "temperature": 0,
                            "ignore_eos": True,
                            "max_new_tokens": 8,
                        },
                    }
                    if logprob_start is not None:
                        payload.update(
                            return_logprob=True, logprob_start_len=logprob_start
                        )
                    response = requests.post(
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        json=payload,
                        timeout=120,
                    )
                    response.raise_for_status()
                    return response.json()

                prompt = (
                    "This deterministic prefix describes science and geography. " * 24
                    + "The capital of France is"
                )
                cold, warm = generate(prompt), generate(prompt)
                # Three requests exercise decode padding onto a four-row bucket.
                batch = generate(["The capital of France is", "2 + 2 =", "The sky is"])
                self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
                self.assertEqual(cold["output_ids"], warm["output_ids"])
                # Three fresh requests pad onto four independent 128-token
                # segments, with both token and request padding exercised.
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model)
                packed_inputs = []
                for country in ("France", "Canada", "Japan"):
                    tail = tokenizer.encode(
                        f"Answer briefly. The capital of {country} is"
                    )
                    prefix = tokenizer.encode(
                        f"{country}: A factual question about geography. " * 30
                    )
                    packed_inputs.append(prefix[: 120 - len(tail)] + tail)
                requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/flush_cache", timeout=30
                ).raise_for_status()
                packed = generate(packed_inputs, input_ids=True)
                packed_warm = generate(packed_inputs, input_ids=True)
                self.assertTrue(
                    all(item["meta_info"]["cached_tokens"] == 0 for item in packed)
                )
                self.assertTrue(
                    all(item["meta_info"]["cached_tokens"] > 0 for item in packed_warm)
                )
                self.assertEqual(
                    [item["output_ids"] for item in packed],
                    [item["output_ids"] for item in packed_warm],
                )
                scored = []

                def score(ids, start):
                    result = generate(ids, input_ids=True, logprob_start=start)
                    values = result["meta_info"]["input_token_logprobs"]
                    self.assertEqual([row[1] for row in values], ids[start:])
                    self.assertIsNone(values[0][0])
                    self.assertTrue(all(math.isfinite(row[0]) for row in values[1:]))
                    scored.append(result["output_ids"])
                    return result

                requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/flush_cache", timeout=30
                ).raise_for_status()
                score(packed_inputs[0], 0)
                requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/flush_cache", timeout=30
                ).raise_for_status()
                suffix = score(packed_inputs[0], 80)
                requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/flush_cache", timeout=30
                ).raise_for_status()
                generate(packed_inputs[0], input_ids=True)
                cached_suffix = score(packed_inputs[0], 80)
                self.assertGreater(cached_suffix["meta_info"]["cached_tokens"], 0)
                self.assertEqual(suffix["output_ids"], cached_suffix["output_ids"])
                requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/flush_cache", timeout=30
                ).raise_for_status()
                score(packed_inputs[0] * 9, 0)
            finally:
                kill_process_tree(process.pid, wait_timeout=30)
                process.wait(timeout=5)
            log.seek(0)
            output = log.read()
        if enabled:
            self.assertIn("exported 9/9 shapes at startup", output)
            self.assertNotIn("MLX region export failed", output)
            self.assertNotIn("MLX region warm-up execution failed", output)
            self.assertNotIn("MLX region disabled for this model", output)
            self.assertRegex(output, r"Prefill batch[^\n]+cuda graph: True")
            self.assertRegex(
                output,
                r"Prefill batch[^\n]+#new-seq: 3[^\n]+#new-token: 360[^\n]+cuda graph: True",
            )
            self.assertRegex(
                output, r"Decode batch[^\n]+#running-req: 3[^\n]+cuda graph: True"
            )
            self.assertRegex(
                output, r"Prefill batch[^\n]+#new-token: 512[^\n]+cuda graph: True"
            )
            self.assertRegex(
                output, r"Prefill batch[^\n]+#cached-token: 80[^\n]+cuda graph: True"
            )
        return (
            [cold["output_ids"], warm["output_ids"]]
            + [item["output_ids"] for item in batch]
            + [item["output_ids"] for item in packed]
            + [item["output_ids"] for item in packed_warm]
            + scored
        )

    def test_region_matches_eager_and_reuses_torch_radix_cache(self):
        self.assertEqual(self._run_server(False), self._run_server(True))


if __name__ == "__main__":
    unittest.main()
