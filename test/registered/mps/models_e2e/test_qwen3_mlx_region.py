"""Compare the exported region with the standard Torch-owned MPS server."""

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
                    "--decode-log-interval",
                    "1",
                    "--cuda-graph-bs-decode",
                    "1",
                    "2",
                    "4",
                    "--cuda-graph-bs-prefill",
                    "128",
                    "512",
                ],
            )
            try:

                def generate(text, *, input_ids=False):
                    response = requests.post(
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        json={
                            "input_ids" if input_ids else "text": text,
                            "sampling_params": {
                                "temperature": 0,
                                "ignore_eos": True,
                                "max_new_tokens": 8,
                            },
                        },
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
            finally:
                kill_process_tree(process.pid, wait_timeout=30)
                process.wait(timeout=5)
            log.seek(0)
            output = log.read()
        if enabled:
            self.assertIn("exported 7/7 shapes at startup", output)
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
        return (
            [cold["output_ids"], warm["output_ids"]]
            + [item["output_ids"] for item in batch]
            + [item["output_ids"] for item in packed]
            + [item["output_ids"] for item in packed_warm]
        )

    def test_region_matches_eager_and_reuses_torch_radix_cache(self):
        self.assertEqual(self._run_server(False), self._run_server(True))


if __name__ == "__main__":
    unittest.main()
