import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="base-c", runner_config="4-gpu-h100")


class TestDFlashDPAttention(CustomTestCase):
    def test_idle_peers_cached_prefixes_and_concurrent_sampling(self):
        model = os.environ.get("SGLANG_TEST_DFLASH_MODEL", DEFAULT_TARGET_MODEL_DFLASH)
        draft = os.environ.get("SGLANG_TEST_DFLASH_DRAFT", DEFAULT_DRAFT_MODEL_DFLASH)
        for dp_size in (2, 4):
            with self.subTest(dp_size=dp_size):
                process = popen_launch_server(
                    model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--trust-remote-code",
                        "--tp-size",
                        "4",
                        "--dp-size",
                        str(dp_size),
                        "--enable-dp-attention",
                        "--enable-dp-lm-head",
                        "--speculative-algorithm",
                        "DFLASH",
                        "--speculative-draft-model-path",
                        draft,
                        "--context-length",
                        "8192",
                        "--max-running-requests",
                        "8",
                        "--cuda-graph-max-bs-decode",
                        "8",
                        "--chunked-prefill-size",
                        "1024",
                        "--mem-fraction-static",
                        "0.7",
                    ],
                    env={"SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1"},
                )
                try:
                    self._exercise(dp_size)
                finally:
                    terminate_and_kill_process_tree(process)

    def _exercise(self, dp_size, base_url=DEFAULT_URL_FOR_TEST):
        prompt = (
            "This is a counting exercise. " * 64
            + "\nCount from 1 to 1000, separated by commas.\n1, 2, 3,"
        )

        def generate(rank, temperature=0):
            response = requests.post(
                base_url + "/generate",
                json={
                    "text": prompt,
                    "routed_dp_rank": rank,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": 96,
                        "ignore_eos": True,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            output = response.json()
            meta = output["meta_info"]
            self.assertEqual(meta["dp_rank"], rank)
            self.assertEqual(meta["completion_tokens"], 96)
            self.assertGreater(meta["spec_verify_ct"], 0)
            self.assertTrue(output["text"])
            return output

        for rank in range(dp_size):
            first = generate(rank)
            cached = generate(rank)
            self.assertGreater(cached["meta_info"]["cached_tokens"], 0)
            self.assertEqual(first["output_ids"], cached["output_ids"])

        with ThreadPoolExecutor(dp_size * 2) as pool:
            futures = [
                pool.submit(generate, rank, temperature)
                for temperature in (0, 1)
                for rank in range(dp_size)
            ]
            for future in futures:
                future.result()


if __name__ == "__main__":
    unittest.main()
