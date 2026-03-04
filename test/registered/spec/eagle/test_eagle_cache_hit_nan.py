import unittest
from typing import Optional

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)

MIN_OVERALL_CACHE_HIT_RATE = 0.15


class TestEagleCacheHitNaN(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE
        cls.draft_model = DEFAULT_DRAFT_MODEL_EAGLE
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model",
            cls.draft_model,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "6",
            "--mem-fraction-static",
            "0.7",
            "--enable-nan-detection",
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _flush_cache(self):
        resp = requests.post(self.base_url + "/flush_cache", timeout=30)
        self.assertEqual(resp.status_code, 200)

    def _send_generate(self, prompt: str) -> Optional[int]:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, msg=response.text[:500])
        self.assertIsNone(
            self.process.poll(), "Server crashed during cache-hit request sequence"
        )
        data = response.json()
        if isinstance(data, dict):
            return (data.get("meta_info") or {}).get("cached_tokens")
        return None

    def test_multiturn_cache_hit_no_nan(self):
        metrics = run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=4,
            num_rounds=3,
            request_length=256,
            output_length=32,
        )
        self.assertIsNone(
            self.process.poll(), "Server crashed, likely due to NaN on cache hit"
        )
        self.assertGreater(
            metrics["overall"]["total_cached_tokens"], 0, "No cache-hit evidence found"
        )
        self.assertGreater(
            metrics["overall"]["cache_hit_rate"],
            MIN_OVERALL_CACHE_HIT_RATE,
            f"Overall cache hit rate below threshold {MIN_OVERALL_CACHE_HIT_RATE}",
        )

    def test_shared_prefix_cache_hit_no_nan(self):
        self._flush_cache()
        shared_prefix = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n" * 20

        self._send_generate(shared_prefix + "Tell me about Rome.[/INST]")
        cached_tokens_b = self._send_generate(
            shared_prefix + "Tell me about Egypt.[/INST]"
        )
        cached_tokens_c = self._send_generate(
            shared_prefix + "Tell me about Greece.[/INST]"
        )

        cached_tokens_values = [
            x for x in (cached_tokens_b, cached_tokens_c) if x is not None
        ]
        self.assertTrue(
            cached_tokens_values, "Missing cached_tokens in response meta_info"
        )
        self.assertGreater(
            max(cached_tokens_values), 0, "No shared-prefix cache hit detected"
        )
        self.assertIsNone(
            self.process.poll(), "Server crashed, likely due to NaN on cache hit"
        )


if __name__ == "__main__":
    unittest.main()
