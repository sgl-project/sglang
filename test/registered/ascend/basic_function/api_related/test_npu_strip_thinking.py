import json
import logging
import os
import time
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)
HARNONY = (
    "Seth is twice as old as Brooke. In 2 years, the sum of their ages will be 28. How old is Seth?"
    * 20
)


class _CacheResultStore:
    """Cross-category shared storage results"""

    disabled_cached = None
    enabled_cached = None


class TestStripThinkingCacheBase(CustomTestCase):
    @classmethod
    def _build_env(cls):
        env = os.environ.copy()
        env["SGLANG_SET_CPU_AFFINITY"] = "1"
        env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        env["STREAMS_PER_DEVICE"] = "32"
        env["HCCL_BUFFSIZE"] = "1536"
        env["HCCL_OP_EXPANSION_MODE"] = "AIV"
        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "32"
        env["SGLANG_DEEPEP_BF16_DISPATCH"] = "1"
        env["ENABLE_ASCEND_MOE_NZ"] = "1"
        return env

    def _post_generate(self, body, timeout=600):
        # Send a generate request.
        req = urllib.request.Request(
            self.base_url + "/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def _flush_cache(self):
        # Clear cache
        req = urllib.request.Request(f"{self.base_url}/flush_cache", method="POST")
        urllib.request.urlopen(req, timeout=30).read()

    def _run_test_and_get_cached(self):
        """Send the request and record cached_tokens."""

        logging.warning("[info]Flushing cache...")
        self._flush_cache()
        time.sleep(1)

        # Sending the request for the first time
        logging.warning("[info]First request (warming up)...")
        request1 = self._post_generate(
            {
                "text": HARNONY,
                "return_logprob": True,
                "logprob_start_len": 0,
                "return_token_str": True,
                "sampling_parms": {"max_new_tokens": 100, "temperature": 0},
                "require_reasoning": True,
            }
        )

        prompt_ids = [x[1] for x in request1["meta_info"]["input_token_logprobs"]]
        oids = request1["output_ids"]
        logging.warning(f"[Info] prompt = {len(prompt_ids)}, output={len(oids)}")
        # Construct and reproduce input_ids
        k = 50
        probe = prompt_ids + oids[:k]
        logging.warning(f"[Info] probe length = {len(probe)}")

        # Send the second request.
        request2 = self._post_generate(
            {
                "input_ids": probe,
                "sampling_parms": {"max_new_tokens": 100, "temperature": 0},
                "require_reasoning": True,
            },
            timeout=60,
        )
        cached = request2["meta_info"].get("cached_tokens", 0)
        return cached


class TestStripThinkingCacheDisable(TestStripThinkingCacheBase):
    """testcase：without configuration --strip-thinking-cache"""

    @classmethod
    def setUpClass(cls):
        env = cls._build_env()
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        args = [
            "--trust-remote-code",
            "--tp-size",
            2,
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "ascend",
            "--chunked-prefill-size",
            -1,
            "--disable-cuda-graph",
            "--reasoning-parser",
            "qwen3",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_strip_thinking_cache_disable(self):
        """Send a request without configuring `--strip-thinking-cache` and record the value of `cached_tokens`."""
        cached = self._run_test_and_get_cached()
        _CacheResultStore.disabled_cached = cached
        logging.warning(f"[Result] Disabled cached_tokens = {cached}")


class TestStripThinkingCacheEnable(TestStripThinkingCacheBase):
    """testcase：Configuration --strip-thinking-cache"""

    @classmethod
    def setUpClass(cls):
        env = cls._build_env()
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        args = [
            "--trust-remote-code",
            "--tp-size",
            2,
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "ascend",
            "--chunked-prefill-size",
            -1,
            "--disable-cuda-graph",
            "--reasoning-parser",
            "qwen3",
            "--strip-thinking-cache",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_strip_thinking_cache_enable(self):
        """Configure with `--strip-thinking-cache` and record the cached_tokens results,
        then compare them with the cached_tokens results obtained without configuring `--strip-thinking-cache`.
        """
        cached = self._run_test_and_get_cached()
        _CacheResultStore.enabled_cached = cached
        logging.warning(f"[Result] Enabled cached_tokens = {cached}")

        # contrast
        self.assertLess(
            _CacheResultStore.enabled_cached,
            _CacheResultStore.disabled_cached,
            f"Cache was not reduced.",
        )


if __name__ == "__main__":
    unittest.main()
