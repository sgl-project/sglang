"""Bit-exact KL tests for the Mooncake direct linker on a FULL + SWA + Mamba model.

The shrunken Inkling checkpoint scores every token identically in prefill and
decode under deterministic inference, so a nonzero KL here means a state
restored from Mooncake (KV, SWA window or Mamba checkpoint) is wrong.
"""

import os
import random
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _replay_and_compare_kl,
)
from sglang.test.kl_test_utils import get_input_ids
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")

_MODEL_PATH = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")

# Measured 0; the floor only absorbs a stray ulp under the strict `<`.
KL_DIV_THRESHOLD = 1e-9
PAGE_SIZE = 128
TRACK_INTERVAL = 128


def _random_suffixes(n: int, length: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


class TestInklingUnifiedCacheLinkerKL(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.mooncake.start()
        cls.process = None
        other_args = [
            "--trust-remote-code",
            "--revision",
            _MODEL_REVISION,
            "--attention-backend",
            "fa4",
            "--page-size",
            str(PAGE_SIZE),
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--mamba-track-interval",
            str(TRACK_INTERVAL),
            "--enable-deterministic-inference",
            "--mem-fraction-static",
            "0.6",
            "--max-running-requests",
            "1",
            "--enable-cache-report",
            "--enable-unified-cache-external-linker",
        ]
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                env=unified_radix_tree_server_env(
                    "python", **cls.mooncake.server_env()
                ),
            )
            cls.input_ids = get_input_ids(
                tokenizer_path=cls.model, num_samples=6, trust_remote_code=True
            )
        except Exception:
            try:
                if cls.process is not None:
                    terminate_and_kill_process_tree(cls.process, wait_timeout=60)
            finally:
                cls.mooncake.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.process is not None:
                terminate_and_kill_process_tree(cls.process, wait_timeout=60)
        finally:
            cls.mooncake.stop()

    def test_load_back_after_flush(self):
        """Every second-turn prefix comes back from Mooncake alone."""
        n = 6
        first_turn = [list(ids[:1024]) for ids in self.input_ids[:n]]
        _flush_cache(self.base_url)
        first = [_generate(self.base_url, [ids], 256)[0] for ids in first_turn]

        # Drop the device tree; Mooncake keeps what the first turn offloaded.
        _flush_cache(self.base_url)
        suffixes = _random_suffixes(n, 256, seed=100)
        second_turn = [
            first_turn[i] + first[i]["output_ids"] + suffixes[i] for i in range(n)
        ]
        results = []
        for i, ids in enumerate(second_turn):
            (result,) = _generate(self.base_url, [ids], 512, return_logprob=True)
            history_len = len(first_turn[i]) + len(first[i]["output_ids"])
            expected = (history_len - 1) // TRACK_INTERVAL * TRACK_INTERVAL
            cached_tokens = result["meta_info"]["cached_tokens"]
            details = result["meta_info"].get("cached_tokens_details") or {}
            self.assertGreaterEqual(cached_tokens, expected)
            self.assertEqual(int(details.get("host", 0)), cached_tokens)
            results.append(result)

        _replay_and_compare_kl(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            [second_turn[i] + results[i]["output_ids"] for i in range(n)],
            [_extract_output_logprobs(result) for result in results],
            label="linker_load_back_after_flush",
            sampling_temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
