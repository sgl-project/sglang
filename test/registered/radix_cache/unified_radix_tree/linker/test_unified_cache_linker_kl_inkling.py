"""Bit-exact KL for the Mooncake direct linker on Inkling (FULL + SWA + Mamba):
prefill and decode score identically, so any nonzero KL is a wrong restored state."""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import _random_suffixes
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

register_cuda_ci(
    est_time=300,
    stage="extra-b",
    # Direct GPU transfers require an RDMA-capable runner, even for TP=1.
    runner_config="4-gpu-h100",
)

_MODEL_PATH = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")

# Measured 0; the floor only absorbs the strict `<`.
KL_DIV_THRESHOLD = 1e-9
PAGE_SIZE = 128
TRACK_INTERVAL = 128


class TestInklingUnifiedCacheLinkerKL(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.mooncake.start()
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
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=unified_radix_tree_server_env("python", **cls.mooncake.server_env()),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=2, trust_remote_code=True
        )

    def test_load_back_after_flush(self):
        """Every second-turn prefix comes back from Mooncake alone."""
        first_turn = [list(ids[:1024]) for ids in self.input_ids]
        histories = [
            ids + _generate(self.base_url, [ids], 256)[0]["output_ids"]
            for ids in first_turn
        ]

        _flush_cache(self.base_url)
        sequences, logprobs = [], []
        for history, suffix in zip(
            histories, _random_suffixes(len(histories), 256, seed=100)
        ):
            ids = history + suffix
            (result,) = _generate(self.base_url, [ids], 512, return_logprob=True)
            expected = (len(history) - 1) // TRACK_INTERVAL * TRACK_INTERVAL
            cached_tokens = result["meta_info"]["cached_tokens"]
            details = result["meta_info"].get("cached_tokens_details") or {}
            self.assertGreaterEqual(cached_tokens, expected)
            self.assertEqual(int(details.get("host", 0)), cached_tokens)
            sequences.append(ids + result["output_ids"])
            logprobs.append(_extract_output_logprobs(result))

        _replay_and_compare_kl(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            sequences,
            logprobs,
            label="linker_load_back_after_flush",
            sampling_temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
