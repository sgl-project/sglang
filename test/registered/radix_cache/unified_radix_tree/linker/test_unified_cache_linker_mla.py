"""Mooncake direct linker on a plain MLA pool (DeepSeek-V2), no hybrid stack.

A plain MLA pool used to have no linker layout at all; this covers the path end
to end: the prefix is written through to Mooncake, evicted from the device pool,
and restored from the store on the next request. The restored decode must match
the device-hit decode token for token, which is the accuracy signal for a KV
restore path.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import _random_suffixes
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    get_input_ids,
)
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
    est_time=150,
    stage="extra-b",
    # Direct GPU transfers require an RDMA-capable runner, even for TP=1.
    runner_config="4-gpu-h100",
)

_MODEL_PATH = os.environ.get(
    "SGLANG_LINKER_MLA_MODEL", "vllm-project/tiny-random-DeepseekV2ForCausalLM"
)

PAGE_SIZE = 64
PREFIX_LEN = 128


class TestPlainMLAUnifiedCacheLinker(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.mooncake.start()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--page-size",
                str(PAGE_SIZE),
                "--enable-deterministic-inference",
                "--mem-fraction-static",
                "0.6",
                # The pool is sized from the prompt, not the card: a tiny MLA
                # checkpoint on a large GPU would otherwise reserve hundreds of
                # GB and OOM while capturing decode CUDA graphs.
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "1",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
            ],
            env=unified_radix_tree_server_env("python", **cls.mooncake.server_env()),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=4, trust_remote_code=True
        )

    def test_host_hit_matches_device_hit(self):
        prompts = [list(ids[: PREFIX_LEN + 1]) for ids in self.input_ids]
        for ids in prompts:
            _generate(self.base_url, [ids], 1)
        requests = [
            prompt + suffix
            for prompt, suffix in zip(
                prompts, _random_suffixes(len(prompts), PAGE_SIZE - 1, seed=100)
            )
        ]

        def run():
            return [
                _generate(self.base_url, [ids], 32, return_logprob=True)[0]
                for ids in requests
            ]

        device_results = run()
        _flush_cache(self.base_url)
        host_results = run()

        for device, host in zip(device_results, host_results):
            for result, host_tokens in ((device, 0), (host, PREFIX_LEN)):
                meta = result["meta_info"]
                details = meta.get("cached_tokens_details") or {}
                self.assertEqual(meta["cached_tokens"], PREFIX_LEN)
                self.assertEqual(int(details.get("host", 0)), host_tokens)
            self.assertEqual(host["output_ids"], device["output_ids"])
            self.assertEqual(
                _extract_output_logprobs(host), _extract_output_logprobs(device)
            )


if __name__ == "__main__":
    unittest.main()
