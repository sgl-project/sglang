"""MiMo V2.5 HiCache host load-back accuracy regression test."""

import random
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MIMO_MODEL = "XiaomiMiMo/MiMo-V2.5"
MIMO_LAUNCH_TIMEOUT = 3600

# MiMo V2.5 is pre-cached on the eight-H200 runner. The test is intentionally
# in extra-b because it exercises the asymmetric MHA host pool end to end.
register_cuda_ci(est_time=1200, stage="extra-b", runner_config="8-gpu-h200")


class TestUnifiedMiMoHiCacheLoadBackKL(CustomTestCase):
    """Verify KL accuracy after asymmetric MHA KV is evicted to and loaded from L2."""

    page_size = 64
    prompt_len = 1024
    max_total_tokens = 4096
    kl_threshold = 0.005

    @classmethod
    def setUpClass(cls):
        cls.model = MIMO_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=MIMO_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--mm-enable-dp-encoder",
                "--mem-fraction-static",
                "0.65",
                "--chunked-prefill-size",
                "16384",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--enable-multi-layer-eagle",
                "--reasoning-parser",
                "mimo",
                "--tool-call-parser",
                "mimo",
                "--page-size",
                str(cls.page_size),
                "--max-total-tokens",
                str(cls.max_total_tokens),
                "--max-running-requests",
                "4",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "kernel",
                "--hicache-mem-layout",
                "page_first",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @classmethod
    def _prompt(cls, seed: int) -> list[int]:
        rng = random.Random(seed)
        return [rng.randint(1, 30000) for _ in range(cls.prompt_len)]

    def _generate_one(self, input_ids, max_new_tokens, return_logprob=False):
        results = _generate(
            self.base_url,
            [input_ids],
            max_new_tokens=max_new_tokens,
            return_logprob=return_logprob,
            temperature=0,
        )
        self.assertEqual(len(results), 1)
        return results[0]

    def test_host_load_back_logprobs_match_prefill_replay(self):
        """Force L2 eviction, then compare load-back output logprobs with replay."""
        base_prompt = self._prompt(1)
        pressure_prompts = [self._prompt(seed) for seed in range(2, 6)]

        _flush_cache(self.base_url)
        self._generate_one(base_prompt, max_new_tokens=1)

        # Four unique page-aligned prefixes fill the 4096-token L1 cache and
        # evict the oldest prefix (base_prompt) to the HiCache host tier.
        for prompt in pressure_prompts:
            self._generate_one(prompt, max_new_tokens=1)

        load_back = self._generate_one(
            base_prompt, max_new_tokens=8, return_logprob=True
        )
        meta_info = load_back["meta_info"]
        cached_details = meta_info.get("cached_tokens_details") or {}
        host_cached_tokens = int(cached_details.get("host", 0))
        self.assertGreater(
            host_cached_tokens,
            0,
            "Expected the original prefix to be restored from the HiCache host tier; "
            f"got cached_tokens={meta_info.get('cached_tokens')}, "
            f"cached_tokens_details={cached_details}",
        )

        output_logprobs = [_extract_output_logprobs(load_back)]
        replay_input_ids = [base_prompt + load_back["output_ids"]]
        input_logprobs = _get_input_logprobs(
            self.base_url,
            replay_input_ids,
            output_logprobs,
            temperature=0,
        )
        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            {self.model: {"kl_div": self.kl_threshold}},
            self.model,
            "hicache_host_load_back",
        )


if __name__ == "__main__":
    unittest.main()
