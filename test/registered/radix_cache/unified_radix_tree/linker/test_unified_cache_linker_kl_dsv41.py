"""DeepSeek-V4.1-Flash UnifiedRadixCache direct-linker load-back KL tests."""

import json
import time
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.kl_test_utils import _flush_cache, _generate, compare_kl_divergence
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=2400, stage="extra-b", runner_config="8-gpu-h200")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
LAUNCH_TIMEOUT = 3600
PAGE_SIZE = 256
KL_THRESHOLD = 0.02
PREFIX_LEN = 2048
MAX_NEW_TOKENS = 64


class _DSV41FlashLinkerServer:
    """The small device pool evicts every turn, so each case must restore
    at least one page through the Mooncake direct linker."""

    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.mooncake.start()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--ep-size",
                "8",
                "--attention-backend",
                "dsv4",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--page-size",
                str(PAGE_SIZE),
                "--chunked-prefill-size",
                "4096",
                "--mem-fraction-static",
                "0.8",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "64",
                "--cuda-graph-max-bs-decode",
                "64",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
                "--hicache-storage-backend-extra-config",
                json.dumps({"enable_group_semantics": True}),
                *cls.extra_args,
            ],
            env=cls.mooncake.server_env(),
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.process)
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    def _record_cache_result(self, result, expected_cached_tokens, label):
        meta_info = result["meta_info"]
        cached_tokens = int(meta_info["cached_tokens"])
        minimum = max(0, expected_cached_tokens - PAGE_SIZE)
        self.assertGreaterEqual(
            cached_tokens,
            minimum,
            f"{label}: expected cached_tokens >= {minimum}, got {cached_tokens}",
        )
        details = meta_info.get("cached_tokens_details") or {}
        remote_tokens = int(details.get("host", 0))
        self._direct_remote_tokens += remote_tokens
        if remote_tokens:
            print(f"{label}: Direct load-back confirmed for {remote_tokens} tokens")

    def _run_linker_case(self, test_case):
        self._direct_remote_tokens = 0
        test_case()
        print(f"Direct load-back total: {self._direct_remote_tokens} tokens")
        self.assertGreater(
            self._direct_remote_tokens,
            0,
            "Expected this case to load KV through the Mooncake Direct Linker",
        )


class TestDSV41FlashLinkerKLPagedSWA(
    _DSV41FlashLinkerServer, UnifiedRadixTreeTestMixin, CustomTestCase
):
    """Target and packed DSpark draft SWA pages restore alongside the shared KV."""

    page_size = PAGE_SIZE
    kl_threshold = KL_THRESHOLD
    sampling_temperature = 0
    max_new_tokens = MAX_NEW_TOKENS
    prefix_len = PREFIX_LEN
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5

    @unittest.skip("Linker CI targets Direct load-back KL accuracy")
    def test_gsm8k(self):
        pass

    @unittest.skip("Linker CI targets Direct load-back KL accuracy")
    def test_mmlu(self):
        pass

    def prefill_cache_assert(self, result, prefix_len, label):
        self._record_cache_result(result, prefix_len, label)

    def decode_cache_assert(self, result, history_len, output_len, label):
        self._record_cache_result(result, history_len + output_len, label)

    def test_multiturn_logprobs_match(self):
        self._run_linker_case(super().test_multiturn_logprobs_match)

    def test_multiturn_prefill_cache_hit_branching(self):
        self._run_linker_case(super().test_multiturn_prefill_cache_hit_branching)

    def test_multiturn_decode_cache_hit_branching(self):
        self._run_linker_case(super().test_multiturn_decode_cache_hit_branching)


class TestDSV41FlashLinkerKLEncoderReplay(_DSV41FlashLinkerServer, CustomTestCase):
    """Encoder replay refuses prompt logprobs, so the rebuilt window is checked
    through decode: a cold run and a run restored from the store must score the
    same greedy continuation."""

    extra_args = [
        "--enable-encoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]

    def test_restored_prefix_decode_logprobs_match(self):
        self._run_linker_case(self._restored_prefix_decode_logprobs_match)

    def _restored_prefix_decode_logprobs_match(self):
        prompts = [ids[:PREFIX_LEN] for ids in self.input_ids[:4]]
        cold = [self._generate(ids) for ids in prompts]
        # Offloads land below the request path; let them finish before flushing.
        time.sleep(15)
        _flush_cache(self.base_url)
        warm = [self._generate(ids) for ids in prompts]

        cold_logprobs, warm_logprobs = [], []
        for i, (before, after) in enumerate(zip(cold, warm)):
            self.assertEqual(before["meta_info"]["cached_tokens"], 0)
            self._record_cache_result(after, PREFIX_LEN, f"restored[{i}]")
            before_tokens = before["meta_info"]["output_token_logprobs"]
            after_tokens = after["meta_info"]["output_token_logprobs"]
            # A wrong window diverges at the first token; late flips are noise.
            agree = 0
            while (
                agree < len(before_tokens)
                and before_tokens[agree][1] == after_tokens[agree][1]
            ):
                agree += 1
            self.assertGreaterEqual(agree, 16, f"restored[{i}] diverged at {agree}")
            cold_logprobs.append([x[0] for x in before_tokens[:agree]])
            warm_logprobs.append([x[0] for x in after_tokens[:agree]])
        compare_kl_divergence(
            warm_logprobs,
            cold_logprobs,
            {self.model: {"kl_div": KL_THRESHOLD}},
            self.model,
            "encoder_replay_restore",
        )

    def _generate(self, input_ids):
        return _generate(self.base_url, input_ids, MAX_NEW_TOKENS, return_logprob=True)


if __name__ == "__main__":
    unittest.main()
