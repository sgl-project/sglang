import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import _random_suffixes
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    compare_kl_divergence,
    get_input_ids,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import unified_radix_tree_server_env

register_cuda_ci(est_time=150, stage="extra-b", runner_config="4-gpu-h100")

MODEL = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")
KL_DIV_THRESHOLD = 1e-9
TRACK_INTERVAL = 128

SERVER_ARGS = [
    "--trust-remote-code",
    "--revision",
    REVISION,
    "--attention-backend",
    "fa4",
    "--page-size",
    "128",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mamba-track-interval",
    str(TRACK_INTERVAL),
    "--enable-deterministic-inference",
    "--mem-fraction-static",
    "0.6",
    "--enable-cache-report",
]


class TestDisaggregationDecodeRadixCacheInkling(PDDisaggregationServerBase):
    extra_prefill_args = SERVER_ARGS
    extra_decode_args = [*SERVER_ARGS, "--disaggregation-decode-enable-radix-cache"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MODEL
        env = unified_radix_tree_server_env("python")
        cls.extra_prefill_env = env
        cls.extra_decode_env = env
        cls.launch_all()
        cls.input_ids = get_input_ids(
            tokenizer_path=MODEL, num_samples=2, trust_remote_code=True
        )

    def _decode(self, ids, flush_urls):
        for url in flush_urls:
            _flush_cache(url)
        (result,) = _generate(self.base_url, [ids], 512, return_logprob=True)
        return result

    def test_decode_radix_hit_matches_cold_transfer(self):
        histories = []
        for ids in self.input_ids:
            prompt = list(ids[:1024])
            histories.append(
                prompt + _generate(self.base_url, [prompt], 256)[0]["output_ids"]
            )
        requests = [
            history + suffix
            for history, suffix in zip(
                histories, _random_suffixes(len(histories), 512, seed=100)
            )
        ]

        hit = [self._decode(ids, [self.prefill_url]) for ids in requests]
        cold = [
            self._decode(ids, [self.prefill_url, self.decode_url]) for ids in requests
        ]

        for history, hit_result, cold_result in zip(histories, hit, cold):
            self.assertGreaterEqual(
                hit_result["meta_info"]["cached_tokens"],
                (len(history) - 1) // TRACK_INTERVAL * TRACK_INTERVAL,
            )
            self.assertEqual(cold_result["meta_info"]["cached_tokens"], 0)
            self.assertEqual(hit_result["output_ids"], cold_result["output_ids"])
        compare_kl_divergence(
            [_extract_output_logprobs(r) for r in cold],
            [_extract_output_logprobs(r) for r in hit],
            {MODEL: {"kl_div": KL_DIV_THRESHOLD}},
            MODEL,
            "decode_radix_hit_vs_cold_transfer",
        )


if __name__ == "__main__":
    unittest.main()
