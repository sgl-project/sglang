"""Muse Spark NVFP4 + DFlash on 8x H200 with the Hopper kernels (the documented launch)."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.test.ci.ci_register import register_cuda_ci

# Aliased so pytest does not collect the imported `test_`-prefixed helpers as tests.
from sglang.test.kl_test_utils import (
    _flush_cache,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_logprobs_match_decode_cache_hit,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_prefill_cache_hit_helper as assert_logprobs_match_prefill_cache_hit,
)
from sglang.test.muse_spark_utils import try_cached_muse_spark_model
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    _wait_for_gpu_idle_in_ci,
    is_in_ci,
    write_github_step_summary,
)

register_cuda_ci(est_time=1500, stage="nightly", runner_config="8-gpu-h200")

MODEL_PATH = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFT_MODEL_PATH = "RadixArk/Muse-Spark-Open-DFlash2"
# Decode logprobs come from the DFlash verify forward, input logprobs from a full prefill
# of the same tokens: same precision but different fused kernels and accumulation order,
# so not bitwise. 1e-2 as the other KL tests that replay verify output through prefill.
KL_THRESHOLD = 1e-2
KL_SAMPLES = 16
KL_MAX_NEW_TOKENS = 256


class TestMuseSparkH200(DefaultServerBase):
    timeout = 3600
    gsm8k_accuracy_thres = 0.93
    # Sampled GSM8K takes chain verify, as on the B300 tests.
    gsm8k_accept_length_thres = 4.7
    server_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_muse_spark_model(MODEL_PATH)
        # A complete local checkpoint needs no gated-model credential.
        if cls.model == MODEL_PATH and (token := os.environ.get("HF_MUSE_SPARK_TOKEN")):
            token_env = patch.dict(os.environ, {"HF_TOKEN": token})
            token_env.start()
            cls.addClassCleanup(token_env.stop)
        # docs/docs/advanced_features/muse_spark_hopper.mdx, plus the reasoning parser for GSM8K.
        cls.other_args = (
            "--speculative-algorithm DFLASH --speculative-num-draft-tokens 8 "
            "--speculative-draft-attention-backend fa4 --tp-size 8 --ep-size 8 "
            "--quantization modelopt_fp4 --moe-runner-backend marlin --mem-fraction-static 0.85 "
            "--context-length 131072 --max-running-requests 32 --cuda-graph-max-bs-decode 32 "
            "--enable-hierarchical-cache --hicache-size 120 --chunked-prefill-size 32768 "
            "--max-prefill-tokens 32768 --enable-mixed-chunk --schedule-policy shortest-prefill-first "
            "--swa-full-tokens-ratio 0.09 --disable-cuda-graph-padding --enable-prefill-delayer "
            "--prefill-delayer-queue-min-ratio 0.2 --prefill-delayer-max-delay-ms 100 "
            "--prefill-delayer-max-delay-passes 6 --prefill-max-requests 32 "
            "--warmups concurrent_prefill --reasoning-parser muse"
        ).split() + [
            "--speculative-draft-model-path",
            try_cached_muse_spark_model(DRAFT_MODEL_PATH, draft=True),
        ]
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        _wait_for_gpu_idle_in_ci(timeout=120)

    def test_gsm8k(self):
        # Flushing also resets the speculative counters: the accept length covers GSM8K only.
        _flush_cache(self.base_url)
        # The model's recommended chat settings, as in the B300 tests.
        gen = dict(sgl_eval_thinking=True, reasoning_effort="xhigh", max_tokens=32768)
        gen.update(temperature=1.0, top_p=0.95, num_examples=200, num_threads=64)
        gen.update(base_url=self.base_url, model=self.model, eval_name="gsm8k")
        metrics = run_sgl_eval(SimpleNamespace(**gen))
        result = json.loads(Path(metrics["sgl_eval_metrics_path"]).read_text())
        self.assertEqual(result["num_examples"], 200)
        self.assertEqual(metrics["error_rate"], 0.0)

        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        accept_length = response.json()["internal_states"][0]["avg_spec_accept_length"]
        summary = (
            f"GSM8K (200): {metrics['score']:.4f}, accept length {accept_length:.3f}"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(f"### Muse Spark NVFP4 8x H200\n{summary}\n")
        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)
        self.assertGreaterEqual(accept_length, self.gsm8k_accept_length_thres)

    def _match(self, helper):
        helper(
            self.base_url,
            {self.model: {"kl_div": KL_THRESHOLD}},
            self.model,
            max_samples=KL_SAMPLES,
            max_new_tokens=KL_MAX_NEW_TOKENS,
        )

    def test_input_output_logprobs_match(self):
        self._match(assert_logprobs_match)

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        self._match(assert_logprobs_match_prefill_cache_hit)

    def test_input_output_logprobs_match_decode_cache_hit(self):
        self._match(assert_logprobs_match_decode_cache_hit)


if __name__ == "__main__":
    unittest.main()
