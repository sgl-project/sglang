"""B300 1P1D coverage for Muse Spark 1.2 NVFP4 with DFlash2."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.muse_spark_utils import try_cached_muse_spark_model
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    _wait_for_gpu_idle_in_ci,
    is_in_ci,
    write_github_step_summary,
)

# ci-cost-override: the 816B checkpoint needs two TP4 replicas; one full GSM8K
# pass covers both NVLink PD transfer and the DFlash2 acceptance regression gate.
register_cuda_ci(est_time=800, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFT_MODEL_PATH = "RadixArk/Muse-Spark-Open-DFlash2"
SERVER_LAUNCH_TIMEOUT = 3600
MOONCAKE_ENV = {
    "MC_INTRANODE_NVLINK": "true",
    "MC_TCP_ENABLE_CONNECTION_POOL": "true",
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "INTRA_NODE_NVLINK",
}


class TestMuseSpark12B300PD(SpecDecodingMixin, PDDisaggregationServerBase):
    gsm8k_accuracy_thres = 0.93
    gsm8k_accept_length_thres = 4.7
    mmmu_pro_score_threshold = 0.8
    mmmu_pro_num_examples = 50
    accept_length_thres = 4.7
    bs_1_speed_thres = 500
    prefill_tp_size = 4
    decode_tp_size = 4
    # CUDA_VISIBLE_DEVICES remaps decode GPUs 4-7 to local ordinals 0-3.
    decode_base_gpu_id = 0
    extra_prefill_env = {**MOONCAKE_ENV, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    extra_decode_env = {**MOONCAKE_ENV, "CUDA_VISIBLE_DEVICES": "4,5,6,7"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_muse_spark_model(MODEL_PATH)
        # A complete local checkpoint needs no gated-model credential.
        if cls.model == MODEL_PATH and (token := os.environ.get("HF_MUSE_SPARK_TOKEN")):
            token_env = patch.dict(os.environ, {"HF_TOKEN": token})
            token_env.start()
            cls.addClassCleanup(token_env.stop)
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        common_args = [
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--ep-size",
            "4",
            "--attention-backend",
            "trtllm_mha",
            "--page-size",
            "128",
            "--mem-fraction-static",
            "0.8",
            "--reasoning-parser",
            "muse",
            "--tool-call-parser",
            "muse",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            try_cached_muse_spark_model(DRAFT_MODEL_PATH, draft=True),
            "--speculative-draft-attention-backend",
            "fa4",
            "--speculative-num-draft-tokens",
            "8",
        ]
        cls.extra_prefill_args = [
            *common_args,
            "--enable-layernorm-sp",
            "--enable-megamoe-for-layernorm-sp",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "200",
        ]
        cls.extra_decode_args = [*common_args, "--max-total-tokens", "4545536"]
        cls.launch_all()

    @classmethod
    def rdma_devices_for(cls, gpu_indices):
        # This recipe transfers within one node over NVLink, not an RDMA NIC.
        return []

    @classmethod
    def wait_server_ready(cls, url, timeout=SERVER_LAUNCH_TIMEOUT, process=None):
        super().wait_server_ready(url, timeout=timeout, process=process)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        _wait_for_gpu_idle_in_ci(timeout=120)

    def flush_cache_for_spec_benchmark(self):
        # The mini-LB does not expose flush_cache. Clear both worker caches.
        for url in (self.prefill_url, self.decode_url):
            requests.post(f"{url}/flush_cache", timeout=60).raise_for_status()

    def _run_accuracy_eval(self, eval_name, num_examples=None):
        self.flush_cache_for_spec_benchmark()

        # Muse Spark has no built-in sgl-eval model preset. Use explicit chat
        # generation settings for both the full GSM8K and the MMMU-Pro smoke test.
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name=eval_name,
                num_examples=num_examples,
                num_threads=64,
                sgl_eval_thinking=True,
                reasoning_effort="xhigh",
                max_tokens=32768,
                temperature=1.0,
                top_p=0.95,
            )
        )
        result = json.loads(Path(metrics["sgl_eval_metrics_path"]).read_text())
        self.assertEqual(result["num_examples"], num_examples or 1319)
        self.assertEqual(result["n_repeats"], 1)
        self.assertEqual(metrics["error_rate"], 0.0)
        return metrics

    def test_mmmu_pro(self):
        metrics = self._run_accuracy_eval("mmmu_pro", self.mmmu_pro_num_examples)
        if is_in_ci():
            write_github_step_summary(
                f"### Muse Spark MMMU-Pro ({self.mmmu_pro_num_examples})\n"
                f"Accuracy: {metrics['score']:.4f}\n"
            )
        self.assertGreaterEqual(metrics["score"], self.mmmu_pro_score_threshold)

    def test_gsm8k(self):
        # Chat evaluation retains all 1,319 test questions.
        metrics = self._run_accuracy_eval("gsm8k")

        # Read the decode endpoint explicitly; missing speculative metrics must
        # fail this test rather than silently bypass the acceptance threshold.
        response = requests.get(f"{self.decode_url}/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        self.assertTrue(states)
        accept_length = states[0]["avg_spec_accept_length"]
        print(
            f"GSM8K: {metrics['score']:.4f}, avg_spec_accept_length={accept_length:.4f}"
        )
        if is_in_ci():
            write_github_step_summary(
                f"### Muse Spark 1.2 NVFP4 1P1D\n"
                f"GSM8K (1319): {metrics['score']:.4f}\n"
                f"avg_spec_accept_length: {accept_length:.4f}\n"
            )
        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)
        self.assertGreaterEqual(accept_length, self.gsm8k_accept_length_thres)


if __name__ == "__main__":
    unittest.main()
