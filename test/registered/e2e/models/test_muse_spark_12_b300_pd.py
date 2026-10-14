"""B300 1P1D coverage for Muse Spark 1.2 NVFP4 with DFlash2."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    _wait_for_gpu_idle_in_ci,
    is_in_ci,
    try_cached_model,
    write_github_step_summary,
)

# ci-cost-override: the 816B checkpoint needs two TP4 replicas; one full GSM8K
# pass covers both NVLink PD transfer and the DFlash2 acceptance regression gate.
register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = "meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open"
DRAFT_MODEL_PATH = "RadixArk/Muse-Spark-1.2-DFlash2-preview"
SERVER_LAUNCH_TIMEOUT = 3600
MOONCAKE_ENV = {
    "MC_INTRANODE_NVLINK": "true",
    "MC_TCP_ENABLE_CONNECTION_POOL": "true",
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "INTRA_NODE_NVLINK",
}


class TestMuseSpark12B300PD(PDDisaggregationServerBase):
    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 3.5
    prefill_tp_size = 4
    decode_tp_size = 4
    # CUDA_VISIBLE_DEVICES remaps decode GPUs 4-7 to local ordinals 0-3.
    decode_base_gpu_id = 0
    extra_prefill_env = {**MOONCAKE_ENV, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    extra_decode_env = {**MOONCAKE_ENV, "CUDA_VISIBLE_DEVICES": "4,5,6,7"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(MODEL_PATH)
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        common_args = [
            # FlashInfer 0.6.18 autotuning uses unresized per-token FP4 scales.
            "--disable-flashinfer-autotune",
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
            try_cached_model(DRAFT_MODEL_PATH),
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

    def test_gsm8k(self):
        for url in (self.prefill_url, self.decode_url):
            requests.post(f"{url}/flush_cache", timeout=60).raise_for_status()

        # The completion-style few-shot evaluator removes its prompt examples
        # from the test split. Chat evaluation retains all 1,319 test questions.
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="sgl_eval",
                num_examples=None,
                num_threads=64,
                sgl_eval_thinking=True,
                reasoning_effort="xhigh",
                max_tokens=32768,
                temperature=1.0,
                top_p=0.95,
            )
        )
        result = json.loads(Path(metrics["sgl_eval_metrics_path"]).read_text())
        self.assertEqual(result["num_examples"], 1319)
        self.assertEqual(result["n_repeats"], 1)
        self.assertEqual(metrics["error_rate"], 0.0)

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
        self.assertGreaterEqual(accept_length, self.accept_length_thres)


if __name__ == "__main__":
    unittest.main()
