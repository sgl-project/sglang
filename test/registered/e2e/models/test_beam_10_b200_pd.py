"""B200 1P1D coverage for Beam 1.0 NVFP4 with DFlash2."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    _wait_for_gpu_idle_in_ci,
    is_in_ci,
    try_cached_model,
    write_github_step_summary,
)

# The 501B checkpoint needs two TP4 replicas; one full GSM8K pass covers both
# NVLink PD transfer and the DFlash2 acceptance regression gate.
register_cuda_ci(est_time=800, stage="nightly", runner_config="8-gpu-b200")

MODEL_PATH = "reflection/Beam-1.0-501B-A23B-NVFP4"
DRAFT_MODEL_PATH = "reflection/Beam-1.0-501B-A23B-DFlash2"
SERVER_LAUNCH_TIMEOUT = 3600
MOONCAKE_ENV = {
    "MC_INTRANODE_NVLINK": "true",
    "MC_TCP_ENABLE_CONNECTION_POOL": "true",
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "INTRA_NODE_NVLINK",
}


class TestBeam10B200PD(SpecDecodingMixin, PDDisaggregationServerBase):
    gsm8k_accuracy_thres = 0.94
    gsm8k_accept_length_thres = 4.0
    accept_length_thres = 4.4
    bs_1_speed_thres = 600
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
        # PD rejects the cookbook's --enable-unified-memory together with speculative
        # decoding over Mooncake, so it is left out.
        cls.extra_prefill_args = cls.extra_decode_args = [
            "--attention-backend",
            "trtllm_mha",
            "--page-size",
            "64",
            "--context-length",
            "1048576",
            "--mem-fraction-static",
            "0.85",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            try_cached_model(DRAFT_MODEL_PATH),
            "--speculative-dflash-block-size",
            "8",
            "--tool-call-parser",
            "beam",
            "--reasoning-parser",
            "beam",
        ]
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

    def test_gsm8k(self):
        self.flush_cache_for_spec_benchmark()

        # Beam has no built-in sgl-eval model preset. Use its generation_config.json
        # sampling; the chat template's default reasoning effort applies.
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_threads=64,
                max_tokens=32768,
                temperature=0.7,
                top_p=0.99,
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
                f"### Beam 1.0 NVFP4 1P1D\n"
                f"GSM8K (1319): {metrics['score']:.4f}\n"
                f"avg_spec_accept_length: {accept_length:.4f}\n"
            )
        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)
        self.assertGreaterEqual(accept_length, self.gsm8k_accept_length_thres)


if __name__ == "__main__":
    unittest.main()
