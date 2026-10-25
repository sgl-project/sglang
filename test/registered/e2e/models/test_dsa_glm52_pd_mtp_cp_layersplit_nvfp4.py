"""Full GSM8K acceptance for NVFP4 KV cache with PD, CP, and layer split."""

import os
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import (
    GSM8KMixin,
    _run_accuracy_eval,
    prepare_full_gsm8k_data,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-b300")


class TestGLM52NVFP4KVLayerSplit(PDDisaggregationServerBase, GSM8KMixin):
    model = os.environ.get("SGLANG_TEST_GLM52_MODEL", "nvidia/GLM-5.2-NVFP4")
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4
    capture_per_side_logs = True
    extra_prefill_env = {"SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0"}
    extra_decode_env = extra_prefill_env.copy()
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 200
    gsm8k_num_shots = 20

    _common_args = [
        "--skip-server-warmup",
        "--attention-backend",
        "dsa",
        "--kv-cache-dtype",
        "nvfp4",
        "--page-size",
        "64",
        "--dsa-prefill-backend",
        "flashmla_sparse",
        "--dsa-decode-backend",
        "flashmla_kv",
        "--mem-fraction-static",
        "0.85",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]
    extra_prefill_args = _common_args + [
        "--dp-size",
        "1",
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--attn-cp-size",
        "4",
        "--enable-dsa-cache-layer-split",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
    ]
    extra_decode_args = _common_args + [
        "--dp-size",
        "4",
        "--enable-dp-attention",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Layer split requires Mooncake even outside CI. Preserve any RDMA
        # device settings selected by the common fixture.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()

    def test_gsm8k(self):
        with tempfile.TemporaryDirectory(prefix="glm52-nvfp4-gsm8k-") as data_dir:
            dataset = prepare_full_gsm8k_data(Path(data_dir) / "full_gsm8k.jsonl")
            for url in (self.prefill_url, self.decode_url):
                requests.post(
                    url + "/flush_cache", params={"timeout": 60}, timeout=65
                ).raise_for_status()
            _run_accuracy_eval(
                self,
                eval_name="gsm8k",
                score_threshold=self.gsm8k_accuracy_thres,
                num_examples=self.gsm8k_num_questions,
                num_threads=self.gsm8k_num_threads,
                api="completion",
                max_tokens=512,
                num_shots=self.gsm8k_num_shots,
                temperature=0,
                gsm8k_data_path=dataset,
            )


if __name__ == "__main__":
    unittest.main()
