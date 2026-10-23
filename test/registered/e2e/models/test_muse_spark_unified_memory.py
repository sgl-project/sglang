"""Muse Spark unified KV + HiCache coverage with TRT-LLM and fused DFlash2."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.muse_spark_utils import try_cached_muse_spark_model
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    _wait_for_gpu_idle_in_ci,
    is_in_ci,
    write_github_step_summary,
)

register_cuda_ci(est_time=800, stage="extra-b", runner_config="8-gpu-b300")

MODEL_PATH = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFT_MODEL_PATH = "RadixArk/Muse-Spark-Open-DFlash2"


class TestMuseSparkUnifiedMemory(DefaultServerBase):
    timeout = 3600
    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_muse_spark_model(MODEL_PATH)
        # A complete local checkpoint needs no gated-model credential.
        if cls.model == MODEL_PATH and (token := os.environ.get("HF_MUSE_SPARK_TOKEN")):
            token_env = patch.dict(os.environ, {"HF_TOKEN": token})
            token_env.start()
            cls.addClassCleanup(token_env.stop)
        cls.other_args = [
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--attention-backend",
            "trtllm_mha",
            "--page-size",
            "128",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "128",
            "--enable-layernorm-sp",
            "--enable-megamoe-for-layernorm-sp",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            try_cached_muse_spark_model(DRAFT_MODEL_PATH, draft=True),
            "--speculative-draft-attention-backend",
            "fa4",
            "--speculative-num-draft-tokens",
            "8",
            "--enable-unified-memory",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "200",
            "--mm-feature-transport",
            "cpu",
            "--reasoning-parser",
            "muse",
            "--tool-call-parser",
            "muse",
        ]
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        _wait_for_gpu_idle_in_ci(timeout=120)

    def test_gsm8k(self):
        # Exercise decode/verify writes into the shared GQA/SWA pool with all
        # 1,319 questions, including the FA4 draft's fused BF16 KV region.
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
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
        if is_in_ci():
            write_github_step_summary(
                f"### Muse Spark NVFP4 unified memory (TP4/EP4)\n"
                f"GSM8K (1319): {metrics['score']:.4f}\n"
            )
        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)


if __name__ == "__main__":
    unittest.main()
