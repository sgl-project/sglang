"""Qwen3.6-27B GQA MXFP8 KV, local pinned-memory L2, TP2 without MTP.

Set QWEN3_6_MODELSLIM_MXFP8KV_MODEL_PATH to the local ModelSlim MXFP8 KV checkpoint.
GPQA_DATASET_PATH selects the local GPQA Diamond dataset. Every accuracy run
retains a fresh EvalScope report.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from test_modelslim_hicache import ModelSlimHiCacheMixin

from sglang.test.ascend.e2e.test_npu_accuracy_utils import assert_metrics, run_evalscope
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    os.environ.get("QWEN3_6_MODELSLIM_MXFP8KV_MODEL_PATH"),
    "Set QWEN3_6_MODELSLIM_MXFP8KV_MODEL_PATH to a local ModelSlim MXFP8 KV checkpoint",
)
class TestQwenGQAMXFP8HiCache(ModelSlimHiCacheMixin, CustomTestCase):
    model = os.environ.get("QWEN3_6_MODELSLIM_MXFP8KV_MODEL_PATH")
    served_model_name = "qwen3.6-27b-modelslim-mxfp8-kv"
    kv_quant_type = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"
    tp_size = 2
    page_size = 512
    model_args = [
        "--served-model-name",
        served_model_name,
        "--mamba-ssm-dtype",
        "bfloat16",
        "--mamba-track-interval",
        "512",
        "--reasoning-parser",
        "qwen3",
        "--chunked-prefill-size",
        "8192",
        "--max-running-requests",
        "96",
        "--cuda-graph-max-bs-decode",
        "96",
    ]
    small_pool_args = ("--max-mamba-cache-size", "64")
    model_env = {
        "ASCEND_LAUNCH_BLOCKING": "0",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "STREAMS_PER_DEVICE": "32",
    }
    accuracy = 0.878
    datasets = ["gpqa_diamond"]

    def test_gpqa(self):
        output = Path(os.getenv("GPQA_OUTPUT_DIR", "./outputs/gpqa_qwen36_27b_hicache"))
        output.mkdir(parents=True, exist_ok=True)
        for hicache in self.accuracy_modes():
            with (
                self.subTest(hicache=hicache),
                self.server(hicache=hicache, small_pool=False),
            ):
                url = urlsplit(self.base_url)
                work_dir = tempfile.mkdtemp(prefix=f"hicache_{hicache}_", dir=output)
                with patch.dict(os.environ, METRICS_DATA_FILE=work_dir):
                    metrics = run_evalscope(
                        url.hostname,
                        url.port,
                        self.served_model_name,
                        self.datasets,
                        dataset_args={
                            "gpqa_diamond": {
                                "local_path": os.getenv(
                                    "GPQA_DATASET_PATH", "./gpqa_diamond"
                                )
                            }
                        },
                        eval_batch_size=64,
                        limit=198,
                        generation_config={
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "presence_penalty": 0.0,
                            "max_tokens": 81920,
                            "timeout": 3600,
                            "extra_body": {
                                "top_k": 20,
                                "min_p": 0.0,
                                "repetition_penalty": 1.0,
                            },
                        },
                    )
                print(
                    f"HiCache={hicache} GPQA Diamond score={metrics['accuracy']:.4f}, reports={work_dir}"
                )
                self.assertLessEqual(metrics["accuracy"], 1.0)
                assert_metrics(self, metrics)


if __name__ == "__main__":
    unittest.main()
