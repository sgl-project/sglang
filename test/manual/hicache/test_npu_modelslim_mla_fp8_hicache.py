"""DeepSeek-V3.1-Terminus MLA FP8 KV on TP8 without MTP, using ACL-registered host L2.

Set DS_V3_MODELSLIM_FP8KV_MODEL_PATH to the local ModelSlim FP8 KV checkpoint.
GSM8K_DATA_PATH optionally selects a local JSONL dataset.
"""

import os
import unittest
from types import SimpleNamespace

from test_modelslim_hicache import ModelSlimHiCacheMixin

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    os.environ.get("DS_V3_MODELSLIM_FP8KV_MODEL_PATH"),
    "Set DS_V3_MODELSLIM_FP8KV_MODEL_PATH to a local ModelSlim FP8 KV checkpoint",
)
class TestDeepSeekMLAFP8HiCache(ModelSlimHiCacheMixin, CustomTestCase):
    model = os.environ.get("DS_V3_MODELSLIM_FP8KV_MODEL_PATH")
    kv_quant_type = "Q_FP8_DYNAMIC_KV_FP8"
    tp_size = 8
    page_size = 128
    model_args = [
        "--dp-size",
        "1",
        "--ep-size",
        "1",
        "--moe-a2a-backend",
        "none",
        "--max-running-requests",
        "16",
        "--cuda-graph-max-bs-decode",
        "16",
        "--chunked-prefill-size",
        "1024",
    ]
    model_env = {
        "SGLANG_USE_FIA_NZ": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_HICACHE_HOST_MEM_BACKEND": "acl",
    }

    def test_gsm8k(self):
        for hicache in self.accuracy_modes():
            with (
                self.subTest(hicache=hicache),
                self.server(hicache=hicache, small_pool=False),
            ):
                metrics = run_eval(
                    SimpleNamespace(
                        base_url=self.base_url,
                        model=self.model,
                        eval_name="gsm8k",
                        api="completion",
                        num_examples=200,
                        num_threads=128,
                        num_shots=5,
                        max_tokens=512,
                        gsm8k_data_path=os.getenv("GSM8K_DATA_PATH"),
                    )
                )
                print(f"HiCache={hicache} GSM8K score={metrics['score']:.4f}")
                self.assertGreaterEqual(metrics["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
