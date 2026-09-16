import unittest

import requests

from sglang.srt.state_capturer.indexer_topk import (
    extract_indexer_topk_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import GLM_5_2_BF16_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=3600, suite="nightly-16-npu-a3", nightly=True)


NUM_INDEXER_LAYERS = 6
INDEX_TOPK = 2048


class TestNPUGLM52ReturnIndexerTopk(CustomTestCase):
    """End-to-end validation of GLM-5.2 NPU indexer-topk responses."""

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            GLM_5_2_BF16_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "16",
                "--dp-size",
                "16",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--enable-dp-lm-head",
                "--disable-shared-experts-fusion",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--enable-return-indexer-topk",
                "--json-model-override-args",
                '{"num_hidden_layers": 6}',
                # The indexer-topk host cache is proportional to this value.
                "--max-total-tokens",
                "1024",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_return_indexer_topk_shape_and_range(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "return_indexer_topk": True,
            },
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("indexer_topk", body["meta_info"])

        topk = extract_indexer_topk_from_meta_info(body).reshape(
            -1, NUM_INDEXER_LAYERS, INDEX_TOPK
        )
        self.assertEqual(topk.ndim, 3)
        seqlen_minus_1, num_layers, topk_size = topk.shape
        self.assertGreater(seqlen_minus_1, 0)
        self.assertEqual(num_layers, NUM_INDEXER_LAYERS)
        self.assertEqual(topk_size, INDEX_TOPK)
        self.assertTrue((topk >= -1).all(), f"min index {topk.min()} < -1")


if __name__ == "__main__":
    unittest.main()
