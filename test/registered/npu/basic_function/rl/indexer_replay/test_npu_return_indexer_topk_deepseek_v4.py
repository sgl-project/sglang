import asyncio
import unittest

import aiohttp
import numpy as np

from sglang.srt.state_capturer.indexer_topk import (
    extract_indexer_topk_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_BF16_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=3600, suite="nightly-4-npu-a3", nightly=True)


# Keep the first 8 layers of the BF16 checkpoint.  The first 8 V4-Flash
# layers have compress_ratios [0, 0, 4, 128, 4, 128, 4, 128], hence three
# C4 indexer layers.
NUM_INDEXER_LAYERS = 3
INDEX_TOPK = 512

DEEPSEEK_V4_BF16_ENVS = {
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
}


class TestNPUDeepSeekV4ReturnIndexerTopk(CustomTestCase):
    """End-to-end validation of DeepSeek-V4 C4 NPU indexer-topk responses."""

    @classmethod
    def setUpClass(cls):
        cls.sampling_args = {"temperature": 0, "max_new_tokens": 16}
        cls.texts = [
            "What is the capital of France?",
            "Solve: 2 + 3 = ?",
        ]
        cls.process = popen_launch_server(
            DEEPSEEK_V4_FLASH_BF16_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--page-size",
                "128",
                "--device",
                "npu",
                "--attention-backend",
                "dsv4",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.6",
                "--tp-size",
                "4",
                "--dp-size",
                "4",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--kv-cache-dtype",
                "auto",
                "--disable-cuda-graph",
                "--enable-return-indexer-topk",
                "--json-model-override-args",
                '{"num_hidden_layers": 8, "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128]}',
            ],
            env=DEEPSEEK_V4_BF16_ENVS,
        )
        try:
            cls.captured = asyncio.run(cls._collect_async())
        except Exception:
            kill_process_tree(cls.process.pid)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_return_indexer_topk_shape_and_range(self):
        for topk in self.captured:
            self._check_shape_and_range(topk)

    def _check_shape_and_range(self, topk: np.ndarray):
        self.assertEqual(topk.ndim, 3)
        seqlen_minus_1, num_layers, topk_size = topk.shape
        self.assertGreater(seqlen_minus_1, 0)
        self.assertEqual(num_layers, NUM_INDEXER_LAYERS)
        self.assertEqual(topk_size, INDEX_TOPK)
        self.assertTrue((topk >= -1).all(), f"min index {topk.min()} < -1")

    @classmethod
    async def _collect_async(cls):
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    make_request(
                        session,
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        {
                            "text": text,
                            "sampling_params": cls.sampling_args,
                            "return_indexer_topk": True,
                        },
                    )
                )
                for text in cls.texts
            ]
            http_results = await asyncio.gather(*tasks)
            return [
                extract_indexer_topk_from_meta_info(res).reshape(
                    -1, NUM_INDEXER_LAYERS, INDEX_TOPK
                )
                for res in http_results
            ]


async def make_request(session, url, payload):
    async with session.post(url=url, json=payload) as response:
        return await response.json()


if __name__ == "__main__":
    unittest.main()
