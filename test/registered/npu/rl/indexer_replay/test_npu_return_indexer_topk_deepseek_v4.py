import asyncio
import os
import subprocess
import unittest

import aiohttp
import numpy as np

from sglang.srt.state_capturer.indexer_topk import (
    extract_indexer_topk_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V4_FLASH_BF16_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)
from sglang.utils import wait_for_server

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="base-b-test-2-npu-a3")


# Keep the first 6 layers of the BF16 checkpoint. The first 6 V4-Flash
# layers have compress_ratios [0, 0, 4, 128, 4, 128], hence two C4 indexer layers.
NUM_INDEXER_LAYERS = 2
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


@unittest.skipUnless(
    os.path.isfile(os.path.join(DEEPSEEK_V4_FLASH_BF16_MODEL_PATH, "config.json")),
    f"DeepSeek-V4-Flash model config not found at {DEEPSEEK_V4_FLASH_BF16_MODEL_PATH}",
)
class TestNPUDeepSeekV4ReturnIndexerTopk(CustomTestCase):
    """End-to-end validation of DeepSeek-V4 C4 NPU indexer-topk responses."""

    @classmethod
    def setUpClass(cls):
        cls.sampling_args = {"temperature": 0, "max_new_tokens": 16}
        cls.texts = [
            "What is the capital of France?",
            "Solve: 2 + 3 = ?",
        ]
        env = os.environ.copy()
        env.update(DEEPSEEK_V4_BF16_ENVS)
        _, host, port = DEFAULT_URL_FOR_TEST.split(":")
        server_command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            DEEPSEEK_V4_FLASH_BF16_MODEL_PATH,
            "--page-size",
            "128",
            "--device",
            "npu",
            "--attention-backend",
            "dsv4",
            "--load-format",
            "dummy",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.75",
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--kv-cache-dtype",
            "auto",
            "--disable-cuda-graph",
            "--enable-return-indexer-topk",
            "--json-model-override-args",
            '{"num_hidden_layers": 6, "compress_ratios": [0, 0, 4, 128, 4, 128]}',
            "--host",
            host[2:],
            "--port",
            port,
        ]
        command = [
            "bash",
            "-c",
            "source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash || true; "
            "source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash || true; "
            'exec "$@"',
            "bash",
            *server_command,
        ]
        cls.process = subprocess.Popen(command, env=env)
        wait_for_server(
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process,
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
