# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import mxfp_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")

MODEL = "nvidia/DeepSeek-R1-0528-NVFP4"
SERVER_LAUNCH_TIMEOUT = 1800


class TestDeepSeekR10528NVFP4(CustomTestCase):
    """NVFP4->MXFP4 online requantization for DeepSeek-R1-0528-NVFP4.

    Exercises the MLA attention path (attention_backend=aiter), multi-threaded
    weight loading (enable_multithread_load=true), and the NVFP4 MoE
    requantization path with TP=8 on MI35X.
    """

    @classmethod
    def setUpClass(cls):
        if not mxfp_supported():
            raise unittest.SkipTest(
                "online MXFP4 quantization requires an AMD ROCm device with "
                "FP4 hardware support (gfx95x, e.g. MI355x)"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--quantization",
                "quark_mxfp4",
                "--tensor-parallel-size",
                "8",
                "--attention-backend",
                "aiter",
                "--chunked-prefill-size",
                "8192",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # DeepSeek-R1-0528 NVFP4->MXFP4 observed accuracy: ~0.95
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=500,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
