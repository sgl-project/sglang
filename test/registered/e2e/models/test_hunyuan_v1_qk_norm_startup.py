"""Startup smoke test for HunYuan v1 (``use_qk_norm``) under piecewise CUDA graph.

Guards the regression fixed by restoring the q/k shape after the per-head
qk_norm in ``models/hunyuan.py``. Since piecewise CUDA graph became the default
(v0.5.10), booting any HunYuan v1 checkpoint with ``use_qk_norm: true`` crashed
during warmup at ``o_proj`` (``mat1 and mat2 shapes cannot be multiplied``) on
every attention backend. Launching ``tencent/Hy-MT2-1.8B`` with default flags
(piecewise CUDA graph on) must reach a serving state.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

_MODEL_PATH = "tencent/Hy-MT2-1.8B"


class TestHunYuanV1QKNormStartup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # No extra flags: piecewise CUDA graph is on by default, and its warmup
        # is the path that routes the qk_norm shape bug into o_proj.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate(self):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["text"])


if __name__ == "__main__":
    unittest.main()
