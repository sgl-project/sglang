"""Startup + serving smoke test for HunYuan v1 (``use_qk_norm``) under a prefill CUDA graph.

Guards the regression fixed by restoring the q/k shape after the per-head qk_norm
in ``models/hunyuan.py``. Booting any HunYuan v1 checkpoint with ``use_qk_norm:
true`` crashed during the prefill CUDA graph warmup at ``o_proj`` (``mat1 and mat2
shapes cannot be multiplied``) on every attention backend, because that path
allocates ``torch.empty_like(q)`` in ``RadixAttention`` and so sees the broken
per-head shape. ``tencent/Hy-MT2-1.8B`` is launched with default flags and must
serve several requests without the layout regression re-appearing.
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
        # No extra flags: the default prefill CUDA graph warmup is the path that
        # routes the qk_norm shape bug into o_proj.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text, max_new_tokens):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["text"]

    def test_serves_multiple_requests(self):
        # Greedy output starts with "Paris"; the layout regression corrupts it,
        # so this catches a silent shape bug and not just a hard crash.
        self.assertIn("Paris", self._generate("The capital of France is", 8))
        # A longer prompt then a repeat of the first request: on the broken build
        # the server boots and answers the first request but dies on a later one.
        self._generate("Explain the causes of the First World War. " * 40, 32)
        self.assertIn("Paris", self._generate("The capital of France is", 8))
        # Scheduler must still be alive after the sequence.
        health = requests.get(f"{self.base_url}/health_generate", timeout=120)
        self.assertEqual(health.status_code, 200)


if __name__ == "__main__":
    unittest.main()
