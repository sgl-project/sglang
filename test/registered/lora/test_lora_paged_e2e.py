# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""E2E test for paged LoRA: launch server, send requests, verify responses.

Tests that a server launched with --lora-page-rank-size > 0 (paged mode)
correctly serves LoRA requests and produces non-empty, valid responses.

Usage:
    python -m pytest test/registered/lora/test_lora_paged_e2e.py -v
"""

import os
import unittest


def _patch_kernels_revision():
    """Patch kernels LayerRepository to default revision='main'."""
    try:
        from kernels.layer.func import FuncRepository as _FR
        from kernels.layer.layer import LayerRepository as _LR

        _lr_orig = _LR.__init__

        def _lr_patched(
            self, repo_id, *, layer_name, revision=None, version=None, **kw
        ):
            if revision is None and version is None:
                revision = "main"
            _lr_orig(
                self,
                repo_id,
                layer_name=layer_name,
                revision=revision,
                version=version,
                **kw,
            )

        _LR.__init__ = _lr_patched

        _fr_orig = _FR.__init__

        def _fr_patched(self, repo_id, *, func_name, revision=None, version=None, **kw):
            if revision is None and version is None:
                revision = "main"
            _fr_orig(
                self,
                repo_id,
                func_name=func_name,
                revision=revision,
                version=version,
                **kw,
            )

        _FR.__init__ = _fr_patched
    except ImportError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"patch_kernels failed: {e}")
        pass


_patch_kernels_revision()

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

LORA_PATH = os.environ.get(
    "SGLANG_TEST_LORA_PATH",
    "nvidia/llama-3.1-nemoguard-8b-topic-control",
)
MODEL_PATH = os.environ.get(
    "SGLANG_TEST_MODEL_PATH",
    DEFAULT_MODEL_NAME_FOR_TEST,
)


class TestPagedLoRAE2E(CustomTestCase):
    """End-to-end test: paged LoRA server serves requests correctly."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:13531"
        cls.process = popen_launch_server(
            model=MODEL_PATH,
            base_url=cls.base_url,
            timeout=300,
            other_args=[
                "--enable-lora",
                f"--lora-paths=lora1={LORA_PATH}",
                "--max-lora-rank",
                "8",
                "--max-loras-per-batch",
                "4",
                "--max-loaded-loras",
                "8",
                "--lora-target-modules",
                "all",
                "--lora-page-rank-size",
                "8",
                "--lora-pages",
                "12",
                "--context-length",
                "1024",
                "--mem-fraction-static",
                "0.85",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text, lora_path="lora1", max_new_tokens=32):
        url = self.base_url + "/generate"
        response = requests.post(
            url,
            json={
                "text": text,
                "lora_path": lora_path,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
        )
        return response

    def test_basic_lora_request(self):
        """Single LoRA request returns valid response."""
        response = self._generate("Hello")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 0)

    def test_lora_output_nonempty(self):
        """LoRA output is non-empty and contains meaningful text."""
        response = self._generate("Tell me about AI", max_new_tokens=64)
        self.assertEqual(response.status_code, 200)
        text = response.json().get("text", "")
        self.assertGreater(len(text), 5)

    def test_multiple_requests(self):
        """Multiple sequential requests all succeed."""
        prompts = ["Hello", "What is 2+2?", "Write a short poem"]
        for prompt in prompts:
            response = self._generate(prompt, max_new_tokens=16)
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.json().get("text", "")), 0)

    def test_base_model_request(self):
        """Request without LoRA (base model) also works on paged server."""
        url = self.base_url + "/generate"
        response = requests.post(
            url,
            json={
                "text": "Hello",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json().get("text", "")), 0)


if __name__ == "__main__":
    unittest.main()
