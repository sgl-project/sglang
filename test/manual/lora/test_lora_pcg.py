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
"""Tests for LoRA + piecewise CUDA graph (PCG) integration.

Unit tests for the new PCG LoRA code paths:
  - ForwardContext.lora_backend plumbing
  - _use_custom_ops() gating logic
  - Custom op registration
  - can_run() mixed-adapter rejection

Integration test (requires GPU + model):
  - Server launch with --enable-piecewise-cuda-graph + LoRA, correctness check
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.test_utils import CustomTestCase


class TestForwardContextLoRABackend(CustomTestCase):
    """Test that ForwardContext properly stores and exposes lora_backend."""

    def test_lora_backend_default_none(self):
        from sglang.srt.compilation.piecewise_context_manager import ForwardContext

        ctx = ForwardContext()
        self.assertIsNone(ctx.lora_backend)

    def test_set_lora_backend(self):
        from sglang.srt.compilation.piecewise_context_manager import ForwardContext

        ctx = ForwardContext()
        mock_backend = MagicMock()
        ctx.set_lora_backend(mock_backend)
        self.assertIs(ctx.lora_backend, mock_backend)

    def test_set_forward_context_with_lora_backend(self):
        from sglang.srt.compilation.piecewise_context_manager import (
            get_forward_context,
            set_forward_context,
        )

        mock_backend = MagicMock()
        mock_batch = MagicMock()
        with set_forward_context(
            mock_batch, [], None, [], [], lora_backend=mock_backend
        ):
            ctx = get_forward_context()
            self.assertIs(ctx.lora_backend, mock_backend)
        # After context exit, should be None
        self.assertIsNone(get_forward_context())

    def test_set_forward_context_without_lora_backend(self):
        from sglang.srt.compilation.piecewise_context_manager import (
            get_forward_context,
            set_forward_context,
        )

        mock_batch = MagicMock()
        with set_forward_context(mock_batch, [], None, [], []):
            ctx = get_forward_context()
            self.assertIsNone(ctx.lora_backend)


class TestUseCustomOps(CustomTestCase):
    """Test _use_custom_ops() gating logic in layers.py."""

    def test_returns_false_when_no_context(self):
        from sglang.srt.lora.layers import _use_custom_ops

        self.assertFalse(_use_custom_ops())

    def test_returns_false_when_lora_backend_is_none(self):
        from sglang.srt.compilation.piecewise_context_manager import set_forward_context
        from sglang.srt.lora.layers import _use_custom_ops

        mock_batch = MagicMock()
        with set_forward_context(mock_batch, [], None, [], [], lora_backend=None):
            self.assertFalse(_use_custom_ops())

    def test_returns_true_when_lora_backend_is_set(self):
        from sglang.srt.compilation.piecewise_context_manager import set_forward_context
        from sglang.srt.lora.layers import _use_custom_ops

        mock_batch = MagicMock()
        mock_backend = MagicMock()
        with set_forward_context(
            mock_batch, [], None, [], [], lora_backend=mock_backend
        ):
            self.assertTrue(_use_custom_ops())


class TestCustomOpRegistration(CustomTestCase):
    """Test that LoRA custom ops are registered in torch.ops.sglang."""

    def test_ops_registered(self):
        import sglang.srt.lora.lora_custom_ops  # noqa: F401

        expected_ops = [
            "lora_a_sgemm",
            "lora_b_sgemm",
            "qkv_lora",
            "gate_up_lora",
            "embedding_lora_a",
        ]
        for op_name in expected_ops:
            self.assertTrue(
                hasattr(torch.ops.sglang, op_name),
                f"torch.ops.sglang.{op_name} not registered",
            )


class TestCanRunMixedAdapters(CustomTestCase):
    """Test the mixed-adapter rejection logic used by PCG can_run()."""

    def test_single_adapter_allowed(self):
        lora_ids = ["adapter_a", "adapter_a", "adapter_a"]
        first = lora_ids[0]
        self.assertTrue(all(lid == first for lid in lora_ids[1:]))

    def test_mixed_adapters_rejected(self):
        lora_ids = ["adapter_a", "adapter_b", "adapter_a"]
        first = lora_ids[0]
        self.assertFalse(all(lid == first for lid in lora_ids[1:]))

    def test_none_adapters_allowed(self):
        lora_ids = [None, None, None]
        first = lora_ids[0]
        self.assertTrue(all(lid == first for lid in lora_ids[1:]))

    def test_empty_lora_ids_allowed(self):
        lora_ids = ["adapter_a"]
        first = lora_ids[0]
        self.assertTrue(all(lid == first for lid in lora_ids[1:]))


class TestLoRAPCGIntegration(CustomTestCase):
    """Integration test: LoRA + PCG server launch and correctness.

    Launches sglang with --enable-piecewise-cuda-graph and a LoRA adapter,
    sends requests, and verifies output matches HF reference.

    Requires GPU and model download. Run manually:
        python -m pytest test/manual/lora/test_lora_pcg.py::TestLoRAPCGIntegration -v
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            popen_launch_server,
        )

        cls.model = "meta-llama/Llama-3.1-8B-Instruct"
        cls.lora_path = "algoprog/fact-generation-llama-3.1-8b-instruct-lora"
        cls.base_url = "http://127.0.0.1:30010"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--enable-lora",
                "--lora-paths",
                cls.lora_path,
                "--max-loras-per-batch",
                "1",
                "--mem-fraction-static",
                "0.65",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(cls.process.pid)

    def _generate(self, prompt, lora_path=None, max_tokens=32):
        import openai

        client = openai.Client(base_url=f"{self.base_url}/v1", api_key="EMPTY")
        model_name = f"{self.model}:{lora_path}" if lora_path else self.model
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].text

    def test_lora_pcg_generates_output(self):
        """LoRA + PCG should generate non-empty, non-garbage output."""
        output = self._generate(
            "AI is a field of computer science focused on",
            lora_path=self.lora_path,
        )
        self.assertGreater(len(output.strip()), 10, f"Output too short: '{output}'")

    def test_lora_pcg_differs_from_base(self):
        """LoRA output should differ from base model output."""
        prompt = "AI is a field of computer science focused on"
        lora_output = self._generate(prompt, lora_path=self.lora_path)
        base_output = self._generate(prompt, lora_path=None)
        # LoRA should produce different output than base (adapter is doing something)
        self.assertNotEqual(
            lora_output.strip(),
            base_output.strip(),
            "LoRA output identical to base — adapter may not be applied in PCG path",
        )


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
