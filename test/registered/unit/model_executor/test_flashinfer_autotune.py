# Copyright 2023-2026 SGLang Team
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
"""CPU tests for the FlashInfer autotune execution context."""

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_executor.runner.flashinfer_autotune as autotune_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@contextlib.contextmanager
def _fake_autotune(*args, **kwargs):
    del args, kwargs
    yield


class TestFlashInferAutotuneContext(CustomTestCase):
    def test_uses_no_grad_without_creating_inference_tensors(self):
        fake_flashinfer = ModuleType("flashinfer")
        fake_autotuner = ModuleType("flashinfer.autotuner")
        fake_autotuner.autotune = _fake_autotune
        fake_flashinfer.autotuner = fake_autotuner

        current_stream = Mock()
        forward_stream = Mock()
        device_module = SimpleNamespace(stream=lambda stream: contextlib.nullcontext())
        model_runner = SimpleNamespace(device="cuda", forward_stream=forward_stream)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "autotune.json"
            with (
                patch.dict(
                    sys.modules,
                    {
                        "flashinfer": fake_flashinfer,
                        "flashinfer.autotuner": fake_autotuner,
                    },
                ),
                patch.object(
                    autotune_module,
                    "flashinfer_autotune_cache_path",
                    return_value=cache_path,
                ),
                patch.object(
                    autotune_module,
                    "get_flashinfer_autotune_skip_ops",
                    return_value=set(),
                ),
                patch.object(
                    autotune_module.torch.cuda,
                    "current_stream",
                    return_value=current_stream,
                ),
                patch.object(
                    autotune_module.torch,
                    "get_device_module",
                    return_value=device_module,
                ),
            ):
                with autotune_module.flashinfer_autotune_context(
                    model_runner, skip_logits=False
                ):
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertFalse(torch.is_inference_mode_enabled())
                    lazy_buffer = torch.zeros(1)

        # A tensor lazily allocated during autotune must remain mutable in the
        # serving context, which is outside torch.inference_mode by default.
        lazy_buffer.add_(1)
        self.assertEqual(lazy_buffer.item(), 1)
        forward_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(forward_stream)


if __name__ == "__main__":
    unittest.main()
