"""Unit tests for FlashInferQuestDecodeBackend (Mode 2 — quest-only, no hisparse).

Smoke-test only: verifies the module imports, the class is registered in
ATTENTION_BACKENDS under ``flashinfer_quest``, and inherits the right base.

End-to-end behavior (constructing against a real ``ModelRunner`` and running
a forward) is exercised in ``test/registered/quest/test_quest_only_e2e.py``.
"""

import unittest

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "Quest backend tests require CUDA/ROCm.",
)
class TestFlashInferQuestBackendRegistration(unittest.TestCase):
    """Verify the backend registers + has the right shape, no construction."""

    def test_backend_class_imports(self):
        from sglang.srt.layers.attention.flashinfer_quest_backend import (
            FlashInferQuestDecodeBackend,
        )
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        self.assertTrue(issubclass(FlashInferQuestDecodeBackend, FlashInferAttnBackend))

    def test_registered_in_attention_registry(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )

        self.assertIn("flashinfer_quest", ATTENTION_BACKENDS)

    def test_in_server_args_choices(self):
        from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

        self.assertIn("flashinfer_quest", ATTENTION_BACKEND_CHOICES)


if __name__ == "__main__":
    unittest.main()
