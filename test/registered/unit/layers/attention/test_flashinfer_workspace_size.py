"""Unit tests for FlashInfer workspace-size resolution.

Deterministic inference and some models raise the workspace above the env
default, but an explicitly set SGLANG_FLASHINFER_WORKSPACE_SIZE must win, in
either direction: a smaller value so an 8 GB card can start, a larger one so a
model that needs more is not cut back to the per-model default.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

MiB = 1024 * 1024
_ENV = "SGLANG_FLASHINFER_WORKSPACE_SIZE"
_resolve = FlashInferAttnBackend._resolve_workspace_size


class TestResolveWorkspaceSize(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(_ENV, None)

    def test_other_models_keep_the_env_default(self):
        self.assertIsNone(_resolve(["LlamaForCausalLM"], False))

    def test_qwen_default_is_512_mib(self):
        self.assertEqual(_resolve(["Qwen3ForCausalLM"], False), 512 * MiB)

    def test_deterministic_default_is_2_gib_and_beats_the_model_default(self):
        self.assertEqual(_resolve(["LlamaForCausalLM"], True), 2048 * MiB)
        self.assertEqual(_resolve(["Qwen3ForCausalLM"], True), 2048 * MiB)

    def test_explicit_smaller_size_wins_in_deterministic_mode(self):
        with envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(384 * MiB):
            self.assertIsNone(_resolve(["Qwen3ForCausalLM"], True))
            self.assertEqual(envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(), 384 * MiB)

    def test_explicit_larger_size_is_not_cut_back_to_the_model_default(self):
        with envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.override(1024 * MiB):
            self.assertIsNone(_resolve(["Qwen3ForCausalLM"], False))
            self.assertEqual(envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(), 1024 * MiB)


if __name__ == "__main__":
    unittest.main()
