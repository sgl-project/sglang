"""Loaded target tactics survive draft warmup, unless cache reuse is disabled."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner import flashinfer_autotune as warmup
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer requires CUDA")
class TestAutotuneCachePhases(CustomTestCase):
    def test_target_and_draft_cache_reuse(self):
        from flashinfer.autotuner import AutoTuner, _collect_metadata

        tuner = AutoTuner.get()
        tuner.clear_cache()
        self.addCleanup(tuner.clear_cache)
        runner = SimpleNamespace(
            device="cuda",
            forward_stream=torch.cuda.Stream(),
            tp_group=SimpleNamespace(world_size=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            target, draft = (
                Path(directory) / name for name in ("target.json", "draft.json")
            )
            for path, key, tactic in (
                (target, "target_prefill", 7),
                (draft, "draft_decode", 3),
            ):
                path.write_text(
                    json.dumps(
                        {"_metadata": _collect_metadata(), key: ["TestRunner", tactic]}
                    )
                )
            with (
                patch.object(
                    warmup,
                    "flashinfer_autotune_cache_path",
                    side_effect=[target, draft, draft],
                ),
                patch.object(
                    warmup, "get_flashinfer_autotune_skip_ops", return_value=set()
                ),
                warmup.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
            ):
                with warmup.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner._file_configs["target_prefill"], ("TestRunner", 7)
                    )
                # No profiling: this models a restart that loads tactics from disk.
                self.assertFalse(tuner.profiling_cache)
                with warmup.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner._file_configs["target_prefill"], ("TestRunner", 7)
                    )
                    self.assertEqual(
                        tuner._file_configs["draft_decode"], ("TestRunner", 3)
                    )
                saved = json.loads(draft.read_text())
                self.assertEqual(saved["target_prefill"], ["TestRunner", 7])
                self.assertEqual(saved["draft_decode"], ["TestRunner", 3])
                with (
                    warmup.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False),
                    warmup.flashinfer_autotune_context(runner, run_lm_head=False),
                ):
                    self.assertNotIn("target_prefill", tuner._file_configs)
                    self.assertNotIn("draft_decode", tuner._file_configs)


if __name__ == "__main__":
    unittest.main()
