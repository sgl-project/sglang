"""A cached target warmup must survive a subsequent draft warmup."""

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
    def test_disabling_cache_reuse_drops_file_loaded_tactics(self):
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
            cache = Path(directory) / "cached.json"
            cache.write_text(
                json.dumps(
                    {
                        "_metadata": _collect_metadata(),
                        "old_tactic": ["TestRunner", 7],
                    }
                )
            )
            tuner.load_configs(str(cache))
            self.assertIn("old_tactic", tuner._file_configs)
            with (
                patch.object(
                    warmup, "flashinfer_autotune_cache_path", return_value=cache
                ),
                patch.object(
                    warmup, "get_flashinfer_autotune_skip_ops", return_value=set()
                ),
                patch.object(
                    warmup.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE,
                    "get",
                    return_value=False,
                ),
                warmup.flashinfer_autotune_context(runner, run_lm_head=False),
            ):
                self.assertNotIn("old_tactic", tuner._file_configs)

    def test_loaded_target_tactics_survive_draft_cache(self):
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
            target = Path(directory) / "target.json"
            draft = Path(directory) / "draft.json"
            metadata = _collect_metadata()
            target.write_text(
                json.dumps({"_metadata": metadata, "target_prefill": ["TestRunner", 7]})
            )
            draft.write_text(
                json.dumps({"_metadata": metadata, "draft_decode": ["TestRunner", 3]})
            )
            with (
                patch.object(
                    warmup,
                    "flashinfer_autotune_cache_path",
                    side_effect=[target, draft],
                ),
                patch.object(
                    warmup, "get_flashinfer_autotune_skip_ops", return_value=set()
                ),
                patch.object(
                    warmup.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE,
                    "get",
                    return_value=True,
                ),
            ):
                with warmup.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner._file_configs["target_prefill"], ("TestRunner", 7)
                    )
                # No operation was profiled: this simulates a restart where all
                # target tactics came from disk, not the in-memory profile cache.
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


if __name__ == "__main__":
    unittest.main()
