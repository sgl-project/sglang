"""
Test that speculative-decoding draft weights survive release/resume_memory_occupation.

Without the draft stash/restore in SchedulerWeightUpdaterManager, releasing the
WEIGHTS region discards the draft model and nothing ever restores it (the
distributed weight-update path only writes the target), so accept_length
silently collapses to ~1.0 after the first release/resume cycle — the exact
lifecycle colocated RL frameworks run every training step.

Two layers of coverage:
- CPU unit tests for the stash/restore helpers and the manager orchestration
  (no GPU, mock workers).
- A GPU end-to-end test: EAGLE engine + memory saver, assert accept_length
  does not degrade across release/resume.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_buffer("cache", torch.arange(4, dtype=torch.float32))


class TestDraftStashHelpers(CustomTestCase):
    """CPU-only: _export_full_state / _import_full_state round-trip semantics."""

    def test_roundtrip_params_and_buffers(self):
        model = _TinyModule()
        state = wu._export_full_state(model)
        names = [n for n, _ in state["tensors"]]
        self.assertIn("linear.weight", names)
        self.assertIn("cache", names)

        # clobber, then restore
        with torch.no_grad():
            model.linear.weight.fill_(0.0)
            model.cache.fill_(-1.0)
        failed = wu._import_full_state(model, state)
        self.assertEqual(failed, [])
        restored = dict(state["tensors"])
        self.assertTrue(torch.equal(model.linear.weight.detach(), restored["linear.weight"]))
        self.assertTrue(torch.equal(model.cache, restored["cache"]))

    def test_two_phase_restore_retries_failed_tensors(self):
        model = _TinyModule()
        state = wu._export_full_state(model)
        golden_cache = dict(state["tensors"])["cache"].clone()

        # Failure contract: a tensor whose in-place copy raises (here via a
        # shape mismatch, standing in for a still-paused region) is returned
        # in `failed` instead of failing the whole restore.
        state_bad = {"tensors": [("cache", torch.full((5,), 2.0))]}
        failed = wu._import_full_state(model, state_bad)
        self.assertEqual([n for n, _ in failed], ["cache"])

        # Retry with the good tensor succeeds — the second phase of the
        # two-phase restore.
        failed = wu._import_full_state(model, {"tensors": [("cache", golden_cache)]})
        self.assertEqual(failed, [])
        self.assertTrue(torch.equal(model.cache, golden_cache))


class TestManagerDraftOrchestration(CustomTestCase):
    """CPU-only: release stashes the draft; resume restores it (mock workers)."""

    def _make_manager(self, draft_model):
        draft_worker = MagicMock()
        draft_worker._draft_worker.draft_runner.model = draft_model
        # match EAGLEWorkerV2 attribute layout probed by _get_draft_model_runner
        draft_worker.draft_model_runner = None
        target_model = _TinyModule()
        tp_worker = MagicMock()
        tp_worker.model_runner.model = target_model
        manager = wu.SchedulerWeightUpdaterManager(
            tp_worker=tp_worker,
            draft_worker=draft_worker,
            tp_cpu_group=MagicMock(),
            memory_saver_adapter=MagicMock(),
            flush_cache=MagicMock(return_value=True),
            is_fully_idle=MagicMock(return_value=True),
        )
        return manager

    def test_release_then_resume_restores_draft(self):
        draft_model = _TinyModule()
        golden = {k: v.clone() for k, v in draft_model.state_dict().items()}
        manager = self._make_manager(draft_model)

        release_req = MagicMock(tags=[GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE])
        resume_req = MagicMock(tags=[GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE])

        with patch.object(torch.distributed, "barrier"):
            manager.release_memory_occupation(release_req)
            self.assertIsNotNone(manager.stashed_draft_model_state)

            # simulate the pause discarding draft contents
            with torch.no_grad():
                draft_model.linear.weight.fill_(0.0)
                draft_model.cache.fill_(0.0)

            manager.resume_memory_occupation(resume_req)

        self.assertIsNone(manager.stashed_draft_model_state)
        for k, v in draft_model.state_dict().items():
            self.assertTrue(torch.equal(v, golden[k]), f"draft tensor {k} not restored")

    def test_release_without_draft_worker_is_noop(self):
        manager = self._make_manager(_TinyModule())
        manager.draft_worker = None
        with patch.object(torch.distributed, "barrier"):
            manager.release_memory_occupation(MagicMock(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        self.assertIsNone(manager.stashed_draft_model_state)


class TestReleaseResumeWithSpeculativeDecoding(CustomTestCase):
    """GPU e2e: accept_length must survive a release/resume cycle."""

    PROMPT = "Explain, step by step, why the sky appears blue on a clear day."

    def _accept_length(self, engine):
        info = engine.get_server_info()
        return info["internal_states"][0]["avg_spec_accept_length"]

    def test_accept_length_survives_release_resume(self):
        import sglang as sgl

        engine = sgl.Engine(
            model_path=DEFAULT_TARGET_MODEL_EAGLE,
            speculative_algorithm="EAGLE",
            speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE,
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            enable_memory_saver=True,
            mem_fraction_static=0.6,
        )
        try:
            sampling = {"temperature": 0.0, "max_new_tokens": 128}
            engine.generate(self.PROMPT, sampling)
            accept_before = self._accept_length(engine)
            self.assertGreater(
                accept_before, 1.5, "speculative decoding not effective at baseline"
            )

            engine.release_memory_occupation()
            engine.resume_memory_occupation()

            engine.generate(self.PROMPT, sampling)
            accept_after = self._accept_length(engine)
            self.assertGreater(
                accept_after,
                accept_before - 0.3,
                f"accept_length degraded across release/resume: "
                f"{accept_before:.2f} -> {accept_after:.2f}",
            )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
