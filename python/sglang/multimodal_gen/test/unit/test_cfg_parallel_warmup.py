"""Unit tests for the --enable-cfg-parallel warmup fix and guard.

Covers two code paths introduced alongside this file:
- Scheduler.prepare_server_warmup_reqs synthesizes warmup Reqs that
  actually enable classifier-free guidance when cfg-parallel is on.
- InputValidationStage.forward rejects non-CFG requests when the server
  has cfg-parallel on.

All tests are CPU-only; no model loading, no distributed init.
"""

import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.managers.scheduler import (
    DEFAULT_PLACEHOLDER_PROMPT,
    Scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)

# Patch path for get_global_server_args used by Stage.__init__
_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


def _make_bare_scheduler(enable_cfg_parallel: bool) -> Scheduler:
    """
    Build a minimal Scheduler without calling __init__ (which requires
    distributed init, ZMQ sockets, pipeline load, etc.). Populates only
    the attributes prepare_server_warmup_reqs reads/writes for a
    text-only task so _prepare_shared_warmup_image_path is skipped.
    """
    scheduler = object.__new__(Scheduler)

    server_args = MagicMock()
    server_args.warmup = True
    server_args.warmup_steps = 1
    server_args.warmup_resolutions = ["512x512"]
    server_args.enable_cfg_parallel = enable_cfg_parallel

    # Text-only task — accepts_image_input() False skips the image-path
    # branch entirely, so we don't need to mock
    # _prepare_shared_warmup_image_path.
    task_type = MagicMock()
    task_type.accepts_image_input.return_value = False
    task_type.data_type.return_value = ModelTaskType.T2I.data_type()
    server_args.pipeline_config.task_type = task_type

    scheduler.server_args = server_args
    scheduler.warmed_up = False
    scheduler.waiting_queue = deque()
    return scheduler


def _make_input_validation_stage() -> InputValidationStage:
    """Construct InputValidationStage with the global server-args patch
    that existing tests in this suite use (see test_input_validation.py)."""
    with patch(_GLOBAL_ARGS_PATCH) as m:
        m.return_value = MagicMock()
        return InputValidationStage()


def _make_validation_server_args(enable_cfg_parallel: bool) -> MagicMock:
    sa = MagicMock()
    sa.enable_cfg_parallel = enable_cfg_parallel
    sa.pipeline_config.task_type = ModelTaskType.T2I
    return sa


class TestWarmupReqCfgParallel(unittest.TestCase):
    """Commit 1 regression: prepare_server_warmup_reqs."""

    def test_warmup_req_cfg_parallel_sets_do_cfg(self):
        scheduler = _make_bare_scheduler(enable_cfg_parallel=True)
        scheduler.prepare_server_warmup_reqs()

        self.assertEqual(len(scheduler.waiting_queue), 1)
        _, req = scheduler.waiting_queue[0]
        self.assertIs(req.do_classifier_free_guidance, True)
        self.assertEqual(req.negative_prompt, DEFAULT_PLACEHOLDER_PROMPT)

    def test_warmup_req_no_cfg_parallel_unchanged(self):
        # Regression guard: the cfg-parallel=on fix must not bleed into
        # the cfg-parallel=off path. Key invariant is do_cfg stays False
        # AND the synthesized Req is not using the cfg-parallel-specific
        # "warmup" placeholder for negative_prompt (which would indicate
        # the fix's kwargs leaked into this branch).
        scheduler = _make_bare_scheduler(enable_cfg_parallel=False)
        scheduler.prepare_server_warmup_reqs()

        self.assertEqual(len(scheduler.waiting_queue), 1)
        _, req = scheduler.waiting_queue[0]
        self.assertIs(req.do_classifier_free_guidance, False)
        self.assertNotEqual(req.negative_prompt, DEFAULT_PLACEHOLDER_PROMPT)


class TestInputValidationCfgParallelGuard(unittest.TestCase):
    """Commit 2: per-request cfg-parallel check.

    Both tests patch _generate_seeds (the first statement of
    InputValidationStage.forward, input_validation.py:274) to sidestep
    its device-lookup / generator-creation code which pulls in torch
    CUDA bindings — keeps the suite strictly CPU-only. We still need
    num_inference_steps on the Req because the stage's
    "num_inference_steps <= 0" check at L305-308 raises TypeError on
    None before the new commit-2 check is reached.
    """

    def test_input_validation_rejects_cfg_parallel_without_cfg(self):
        # negative_prompt="" (non-None) ensures the existing
        # negative_prompt-is-None check at input_validation.py:295-298
        # does NOT fire first — this isolates the new commit-2 check.
        # width/height/num_outputs_per_prompt pre-set so the stage's
        # default-dimension block at L352-361 doesn't mutate the Req
        # in a way that obscures the assertion target.
        req = Req(
            prompt="test",
            negative_prompt="",
            guidance_scale=1.0,
            true_cfg_scale=None,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
            width=512,
            height=512,
        )
        self.assertIs(
            req.do_classifier_free_guidance,
            False,
            "Sanity: test setup must leave do_cfg=False so the "
            "commit-2 check is the one that fires, not an upstream check.",
        )

        stage = _make_input_validation_stage()
        server_args = _make_validation_server_args(enable_cfg_parallel=True)

        with patch.object(InputValidationStage, "_generate_seeds"):
            with self.assertRaises(ValueError) as ctx:
                stage.forward(req, server_args)

        msg = str(ctx.exception).lower()
        self.assertIn("cfg-parallel", msg)
        for field in (
            "do_classifier_free_guidance",
            "guidance_scale",
            "true_cfg_scale",
            "negative_prompt",
        ):
            self.assertIn(field, str(ctx.exception))

    def test_input_validation_passes_cfg_parallel_with_cfg(self):
        req = Req(
            prompt="test",
            negative_prompt="bad",
            guidance_scale=4.0,
            true_cfg_scale=4.0,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
            width=512,
            height=512,
        )
        self.assertIs(
            req.do_classifier_free_guidance,
            True,
            "Sanity: req must enable CFG for this positive-case test.",
        )

        stage = _make_input_validation_stage()
        server_args = _make_validation_server_args(enable_cfg_parallel=True)

        with patch.object(InputValidationStage, "_generate_seeds"):
            try:
                stage.forward(req, server_args)
            except ValueError as e:
                self.fail(f"forward() raised ValueError on a valid CFG request: {e}")


if __name__ == "__main__":
    unittest.main()
