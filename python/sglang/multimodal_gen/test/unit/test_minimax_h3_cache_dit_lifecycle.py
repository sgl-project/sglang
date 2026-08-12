# -*- coding: utf-8 -*-
"""Lifecycle tests for MiniMax-H3 Cache-DiT input preservation.

These drive MiniMaxH3DenoisingStage._maybe_enable_cache_dit and assert on the
real block flags rather than stage state, and they record every setter call so
that a path which arms and then rolls back is distinguishable from one that
never arms at all.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (  # noqa: E501
    MiniMaxH3DenoisingStage,
)

PARENT = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
    "DenoisingStage._maybe_enable_cache_dit"
)
H3_MOD = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.minimax_h3.stages.denoising."
)


def _model(num_blocks=4, *, recorder=None):
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(num_blocks)])
    model.set_cache_dit_input_preservation(False)
    if recorder is not None:
        real = model.set_cache_dit_input_preservation

        def spy(enabled):
            recorder.append(enabled)
            real(enabled)

        model.set_cache_dit_input_preservation = spy
    return model


def _stage(*, bcg=False, torch_compile=False, recorder=None):
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.transformer = _model(recorder=recorder)
    stage.transformer_2 = None
    stage.server_args = types.SimpleNamespace(
        enable_breakable_cuda_graph=bcg,
        enable_torch_compile=torch_compile,
    )
    stage._cache_dit_enabled = False
    stage._cached_num_steps = None
    stage._minimax_h3_cache_mode = None
    stage._minimax_h3_quality = "lossless"
    return stage


def _batch(quality="high", *, warmup=False, explicit=("quality",)):
    return types.SimpleNamespace(
        sampling_params=types.SimpleNamespace(
            quality=quality, _explicit_fields=explicit
        ),
        is_warmup=warmup,
        do_classifier_free_guidance=False,
    )


def _flags(stage):
    return [b.preserve_input_for_cache_dit for b in stage.transformer.blocks]


class _ParentMount:
    """Stand-in for the parent mount, recording the flags it observes."""

    def __init__(self, *, succeed=True, raises=False, stage=None):
        self.succeed = succeed
        self.raises = raises
        self.stage = stage
        self.calls = 0
        self.flags_at_mount = None

    def __call__(self, stage, num_inference_steps, batch):
        self.calls += 1
        self.flags_at_mount = _flags(self.stage or stage)
        if self.raises:
            raise RuntimeError("mount failed")
        if self.succeed:
            stage._cache_dit_enabled = True


def _run(stage, batch, parent):
    with patch(PARENT, new=lambda self, n, b: parent(self, n, b)):
        stage._maybe_enable_cache_dit(50, batch)


class TestH3CacheDitLifecycle(unittest.TestCase):
    def test_successful_mount_arms_every_block_before_mounting(self):
        calls = []
        stage = _stage(recorder=calls)
        parent = _ParentMount(stage=stage)
        _run(stage, _batch("high"), parent)
        # Armed before the parent ran, not after.
        self.assertEqual(parent.flags_at_mount, [True] * 4)
        self.assertTrue(all(_flags(stage)))
        self.assertEqual(calls, [True])

    def test_mount_that_does_not_enable_leaves_no_residue(self):
        calls = []
        stage = _stage(recorder=calls)
        _run(stage, _batch("high"), _ParentMount(succeed=False, stage=stage))
        self.assertFalse(any(_flags(stage)))
        self.assertEqual(calls, [True, False])

    def test_mount_failure_unmounts_before_rolling_back(self):
        calls = []
        stage = _stage(recorder=calls)
        seen = {}

        def fake_disable(transformer):
            seen["flags"] = _flags(stage)
            return transformer

        with patch(H3_MOD + "disable_cache_on_transformer", new=fake_disable):
            with self.assertRaisesRegex(RuntimeError, "mount failed"):
                _run(stage, _batch("high"), _ParentMount(raises=True, stage=stage))

        # cache_dit may already have swapped the blocks before raising, so the
        # unmount has to happen while preservation is still on.
        self.assertEqual(seen["flags"], [True] * 4)
        self.assertFalse(any(_flags(stage)))
        self.assertEqual(calls, [True, False])

    def test_stays_armed_if_the_unmount_also_fails(self):
        """Better to pay throughput than to hand Cache-DiT unpreserved blocks."""
        calls = []
        stage = _stage(recorder=calls)

        def fake_disable(_transformer):
            raise RuntimeError("unmount failed")

        with patch(H3_MOD + "disable_cache_on_transformer", new=fake_disable):
            with self.assertRaisesRegex(RuntimeError, "mount failed"):
                _run(stage, _batch("high"), _ParentMount(raises=True, stage=stage))

        self.assertTrue(all(_flags(stage)))
        self.assertNotIn(False, calls)

    def test_mount_that_mutates_then_raises_does_not_disarm_a_live_cache(self):
        """The dangerous shape: enable_cache swaps blocks, then something later
        in the mount raises."""
        stage = _stage()
        unmounted = []

        def mutate_then_raise(inner_stage, _n, _b):
            inner_stage.transformer.blocks = torch.nn.ModuleList(
                [torch.nn.Identity()]
            )  # stand-in for the CachedBlocks swap
            raise RuntimeError("mount failed")

        def fake_disable(transformer):
            # disable_cache_on_transformer owns restoration of the original
            # block list. This test isolates the lifecycle contract: unmount
            # before disarming, then settle the stage state.
            unmounted.append(True)
            transformer.blocks = _model().blocks
            return transformer

        with patch(H3_MOD + "disable_cache_on_transformer", new=fake_disable):
            with self.assertRaisesRegex(RuntimeError, "mount failed"):
                _run(stage, _batch("high"), mutate_then_raise)

        self.assertEqual(unmounted, [True])
        self.assertFalse(any(_flags(stage)))
        # State must be settled too, or the next request takes the refresh path
        # against an unmounted transformer.
        self.assertFalse(stage._cache_dit_enabled)
        self.assertIsNone(stage._minimax_h3_cache_mode)

    def test_unexpected_mount_is_still_armed(self):
        """Conservative arming means an unforeseen parent gate is covered too.

        This is the case a mirrored predicate could only ever get wrong: the
        parent mounts on a path this stage did not anticipate.
        """
        stage = _stage(bcg=True)  # a path we would not have predicted mounts on
        seen = {}

        def parent_mounts_anyway(inner_stage, _n, _b):
            seen["flags"] = _flags(inner_stage)
            inner_stage._cache_dit_enabled = True

        _run(stage, _batch("high"), parent_mounts_anyway)
        self.assertEqual(seen["flags"], [True] * 4)
        self.assertTrue(all(_flags(stage)))

    def test_breakable_cuda_graph_ends_disarmed(self):
        calls = []
        stage = _stage(bcg=True, recorder=calls)
        _run(stage, _batch("high"), _ParentMount(succeed=False, stage=stage))
        # Armed conservatively, then restored once the parent declined.
        self.assertEqual(calls, [True, False])
        self.assertFalse(any(_flags(stage)))

    def test_ordinary_warmup_ends_disarmed(self):
        calls = []
        stage = _stage(recorder=calls)
        _run(
            stage, _batch("high", warmup=True), _ParentMount(succeed=False, stage=stage)
        )
        self.assertEqual(calls, [True, False])
        self.assertFalse(any(_flags(stage)))

    def test_torch_compile_warmup_does_arm(self):
        calls = []
        stage = _stage(torch_compile=True, recorder=calls)
        _run(stage, _batch("high", warmup=True), _ParentMount(stage=stage))
        self.assertEqual(calls, [True])
        self.assertTrue(all(_flags(stage)))

    def test_not_requested_ends_disarmed(self):
        calls = []
        stage = _stage(recorder=calls)
        _run(stage, _batch("lossless"), _ParentMount(succeed=False, stage=stage))
        self.assertFalse(any(_flags(stage)))

    def test_same_mode_repeat_does_not_re_arm(self):
        calls = []
        stage = _stage(recorder=calls)
        parent = _ParentMount(stage=stage)
        _run(stage, _batch("high"), parent)
        _run(stage, _batch("high"), parent)
        self.assertTrue(all(_flags(stage)))
        self.assertEqual(calls, [True])  # second pass refreshes context only
        self.assertEqual(parent.calls, 2)

    def test_disable_happens_before_restoring_in_place(self):
        stage = _stage()
        parent = _ParentMount(stage=stage)
        seen = {}

        def fake_disable(transformer):
            seen["flags"] = _flags(stage)
            return transformer

        with patch(H3_MOD + "disable_cache_on_transformer", new=fake_disable):
            _run(stage, _batch("high"), parent)
            self.assertTrue(all(_flags(stage)))
            _run(stage, _batch("lossless"), parent)

        # Cache is unmounted while preservation is still on, then restored.
        self.assertEqual(seen["flags"], [True] * 4)
        self.assertFalse(any(_flags(stage)))

    def test_high_to_lossless_to_high_round_trips(self):
        stage = _stage()
        parent = _ParentMount(stage=stage)
        with patch(H3_MOD + "disable_cache_on_transformer", new=lambda t: t):
            _run(stage, _batch("high"), parent)
            self.assertTrue(all(_flags(stage)))
            _run(stage, _batch("lossless"), parent)
            self.assertFalse(any(_flags(stage)))
            _run(stage, _batch("high"), parent)
            self.assertTrue(all(_flags(stage)))

    def test_missing_setter_fails_closed(self):
        stage = _stage()
        stage.transformer = torch.nn.Identity()  # no H3 helper on it
        with self.assertRaisesRegex(TypeError, "set_cache_dit_input_preservation"):
            _run(stage, _batch("high"), _ParentMount(stage=stage))

    def test_setter_is_found_through_wrappers(self):
        """torch.compile exposes _orig_mod, DDP/FSDP expose module."""
        for depth, attrs in ((1, ["_orig_mod"]), (2, ["module", "_orig_mod"])):
            with self.subTest(depth=depth):
                inner = _model()
                wrapped = inner
                for attr in attrs:
                    outer = torch.nn.Identity()
                    setattr(outer, attr, wrapped)
                    wrapped = outer

                stage = _stage()
                stage.transformer = wrapped
                stage._set_cache_dit_input_preservation(True)
                self.assertTrue(
                    all(b.preserve_input_for_cache_dit for b in inner.blocks)
                )
                stage._set_cache_dit_input_preservation(False)
                self.assertFalse(
                    any(b.preserve_input_for_cache_dit for b in inner.blocks)
                )

    def test_unwrapping_gives_up_rather_than_walking_forever(self):
        """A wrapper chain with no H3 model at the end must raise, not hang."""
        node = torch.nn.Identity()
        for _ in range(8):
            outer = torch.nn.Identity()
            outer.module = node
            node = outer

        stage = _stage()
        stage.transformer = node
        with self.assertRaisesRegex(TypeError, "set_cache_dit_input_preservation"):
            stage._set_cache_dit_input_preservation(True)


class TestH3CacheDitLifecycleAgainstRealParent(unittest.TestCase):
    """Exercise the real parent gates, mocking only the mount call itself."""

    def _stage_with_real_parent(self):
        stage = _stage()
        stage._cache_dit_step_counts = lambda n: (n, None)
        stage._cache_dit_scm_masks = lambda p, s=None: ("none", "dynamic", None, None)
        stage._build_cache_dit_config = lambda n, **kw: types.SimpleNamespace(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.24,
        )
        return stage

    def test_real_parent_mounts_with_preservation_already_armed(self):
        stage = self._stage_with_real_parent()
        seen = {}

        def fake_enable(transformer, config, **kwargs):
            seen["flags"] = _flags(stage)
            return transformer

        parent_mod = "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
        with (
            patch(parent_mod + "enable_cache_on_transformer", new=fake_enable),
            patch(parent_mod + "get_world_size", new=lambda: 1),
        ):
            stage._maybe_enable_cache_dit(50, _batch("high"))

        self.assertEqual(seen["flags"], [True] * 4)
        self.assertTrue(stage._cache_dit_enabled)
        self.assertTrue(all(_flags(stage)))

    def test_real_parent_bcg_path_leaves_blocks_untouched(self):
        stage = self._stage_with_real_parent()
        stage.server_args.enable_breakable_cuda_graph = True
        stage._maybe_enable_cache_dit(50, _batch("high"))
        self.assertFalse(stage._cache_dit_enabled)
        self.assertFalse(any(_flags(stage)))

    def test_real_parent_ordinary_warmup_does_not_mount(self):
        stage = self._stage_with_real_parent()
        stage._maybe_enable_cache_dit(50, _batch("high", warmup=True))
        self.assertFalse(stage._cache_dit_enabled)
        self.assertFalse(any(_flags(stage)))

    def test_real_parent_not_requested_does_not_mount(self):
        stage = self._stage_with_real_parent()
        stage._maybe_enable_cache_dit(50, _batch("lossless"))
        self.assertFalse(stage._cache_dit_enabled)
        self.assertFalse(any(_flags(stage)))


if __name__ == "__main__":
    unittest.main()
