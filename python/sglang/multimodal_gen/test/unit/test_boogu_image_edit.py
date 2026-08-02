# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Boogu-Image edit (reference-image / TI2I) path.

These are CPU-only and never load the checkpoint. They pin the pieces that a
future diff could silently break without the model actually failing to run:

* the ``--pipeline-class-name`` registry wiring that lets the edit pipeline
  override the path-resolved T2I config (the decoupling contract with PR
  #33182 — the base T2I registration must stay untouched);
* the dual (text + image) CFG branch construction and combination math; and
* the VAE-encode scale/shift order, which is the inverse of the decode
  normalization and is easy to invert by accident.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.boogu_image import (
    BOOGU_SYSTEM_PROMPT_DROP,
    BooguImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.boogu_image_edit import (
    BooguImageEditCFGPolicy,
    BooguImageEditPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.boogu_image import BooguImageSamplingParams
from sglang.multimodal_gen.configs.sample.boogu_image_edit import (
    BooguImageEditSamplingParams,
)
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.boogu_image_edit import (
    BooguImageEditEncodingStage,
)


def _edit_batch(
    *,
    ref,
    do_cfg: bool,
    guidance_scale: float,
    guidance_scale_2: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        ref_image_hidden_states=ref,
        do_classifier_free_guidance=do_cfg,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
    )


class TestBooguImageEditRegistryWiring(unittest.TestCase):
    """The decoupled wiring: `--pipeline-class-name BooguImageEditPipeline`."""

    def test_edit_pipeline_registers_edit_config_and_sampling_classes(self) -> None:
        classes = get_pipeline_config_classes("BooguImageEditPipeline")
        self.assertIsNotNone(
            classes,
            "BooguImageEditPipeline must expose pipeline_config_cls / "
            "sampling_params_cls so get_pipeline_config_classes can override the "
            "path-resolved T2I config.",
        )
        pipeline_config_cls, sampling_params_cls = classes
        self.assertIs(pipeline_config_cls, BooguImageEditPipelineConfig)
        self.assertIs(sampling_params_cls, BooguImageEditSamplingParams)

    def test_base_t2i_pipeline_stays_path_resolved(self) -> None:
        # Decoupling contract: PR #33182's T2I pipeline registers no config
        # override, so it keeps resolving its config by model path. If this ever
        # returns a value the edit PR has leaked into the T2I registration.
        self.assertIsNone(get_pipeline_config_classes("BooguImagePipeline"))

    def test_edit_config_strictly_subclasses_t2i_config(self) -> None:
        # base.py refines the model-default config to the pipeline's config ONLY
        # when it is a strict subclass; breaking this inheritance would silently
        # fall back to the T2I config with no error.
        self.assertTrue(
            issubclass(BooguImageEditPipelineConfig, BooguImagePipelineConfig)
        )
        self.assertIsNot(BooguImageEditPipelineConfig, BooguImagePipelineConfig)
        self.assertTrue(
            issubclass(BooguImageEditSamplingParams, BooguImageSamplingParams)
        )

    def test_edit_config_task_type_is_ti2i(self) -> None:
        self.assertEqual(BooguImageEditPipelineConfig().task_type, ModelTaskType.TI2I)


class TestBooguImageEditSamplingParams(unittest.TestCase):
    def test_second_guidance_scale_defaults_off(self) -> None:
        # Upstream default edit request uses text-only guidance; image guidance
        # is opt-in (scale_2 > 1). Inherits the T2I 4.0 text guidance default.
        params = BooguImageEditSamplingParams()
        self.assertEqual(params.guidance_scale_2, 1.0)
        self.assertEqual(params.guidance_scale, 4.0)
        self.assertEqual(params.num_inference_steps, 50)


class TestBooguImageEditCFGPolicyBuild(unittest.TestCase):
    """Branch construction reproduces the upstream drop priority."""

    def setUp(self) -> None:
        self.policy = BooguImageEditCFGPolicy()
        self.ref = [[torch.zeros(4, 2, 2)]]
        self.image_kwargs = {"hidden_states": torch.zeros(1)}
        self.pos = {"encoder_hidden_states": "pos"}
        self.neg = {"encoder_hidden_states": "neg"}

    def _build(self, batch):
        return self.policy.build(batch, self.image_kwargs, self.pos, self.neg)

    def test_double_guidance_builds_three_branches(self) -> None:
        batch = _edit_batch(
            ref=self.ref, do_cfg=True, guidance_scale=4.0, guidance_scale_2=2.0
        )
        branches = self._build(batch).branches
        self.assertEqual(
            [b.name for b in branches], ["cond_ref", "drop_text", "drop_all"]
        )
        # cond keeps positive text + ref; drop_text keeps ref but negative text;
        # drop_all drops both text and ref.
        self.assertEqual(branches[0].kwargs["encoder_hidden_states"], "pos")
        self.assertIs(branches[0].kwargs["ref_image_hidden_states"], self.ref)
        self.assertEqual(branches[1].kwargs["encoder_hidden_states"], "neg")
        self.assertIs(branches[1].kwargs["ref_image_hidden_states"], self.ref)
        self.assertEqual(branches[2].kwargs["encoder_hidden_states"], "neg")
        self.assertIsNone(branches[2].kwargs["ref_image_hidden_states"])

    def test_text_only_guidance_builds_two_branches(self) -> None:
        batch = _edit_batch(
            ref=self.ref, do_cfg=True, guidance_scale=4.0, guidance_scale_2=1.0
        )
        branches = self._build(batch).branches
        self.assertEqual([b.name for b in branches], ["cond_ref", "drop_text"])
        # Both branches keep the reference latent (only text guidance is active).
        self.assertIs(branches[1].kwargs["ref_image_hidden_states"], self.ref)

    def test_image_only_guidance_drops_only_reference(self) -> None:
        batch = _edit_batch(
            ref=self.ref, do_cfg=False, guidance_scale=1.0, guidance_scale_2=2.0
        )
        branches = self._build(batch).branches
        self.assertEqual([b.name for b in branches], ["cond_ref", "drop_image"])
        # Image-only guidance keeps the positive instruction and drops the ref.
        self.assertEqual(branches[1].kwargs["encoder_hidden_states"], "pos")
        self.assertIsNone(branches[1].kwargs["ref_image_hidden_states"])

    def test_no_guidance_builds_single_branch(self) -> None:
        batch = _edit_batch(
            ref=None, do_cfg=False, guidance_scale=1.0, guidance_scale_2=1.0
        )
        branches = self._build(batch).branches
        self.assertEqual([b.name for b in branches], ["cond_ref"])


class TestBooguImageEditCFGPolicyCombine(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BooguImageEditCFGPolicy()
        # Neutralize normalization / rescale so we isolate the guidance formula.
        self.batch = SimpleNamespace(
            cfg_normalization=0.0,
            guidance_rescale=0.0,
            guidance_scale_2=3.0,
        )
        self.pipeline_config = SimpleNamespace(
            postprocess_cfg_noise=lambda batch, noise, cond: noise
        )

    def test_single_prediction_returned_unchanged(self) -> None:
        pred = torch.tensor([1.0, 2.0])
        out = self.policy.combine(
            [pred], self.batch, cfg_scale=4.0, pipeline_config=self.pipeline_config
        )
        self.assertTrue(torch.equal(out, pred))

    def test_three_branch_formula_matches_upstream(self) -> None:
        c = torch.tensor([1.0])
        dt = torch.tensor([0.5])
        da = torch.tensor([0.25])
        text_gs, image_gs = 4.0, 3.0
        self.batch.guidance_scale_2 = image_gs

        out = self.policy.combine(
            [c, dt, da],
            self.batch,
            cfg_scale=text_gs,
            pipeline_config=self.pipeline_config,
        )

        expected = c + (text_gs - 1.0) * (c - dt) + (image_gs - 1.0) * (dt - da)
        self.assertTrue(torch.allclose(out, expected))

    def test_two_branch_uses_active_scale(self) -> None:
        c = torch.tensor([1.0])
        u = torch.tensor([0.2])
        # Text-only: cfg_scale > 1, guidance_scale_2 == 1 -> use text scale.
        self.batch.guidance_scale_2 = 1.0
        out = self.policy.combine(
            [c, u], self.batch, cfg_scale=4.0, pipeline_config=self.pipeline_config
        )
        self.assertTrue(torch.allclose(out, c + (4.0 - 1.0) * (c - u)))

        # Image-only: cfg_scale == 1, guidance_scale_2 > 1 -> use image scale.
        self.batch.guidance_scale_2 = 3.0
        out = self.policy.combine(
            [c, u], self.batch, cfg_scale=1.0, pipeline_config=self.pipeline_config
        )
        self.assertTrue(torch.allclose(out, c + (3.0 - 1.0) * (c - u)))


class TestBooguImageEditVaeEncodeNormalization(unittest.TestCase):
    def test_encode_applies_shift_then_scale(self) -> None:
        # decode is `latent / scale + shift`, so encode must be
        # `(sample - shift) * scale`. A swapped order would give
        # `sample * scale - shift`, which this pins against.
        latent = torch.tensor([1.0, 2.0])
        scale, shift = 0.3611, 0.1159

        out = BooguImageEditEncodingStage._apply_encode_scale_shift(
            latent, scale, shift
        )

        expected = (latent - shift) * scale
        self.assertTrue(torch.allclose(out, expected))
        # Guard the ordering explicitly: scale-first would not match.
        self.assertFalse(torch.allclose(out, latent * scale - shift))

    def test_encode_is_noop_when_factors_missing(self) -> None:
        latent = torch.tensor([1.0, 2.0])
        out = BooguImageEditEncodingStage._apply_encode_scale_shift(latent, None, None)
        self.assertTrue(torch.equal(out, latent))


class TestBooguImageEditMessageBuilding(unittest.TestCase):
    def test_image_present_uses_drop_prompt_and_image_first(self) -> None:
        stage = object.__new__(BooguImageEditEncodingStage)
        pil = SimpleNamespace(name="fake-pil")

        messages = stage._build_edit_messages("make it night", pil)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"][0]["text"], BOOGU_SYSTEM_PROMPT_DROP)
        user = messages[1]
        self.assertEqual(user["role"], "user")
        # Image must precede text in the user turn (upstream ordering).
        self.assertEqual(user["content"][0]["type"], "image")
        self.assertIs(user["content"][0]["image"], pil)
        self.assertEqual(user["content"][1]["type"], "text")
        self.assertEqual(user["content"][1]["text"], "make it night")

    def test_empty_instruction_still_uses_drop_prompt(self) -> None:
        stage = object.__new__(BooguImageEditEncodingStage)
        messages = stage._build_edit_messages("", SimpleNamespace())
        self.assertEqual(messages[0]["content"][0]["text"], BOOGU_SYSTEM_PROMPT_DROP)
        self.assertEqual(messages[1]["content"][1]["text"], "")


if __name__ == "__main__":
    unittest.main()
