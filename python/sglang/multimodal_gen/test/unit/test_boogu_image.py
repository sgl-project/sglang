# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.boogu_image import (
    BOOGU_SYSTEM_PROMPT_DROP,
    BooguImageCFGPolicy,
    BooguImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.boogu_image import BooguImageSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.boogu_image import (
    BooguImageEncodingStage,
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


class TestBooguImageConfig(unittest.TestCase):
    def test_task_type_is_ti2i(self) -> None:
        self.assertEqual(BooguImagePipelineConfig().task_type, ModelTaskType.TI2I)


class TestBooguImageSamplingParams(unittest.TestCase):
    def test_second_guidance_scale_defaults_off(self) -> None:
        params = BooguImageSamplingParams()
        self.assertIsNotNone(params.guidance_scale_2)
        self.assertEqual(params.guidance_scale_2, 1.0)
        self.assertEqual(params.guidance_scale, 4.0)
        self.assertEqual(params.num_inference_steps, 50)


class TestBooguImageCFGPolicyBuild(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BooguImageCFGPolicy()
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
        self.assertIs(branches[1].kwargs["ref_image_hidden_states"], self.ref)

    def test_image_only_guidance_drops_only_reference(self) -> None:
        batch = _edit_batch(
            ref=self.ref, do_cfg=False, guidance_scale=1.0, guidance_scale_2=2.0
        )
        branches = self._build(batch).branches
        self.assertEqual([b.name for b in branches], ["cond_ref", "drop_image"])
        self.assertEqual(branches[1].kwargs["encoder_hidden_states"], "pos")
        self.assertIsNone(branches[1].kwargs["ref_image_hidden_states"])

    def test_no_guidance_builds_single_branch(self) -> None:
        batch = _edit_batch(
            ref=None, do_cfg=False, guidance_scale=1.0, guidance_scale_2=1.0
        )
        branches = self._build(batch).branches
        self.assertEqual([b.name for b in branches], ["cond_ref"])


class TestBooguImageCFGPolicyCombine(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BooguImageCFGPolicy()
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
        self.batch.guidance_scale_2 = 1.0
        out = self.policy.combine(
            [c, u], self.batch, cfg_scale=4.0, pipeline_config=self.pipeline_config
        )
        self.assertTrue(torch.allclose(out, c + (4.0 - 1.0) * (c - u)))

        self.batch.guidance_scale_2 = 3.0
        out = self.policy.combine(
            [c, u], self.batch, cfg_scale=1.0, pipeline_config=self.pipeline_config
        )
        self.assertTrue(torch.allclose(out, c + (3.0 - 1.0) * (c - u)))


class TestBooguImageCFGPolicyT2IUnchanged(unittest.TestCase):
    def test_reference_free_text_guidance_is_standard_cfg(self) -> None:
        policy = BooguImageCFGPolicy()
        build_batch = _edit_batch(
            ref=None, do_cfg=True, guidance_scale=4.0, guidance_scale_2=1.0
        )
        branches = policy.build(
            build_batch,
            {"hidden_states": torch.zeros(1)},
            {"encoder_hidden_states": "pos"},
            {"encoder_hidden_states": "neg"},
        ).branches
        self.assertEqual([b.name for b in branches], ["cond_ref", "drop_text"])
        self.assertIsNone(branches[0].kwargs["ref_image_hidden_states"])
        self.assertIsNone(branches[1].kwargs["ref_image_hidden_states"])

        combine_batch = SimpleNamespace(
            cfg_normalization=0.0,
            guidance_rescale=0.0,
            guidance_scale_2=1.0,
        )
        pipeline_config = SimpleNamespace(
            postprocess_cfg_noise=lambda batch, noise, cond: noise
        )
        c = torch.tensor([1.0])
        u = torch.tensor([0.2])
        out = policy.combine(
            [c, u], combine_batch, cfg_scale=4.0, pipeline_config=pipeline_config
        )
        self.assertTrue(torch.allclose(out, c + (4.0 - 1.0) * (c - u)))


class TestBooguImageVaeEncodeNormalization(unittest.TestCase):
    def test_encode_applies_shift_then_scale(self) -> None:
        latent = torch.tensor([1.0, 2.0])
        scale, shift = 0.3611, 0.1159

        out = BooguImageEncodingStage._apply_encode_scale_shift(latent, scale, shift)

        expected = (latent - shift) * scale
        self.assertTrue(torch.allclose(out, expected))
        self.assertFalse(torch.allclose(out, latent * scale - shift))

    def test_encode_is_noop_when_factors_missing(self) -> None:
        latent = torch.tensor([1.0, 2.0])
        out = BooguImageEncodingStage._apply_encode_scale_shift(latent, None, None)
        self.assertTrue(torch.equal(out, latent))


class TestBooguImageMessageBuilding(unittest.TestCase):
    def test_image_present_uses_drop_prompt_and_image_first(self) -> None:
        stage = object.__new__(BooguImageEncodingStage)
        pil = SimpleNamespace(name="fake-pil")

        messages = stage._build_edit_messages("make it night", pil)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"][0]["text"], BOOGU_SYSTEM_PROMPT_DROP)
        user = messages[1]
        self.assertEqual(user["role"], "user")
        self.assertEqual(user["content"][0]["type"], "image")
        self.assertIs(user["content"][0]["image"], pil)
        self.assertEqual(user["content"][1]["type"], "text")
        self.assertEqual(user["content"][1]["text"], "make it night")

    def test_empty_instruction_still_uses_drop_prompt(self) -> None:
        stage = object.__new__(BooguImageEncodingStage)
        messages = stage._build_edit_messages("", SimpleNamespace())
        self.assertEqual(messages[0]["content"][0]["text"], BOOGU_SYSTEM_PROMPT_DROP)
        self.assertEqual(messages[1]["content"][1]["text"], "")


if __name__ == "__main__":
    unittest.main()
