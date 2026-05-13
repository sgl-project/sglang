# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    build_sensenova_u1_sampling_params,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1.pixel_flow import (
    SenseNovaU1PixelFlowStage,
    _resolve_u1_contexts,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args


class TestSenseNovaU1PixelFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args(ServerArgs(model_path="dummy"))

    def test_stage_prepare_keeps_position_and_timestep_baseline(self):
        params = build_sensenova_u1_sampling_params(
            {
                "width": 16,
                "height": 16,
                "num_inference_steps": 2,
                "seed": 7,
                "cfg_text_scale": 1.0,
                "timestep_shift": 1.0,
                "omni_generation_mode": "t2i",
            }
        )
        batch = Req(sampling_params=params, prompt="draw")
        stage = SenseNovaU1PixelFlowStage()

        prepared = stage._prepare(
            model=_FakeModel(),
            context_metadata={"attention_math_mode": "reference_eager"},
            batch=batch,
            u1_context=SimpleNamespace(
                session_id="s0",
                condition_path_role=None,
                position_count=5,
            ),
        )

        self.assertEqual((4, 4), (prepared.token_h, prepared.token_w))
        self.assertEqual((8, 8), (prepared.grid_h, prepared.grid_w))
        torch.testing.assert_close(
            prepared.timesteps,
            torch.tensor([0.0, 0.5, 1.0]),
        )
        torch.testing.assert_close(
            prepared.condition.indexes_image[:, :3],
            torch.tensor([[5, 5, 5], [0, 0, 0], [0, 1, 2]]),
        )
        self.assertEqual(7, prepared.seed)
        self.assertEqual((1, 3, 16, 16), tuple(prepared.image_prediction.shape))
        self.assertFalse(prepared.commit_generated_image)
        self.assertNotIn(
            "cross_attention_custom_mask",
            prepared.condition.prepared.generation_input,
        )
        self.assertEqual(
            "reference_eager",
            prepared.condition.prepared.generation_input["attention_math_mode"],
        )

    def test_pixel_flow_stage_uses_denoising_stage_contract(self):
        self.assertIsInstance(SenseNovaU1PixelFlowStage(), DenoisingStage)

    def test_edit_cfg_uses_image_condition_path(self):
        params = build_sensenova_u1_sampling_params(
            {
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.0,
                "omni_generation_mode": "edit",
            }
        )
        batch = Req(sampling_params=params, prompt="edit")

        full, img_condition, uncondition = _resolve_u1_contexts(
            context_ops=_FakeContextOps(),
            batch=batch,
        )

        self.assertEqual("s0", full.session_id)
        self.assertEqual(10, full.position_count)
        self.assertEqual("u1_edit_img_condition", img_condition.condition_path_role)
        self.assertEqual(8, img_condition.position_count)
        self.assertIsNone(uncondition)

    def test_interleave_prepare_keeps_generated_image_commit(self):
        params = build_sensenova_u1_sampling_params(
            {
                "width": 16,
                "height": 16,
                "num_inference_steps": 1,
                "cfg_text_scale": 1.0,
                "omni_generation_mode": "interleave",
            }
        )
        batch = Req(sampling_params=params, prompt="draw")
        stage = SenseNovaU1PixelFlowStage()

        prepared = stage._prepare(
            model=_FakeModel(),
            context_metadata={},
            batch=batch,
            u1_context=SimpleNamespace(
                session_id="s0",
                condition_path_role=None,
                position_count=5,
            ),
        )

        self.assertTrue(prepared.commit_generated_image)


class _FakeModel:
    def __init__(self):
        self.vision_model = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=2),
            downsample_ratio=0.5,
            time_schedule="standard",
            time_shift_type="exponential",
            noise_scale=1.0,
            noise_scale_mode="constant",
            noise_scale_base_image_seq_len=256,
            noise_scale_max_value=10.0,
            add_noise_scale_embedding=False,
        )


class _FakeContextOps:
    session_id = "s0"

    def get_role(self, name, default):
        return default

    def get_position_count(self, *, condition_path_role=None):
        counts = {
            None: 10,
            "u1_edit_img_condition": 8,
        }
        return counts[condition_path_role]


if __name__ == "__main__":
    unittest.main()
