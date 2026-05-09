# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    build_sensenova_u1_sampling_params,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_executor import (
    _resolve_u1_contexts,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_denoise import (
    _should_apply_cfg,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_prepare import (
    SenseNovaU1PixelFlowPreparer,
)


class TestSenseNovaU1PixelFlow(unittest.TestCase):
    def test_preparer_keeps_position_and_timestep_baseline(self):
        params = build_sensenova_u1_sampling_params(
            {
                "width": 16,
                "height": 16,
                "num_inference_steps": 2,
                "seed": 7,
                "timestep_shift": 1.0,
            }
        )
        batch = Req(sampling_params=params, prompt="draw")
        preparer = SenseNovaU1PixelFlowPreparer(_FakeModel())

        prepared = preparer.forward(
            context_metadata={},
            batch=batch,
            u1_context=SimpleNamespace(
                session_id="s0",
                sidecar_role=None,
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

    def test_preparer_prefers_expanded_batch_seed(self):
        params = build_sensenova_u1_sampling_params(
            {
                "width": 16,
                "height": 16,
                "num_inference_steps": 2,
                "seed": [7],
            }
        )
        batch = Req(sampling_params=params, prompt="draw")
        batch.seeds = [19]
        preparer = SenseNovaU1PixelFlowPreparer(_FakeModel())

        prepared = preparer.forward(
            context_metadata={},
            batch=batch,
            u1_context=SimpleNamespace(
                session_id="s0",
                sidecar_role=None,
                position_count=5,
            ),
        )

        self.assertEqual(19, prepared.seed)

    def test_edit_cfg_uses_image_condition_sidecar(self):
        params = build_sensenova_u1_sampling_params(
            {
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.0,
            }
        )
        params.ug_generation_mode = "edit"
        batch = Req(sampling_params=params, prompt="edit")

        full, img_condition, uncondition = _resolve_u1_contexts(
            context_ops=_FakeContextOps(),
            batch=batch,
        )

        self.assertEqual("s0", full.session_id)
        self.assertEqual(10, full.position_count)
        self.assertEqual("u1_edit_img_condition", img_condition.sidecar_role)
        self.assertEqual(8, img_condition.position_count)
        self.assertIsNone(uncondition)

    def test_cfg_interval_start_zero_still_respects_end(self):
        cfg = build_sensenova_u1_sampling_params(
            {
                "cfg_text_scale": 4.0,
                "cfg_interval": [0.0, 0.5],
            }
        ).resolve_pixel_flow_cfg()

        self.assertTrue(_should_apply_cfg(cfg, torch.tensor(0.0)))
        self.assertTrue(_should_apply_cfg(cfg, torch.tensor(0.49)))
        self.assertFalse(_should_apply_cfg(cfg, torch.tensor(0.5)))
        self.assertFalse(_should_apply_cfg(cfg, torch.tensor(0.75)))


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
        del name
        return default

    def get_position_count(self, *, sidecar_role=None):
        counts = {
            None: 10,
            "u1_edit_img_condition": 8,
        }
        return counts[sidecar_role]


if __name__ == "__main__":
    unittest.main()
