import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.test.test_utils import DEFAULT_SANA_WM_MODEL_NAME_FOR_TEST


class TestSanaWMPipelineConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SanaWMPipelineConfig()

    def test_task_type_is_ti2v(self) -> None:
        self.assertEqual(self.config.task_type, ModelTaskType.TI2V)

    def test_adjust_num_frames_rounds_to_temporal_stride(self) -> None:
        self.assertEqual(self.config.adjust_num_frames(50), 49)
        self.assertEqual(self.config.adjust_num_frames(49), 49)

    def test_prepare_latent_shape_returns_5d(self) -> None:
        batch = SimpleNamespace(height=704, width=1280)
        shape = self.config.prepare_latent_shape(batch, batch_size=1, num_frames=49)
        self.assertEqual(shape, (1, 128, 7, 22, 40))

    def test_prepare_pos_cond_kwargs_passes_camera_from_batch_extra(self) -> None:
        # SanaWMBeforeDenoisingStage flattens c2w + intrinsics into the upstream
        # (B, F_orig, 20) camera_conditions tensor, and packs Plücker into a
        # (B, 48, T_lat, H, W) ``chunk_plucker``. The pipeline config just
        # forwards those tensors verbatim.
        camera_conditions = torch.zeros(1, 49, 20)
        chunk_plucker = torch.zeros(1, 48, 7, 22, 40)
        batch = SimpleNamespace(
            prompt_attention_mask=torch.ones(1, 16),
            extra={
                "camera_conditions": camera_conditions,
                "chunk_plucker": chunk_plucker,
            },
        )
        kwargs = self.config.prepare_pos_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.bfloat16,
        )
        self.assertIs(kwargs["camera_conditions"], camera_conditions)
        self.assertIs(kwargs["chunk_plucker"], chunk_plucker)

    def test_get_model_deployment_config_enables_dit_layerwise_offload(self) -> None:
        deployment = self.config.get_model_deployment_config()
        self.assertTrue(deployment.auto_dit_layerwise_offload)

    def test_prepare_neg_cond_kwargs_omits_camera(self) -> None:
        batch = SimpleNamespace(
            negative_attention_mask=torch.ones(1, 16),
            extra={
                "camera_conditions": torch.zeros(1, 49, 20),
                "chunk_plucker": torch.zeros(1, 48, 7, 22, 40),
            },
        )
        kwargs = self.config.prepare_neg_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.bfloat16,
        )
        self.assertIn("encoder_attention_mask", kwargs)
        self.assertNotIn("camera_conditions", kwargs)
        self.assertNotIn("chunk_plucker", kwargs)


class TestSanaWMSamplingParams(unittest.TestCase):
    def test_defaults_match_video_ti2v_contract(self) -> None:
        params = SanaWMSamplingParams()
        self.assertEqual(params.height, 704)
        self.assertEqual(params.width, 1280)
        self.assertEqual(params.num_frames, 49)
        self.assertEqual(params.num_inference_steps, 20)
        self.assertEqual(params.guidance_scale, 4.5)

    def test_build_request_extra_omits_camera_by_default(self) -> None:
        params = SanaWMSamplingParams()
        extra = params.build_request_extra()
        self.assertNotIn("camera_to_world", extra)
        self.assertNotIn("intrinsics", extra)

    def test_build_request_extra_includes_camera_when_set(self) -> None:
        cam = torch.eye(4).unsqueeze(0).expand(49, 4, 4)
        intr = torch.eye(3).unsqueeze(0).expand(49, 3, 3)
        params = SanaWMSamplingParams(camera_to_world=cam, intrinsics=intr)
        extra = params.build_request_extra()
        self.assertIs(extra["camera_to_world"], cam)
        self.assertIs(extra["intrinsics"], intr)


class TestSanaWMRegistry(unittest.TestCase):
    def setUp(self) -> None:
        _get_config_info.cache_clear()

    def test_model_path_resolves_sana_wm_pipeline_config(self) -> None:
        info = _get_config_info(DEFAULT_SANA_WM_MODEL_NAME_FOR_TEST)
        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_config_cls, SanaWMPipelineConfig)
        self.assertIs(info.sampling_param_cls, SanaWMSamplingParams)


if __name__ == "__main__":
    unittest.main()
