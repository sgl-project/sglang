"""Regression tests for Qwen3-VL multimodal feature materialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingVisual:
    device = torch.device("meta")
    dtype = torch.bfloat16

    def __init__(self):
        self.pixel_values = None
        self.grid_thw = None

    def __call__(self, pixel_values, *, grid_thw):
        self.pixel_values = pixel_values
        self.grid_thw = grid_thw
        return pixel_values


class TestQwen3VLFeatureMaterialization(CustomTestCase):
    @staticmethod
    def _model(visual, *, use_data_parallel):
        model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.visual = visual
        model.use_data_parallel = use_data_parallel
        return model

    def test_processor_defers_gpu_transport_for_encoder_dp(self):
        for transport in ("cuda_ipc", "cuda_vmm"):
            with self.subTest(transport=transport):
                processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
                processor.mm_feature_transport = transport
                processor.server_args = SimpleNamespace(mm_enable_dp_encoder=True)
                processor.model_type = "qwen3_vl"
                items = [
                    MultimodalDataItem(modality=Modality.IMAGE),
                    MultimodalDataItem(modality=Modality.VIDEO),
                    MultimodalDataItem(modality=Modality.AUDIO),
                ]

                processor._mark_dp_encoder_features_for_deferred_reconstruction(items)

                self.assertTrue(
                    items[0].model_specific_data[
                        DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY
                    ]
                )
                self.assertTrue(
                    items[1].model_specific_data[
                        DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY
                    ]
                )
                self.assertNotIn(
                    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
                    items[2].model_specific_data,
                )

    def test_processor_does_not_defer_cpu_transport(self):
        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        processor.mm_feature_transport = "cpu"
        processor.server_args = SimpleNamespace(mm_enable_dp_encoder=True)
        processor.model_type = "qwen3_vl"
        item = MultimodalDataItem(modality=Modality.IMAGE)

        processor._mark_dp_encoder_features_for_deferred_reconstruction([item])

        self.assertNotIn(
            DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
            item.model_specific_data,
        )

    def test_image_features_are_packed_on_the_visual_device(self):
        visual = _RecordingVisual()
        model = self._model(visual, use_data_parallel=False)
        items = [
            SimpleNamespace(
                feature=torch.ones(2, 3),
                image_grid_thw=torch.tensor([[1, 1, 2]]),
            ),
            SimpleNamespace(
                feature=torch.ones(1, 3),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
            ),
        ]
        output = model.get_image_feature(items)

        self.assertIs(visual.pixel_values, output)
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(output.device, visual.device)
        self.assertEqual(output.dtype, visual.dtype)

    def test_video_features_are_packed_on_the_visual_device(self):
        visual = _RecordingVisual()
        model = self._model(visual, use_data_parallel=False)
        items = [
            SimpleNamespace(
                feature=torch.ones(3, 4),
                video_grid_thw=torch.tensor([[1, 1, 3]]),
            ),
            SimpleNamespace(
                feature=torch.ones(2, 4),
                video_grid_thw=torch.tensor([[1, 1, 2]]),
            ),
        ]
        output = model.get_video_feature(items)

        self.assertIs(visual.pixel_values, output)
        self.assertEqual(output.shape, (5, 4))
        self.assertEqual(output.device, visual.device)
        self.assertEqual(output.dtype, visual.dtype)
        self.assertTrue(
            torch.equal(visual.grid_thw, torch.tensor([[1, 1, 3], [1, 1, 2]]))
        )

    def test_encoder_dp_materializes_only_locally_assigned_visual_items(self):
        visual = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16)
        model = self._model(visual, use_data_parallel=True)

        for modality, feature_method, grid_attribute in (
            ("image", model.get_image_feature, "image_grid_thw"),
            ("video", model.get_video_feature, "video_grid_thw"),
        ):
            with self.subTest(modality=modality):
                items = [
                    SimpleNamespace(
                        feature=torch.ones(2, 3),
                        reconstruct=Mock(),
                        **{grid_attribute: torch.tensor([[1, 1, 2]])},
                    ),
                    SimpleNamespace(
                        feature=torch.ones(1, 3),
                        reconstruct=Mock(),
                        **{grid_attribute: torch.tensor([[1, 1, 1]])},
                    ),
                ]
                local_features = object()
                encoded = object()

                def run_dp(_visual, pixel_values, grid_thw, **kwargs):
                    self.assertIsNone(pixel_values)
                    self.assertEqual(grid_thw, [[1, 1, 2], [1, 1, 1]])
                    self.assertIs(
                        kwargs["load_local_pixel_values"]([1]), local_features
                    )
                    return encoded

                with patch(
                    "sglang.srt.models.qwen3_vl.run_dp_sharded_mrope_vision_model",
                    side_effect=run_dp,
                ), patch(
                    "sglang.srt.models.qwen3_vl.materialize_multimodal_features",
                    return_value=local_features,
                ) as materialize, patch(
                    "sglang.srt.models.qwen3_vl.get_parallel",
                    return_value=SimpleNamespace(tp_size=8),
                ):
                    output = feature_method(items)

                self.assertIs(output, encoded)
                items[0].reconstruct.assert_not_called()
                items[1].reconstruct.assert_called_once_with(0, ipc_consumer_count=8)
                materialize.assert_called_once_with(
                    [items[1].feature], device=visual.device, dtype=visual.dtype
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
