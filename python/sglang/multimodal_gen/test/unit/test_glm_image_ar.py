import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _build_image_response_kwargs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
    GlmImageDecodingStage,
    center_crop_glm_image_output,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class _ProcessorInputs(dict):
    def to(self, device):
        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class _FakeProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return _ProcessorInputs(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 32, 32]], dtype=torch.long),
            }
        )


class _FakeResponse:
    def __init__(self, output_ids, meta_info=None):
        self._output_ids = output_ids
        self._meta_info = meta_info

    def raise_for_status(self):
        return None

    def json(self):
        data = {"output_ids": self._output_ids}
        if self._meta_info is not None:
            data["meta_info"] = self._meta_info
        return data


class _FakeBatchResponse:
    def __init__(self, outputs):
        self._outputs = outputs

    def raise_for_status(self):
        return None

    def json(self):
        return self._outputs


class TestGlmImageARSrtBackend(unittest.TestCase):
    def _server_args(self):
        return SimpleNamespace(
            srt_encoder_url="http://127.0.0.1:8764",
            srt_encoder_connect_timeout=3.05,
            srt_encoder_timeout=100,
        )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.requests.post"
    )
    def test_srt_ar_uses_ignore_eos_for_fixed_length_tokens(
        self, mock_post, _mock_device
    ):
        set_global_server_args(self._server_args())
        mock_post.return_value = _FakeResponse(list(range(1025)))
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)

        prior_token_ids, _, usage = stage.generate_prior_tokens(
            prompt="A simple product sketch",
            height=1024,
            width=1024,
            server_args=self._server_args(),
        )

        payload = mock_post.call_args.kwargs["json"]
        self.assertTrue(payload["sampling_params"]["ignore_eos"])
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 1025)
        self.assertEqual(prior_token_ids.shape, (1, 4096))
        self.assertIsNone(usage)

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.requests.post"
    )
    def test_srt_ar_extracts_usage_from_meta_info(self, mock_post, _mock_device):
        set_global_server_args(self._server_args())
        mock_post.return_value = _FakeResponse(
            list(range(1025)),
            meta_info={
                "prompt_tokens": 13,
                "completion_tokens": 25,
                "reasoning_tokens": 0,
                "cached_tokens": 5,
            },
        )
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)

        _, _, usage = stage.generate_prior_tokens(
            prompt="A simple product sketch",
            height=1024,
            width=1024,
            server_args=self._server_args(),
        )

        self.assertEqual(
            usage,
            {
                "prompt_tokens": 13,
                "completion_tokens": 25,
                "reasoning_tokens": 0,
                "cached_tokens": 5,
                "total_tokens": 38,
            },
        )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.requests.post"
    )
    def test_srt_ar_forward_aggregates_usage(self, mock_post, _mock_device):
        set_global_server_args(self._server_args())
        mock_post.return_value = _FakeBatchResponse(
            [
                {
                    "output_ids": list(range(1025)),
                    "meta_info": {"prompt_tokens": 13, "completion_tokens": 25},
                },
                {
                    "output_ids": list(range(1025)),
                    "meta_info": {"prompt_tokens": 13, "completion_tokens": 25},
                },
            ]
        )
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)
        batch = SimpleNamespace(
            prompt="A simple product sketch",
            height=1025,
            width=1001,
            image_path=None,
            num_outputs_per_prompt=2,
            seed=None,
            extra={},
        )

        batch = stage.forward(batch, self._server_args())

        self.assertEqual(batch.usage["prompt_tokens"], 26)
        self.assertEqual(batch.usage["completion_tokens"], 50)
        self.assertEqual(batch.usage["total_tokens"], 76)

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.requests.post"
    )
    def test_srt_ar_rejects_short_output_ids(self, mock_post, _mock_device):
        set_global_server_args(self._server_args())
        mock_post.return_value = _FakeResponse(list(range(993)))
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)

        with self.assertRaisesRegex(
            RuntimeError,
            "GLM-Image AR returned too few output_ids: got 993, need at least 1024",
        ):
            stage.generate_prior_tokens(
                prompt="A simple product sketch",
                height=1024,
                width=1024,
                server_args=self._server_args(),
            )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_forward_aligns_runtime_dimensions_before_ar_generation(self, _mock_device):
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)
        stage.generate_prior_tokens = MagicMock(
            return_value=(torch.zeros((1, 1), dtype=torch.long), None, None)
        )
        cases = [
            ((500, 500), (512, 512)),
            ((550, 1009), (576, 1024)),
            ((1280, 720), (1280, 736)),
        ]

        for requested, expected in cases:
            with self.subTest(requested=requested):
                stage.generate_prior_tokens.reset_mock()
                sampling = GlmImageSamplingParams(
                    prompt="A simple product sketch",
                    width=requested[0],
                    height=requested[1],
                )
                sampling.seed = None
                batch = Req(sampling_params=sampling)

                stage.forward(batch, self._server_args())

                self.assertEqual((batch.width, batch.height), expected)
                self.assertEqual(
                    (batch.requested_width, batch.requested_height), requested
                )
                stage.generate_prior_tokens.assert_called_once_with(
                    prompt="A simple product sketch",
                    image=None,
                    height=expected[1],
                    width=expected[0],
                    server_args=self._server_args(),
                )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.load_image",
        return_value=Image.new("RGB", (1280, 720)),
    )
    def test_forward_resizes_edit_image_up_to_d32_grid(
        self, _mock_load_image, _mock_device
    ):
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)
        stage.generate_prior_tokens = MagicMock(
            return_value=(torch.zeros((1, 1), dtype=torch.long), None, None)
        )
        sampling = GlmImageSamplingParams(
            prompt="Edit this image",
            width=1280,
            height=720,
            image_path="input.png",
        )
        sampling.seed = None
        batch = Req(sampling_params=sampling)

        stage.forward(batch, self._server_args())

        call_kwargs = stage.generate_prior_tokens.call_args.kwargs
        self.assertEqual((batch.width, batch.height), (1280, 736))
        self.assertEqual((batch.requested_width, batch.requested_height), (1280, 720))
        self.assertEqual(call_kwargs["image"][0].size, (1280, 736))
        self.assertEqual((call_kwargs["width"], call_kwargs["height"]), (1280, 736))

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.load_image",
        return_value=Image.new("RGB", (1280, 720)),
    )
    def test_forward_preserves_implicit_edit_image_size(
        self, _mock_load_image, _mock_device
    ):
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)
        stage.generate_prior_tokens = MagicMock(
            return_value=(torch.zeros((1, 1), dtype=torch.long), None, None)
        )
        sampling = GlmImageSamplingParams(
            prompt="Edit this image",
            image_path="input.png",
        )
        sampling.seed = None
        batch = Req(sampling_params=sampling)

        stage.forward(batch, self._server_args())

        self.assertEqual((batch.width, batch.height), (1280, 736))
        self.assertEqual((batch.requested_width, batch.requested_height), (1280, 720))
        call_kwargs = stage.generate_prior_tokens.call_args.kwargs
        self.assertEqual(call_kwargs["image"][0].size, (1280, 736))
        self.assertEqual((call_kwargs["width"], call_kwargs["height"]), (1280, 736))

    def test_center_crop_restores_requested_size(self):
        frames = torch.arange(1024 * 1024).reshape(1, 1, 1024, 1024)

        cropped = center_crop_glm_image_output(frames, 1000, 999)

        self.assertEqual(tuple(cropped.shape), (1, 1, 999, 1000))
        self.assertEqual(cropped[0, 0, 0, 0], frames[0, 0, 12, 12])
        self.assertEqual(cropped[0, 0, -1, -1], frames[0, 0, 1010, 1011])
        self.assertTrue(cropped.is_contiguous())

    @patch.object(DecodingStage, "forward")
    def test_decoding_stage_crops_outputs_and_trajectory(self, mock_decode):
        frames = torch.zeros((2, 3, 736, 1280))
        trajectory = [
            torch.zeros((2, 3, 1, 736, 1280)),
            torch.ones((2, 3, 1, 736, 1280)),
        ]
        mock_decode.return_value = OutputBatch(
            output=frames,
            trajectory_decoded=trajectory,
        )
        stage = GlmImageDecodingStage(vae=None)
        sampling = GlmImageSamplingParams(width=1280, height=736)
        sampling.requested_width = 1280
        sampling.requested_height = 720
        batch = Req(sampling_params=sampling)

        output_batch = stage.forward(batch, self._server_args())

        self.assertEqual(tuple(output_batch.output.shape), (2, 3, 720, 1280))
        self.assertEqual(len(output_batch.trajectory_decoded), 2)
        for decoded in output_batch.trajectory_decoded:
            self.assertEqual(tuple(decoded.shape), (2, 3, 1, 720, 1280))
        mock_decode.assert_called_once_with(batch, self._server_args())

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_generate_prior_tokens_rejects_unaligned_internal_dimensions(
        self, _mock_device
    ):
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)

        with self.assertRaisesRegex(
            ValueError,
            "GLM-Image dimensions must be aligned before AR token generation",
        ):
            stage.generate_prior_tokens(
                prompt="A simple product sketch",
                height=1024,
                width=550,
                server_args=self._server_args(),
            )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.current_platform.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_generate_prior_tokens_batch_rejects_unaligned_internal_dimensions(
        self, _mock_device
    ):
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)

        with self.assertRaisesRegex(
            ValueError,
            "GLM-Image dimensions must be aligned before AR token generation",
        ):
            stage.generate_prior_tokens_batch(
                prompts=["A simple product sketch"],
                seeds=[42],
                height=1024,
                width=550,
                server_args=self._server_args(),
            )

    def test_image_response_adds_image_count_to_usage(self):
        set_global_server_args(SimpleNamespace(enable_cache_report=False))
        response = _build_image_response_kwargs(
            ["/tmp/glm-image-0.jpg", "/tmp/glm-image-1.jpg"],
            "b64_json",
            "A simple product sketch",
            "req-0",
            OutputBatch(
                usage={
                    "prompt_tokens": 13,
                    "completion_tokens": 25,
                    "total_tokens": 38,
                    "reasoning_tokens": 0,
                    "cached_tokens": 5,
                }
            ),
            b64_list=["aGVsbG8=", "d29ybGQ="],
            is_persistent=False,
        )

        self.assertEqual(response["usage"]["image_count"], 2)
        self.assertNotIn("prompt_tokens_details", response["usage"])
        self.assertNotIn("cached_tokens", response["usage"])

    def test_image_response_reports_cached_tokens_when_cache_report_enabled(self):
        set_global_server_args(SimpleNamespace(enable_cache_report=True))
        response = _build_image_response_kwargs(
            ["/tmp/glm-image-0.jpg"],
            "b64_json",
            "A simple product sketch",
            "req-0",
            OutputBatch(
                usage={
                    "prompt_tokens": 13,
                    "completion_tokens": 25,
                    "total_tokens": 38,
                    "reasoning_tokens": 0,
                    "cached_tokens": 5,
                }
            ),
            b64_list=["aGVsbG8="],
            is_persistent=False,
        )

        self.assertEqual(
            response["usage"]["prompt_tokens_details"], {"cached_tokens": 5}
        )
        self.assertNotIn("cached_tokens", response["usage"])


if __name__ == "__main__":
    unittest.main()
