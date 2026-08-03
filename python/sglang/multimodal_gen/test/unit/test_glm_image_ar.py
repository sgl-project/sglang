import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
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
    def __init__(self, output_ids):
        self._output_ids = output_ids

    def json(self):
        return {"output_ids": self._output_ids}


class TestGlmImageARSrtBackend(unittest.TestCase):
    def _server_args(self):
        return SimpleNamespace(
            srt_encoder_url="http://127.0.0.1:8764",
            srt_encoder_connect_timeout=3.05,
            srt_encoder_timeout=100,
        )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.get_local_torch_device",
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

        prior_token_ids, _ = stage.generate_prior_tokens(
            prompt="A simple product sketch",
            height=1024,
            width=1024,
            server_args=self._server_args(),
        )

        payload = mock_post.call_args.kwargs["json"]
        self.assertTrue(payload["sampling_params"]["ignore_eos"])
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 1025)
        self.assertEqual(prior_token_ids.shape, (1, 4096))

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.get_local_torch_device",
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
        "model_specific_stages.glm_image.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_forward_aligns_runtime_dimensions_before_ar_generation(self, _mock_device):
        stage = GlmImageAR(processor=_FakeProcessor(), vision_language_encoder=None)
        stage.generate_prior_tokens = MagicMock(
            return_value=(torch.zeros((1, 1), dtype=torch.long), None)
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
                stage.generate_prior_tokens.assert_called_once_with(
                    prompt="A simple product sketch",
                    image=None,
                    height=expected[1],
                    width=expected[0],
                    server_args=self._server_args(),
                )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.get_local_torch_device",
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
            return_value=(torch.zeros((1, 1), dtype=torch.long), None)
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
        self.assertEqual(call_kwargs["image"][0].size, (1280, 736))
        self.assertEqual((call_kwargs["width"], call_kwargs["height"]), (1280, 736))

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.glm_image.get_local_torch_device",
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


if __name__ == "__main__":
    unittest.main()
