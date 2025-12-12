import unittest
from unittest.mock import MagicMock

import openai
from test_vision_openai_server_common import (
    AUDIO_TRUMP_SPEECH_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    IMAGE_MAN_IRONING_URL,
    VIDEO_JOBS_URL,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.server_args import ServerArgs


class MultimodalTestProcessor(BaseMultimodalProcessor):
    models = []

    async def process_mm_data_async(self, *args, **kwargs):
        pass


class TestMMProcessConfigExtraction(unittest.TestCase):
    """Unit tests for mm_process_config extraction and passthrough."""

    # Define configs once - reuse across all tests
    IMAGE_CONFIG = {"max_pixels": 5000000, "min_pixels": 100}
    VIDEO_CONFIG = {"num_frames": 32, "fps": 2, "max_pixels": 128 * 28 * 28}
    AUDIO_CONFIG = {"sampling_rate": 16000}

    def _create_processor(self, mm_process_config):
        """Helper to create a processor with given config."""
        server_args = ServerArgs(model_path="dummy")
        server_args.mm_process_config = mm_process_config

        return MultimodalTestProcessor(
            hf_config=MagicMock(),
            server_args=server_args,
            _processor=MagicMock(),
            transport_mode=MagicMock(),
        )

    def test_extract_config_from_server_args(self):
        """Test that configs are extracted from server_args in __init__."""
        image_processor = self._create_processor({"image": self.IMAGE_CONFIG})
        video_processor = self._create_processor({"video": self.VIDEO_CONFIG})
        audio_processor = self._create_processor({"audio": self.AUDIO_CONFIG})

        self.assertEqual(image_processor.image_config, self.IMAGE_CONFIG)
        self.assertEqual(video_processor.video_config, self.VIDEO_CONFIG)
        self.assertEqual(audio_processor.audio_config, self.AUDIO_CONFIG)

    def test_image_config_passed_to_hf_processor(self):
        """Test that image_config is passed to HuggingFace processor in process_mm_data()."""
        processor = self._create_processor({"image": self.IMAGE_CONFIG})
        processor.process_mm_data(input_text="test", images=["dummy.png"])

        processor._processor.assert_called_once()
        call_kwargs = processor._processor.call_args.kwargs

        self.assertEqual(call_kwargs.get("max_pixels"), self.IMAGE_CONFIG["max_pixels"])
        self.assertEqual(call_kwargs.get("min_pixels"), self.IMAGE_CONFIG["min_pixels"])

    def test_video_config_passed_to_hf_processor(self):
        """Test that video_config is passed to HuggingFace processor in process_mm_data()."""
        processor = self._create_processor({"video": self.VIDEO_CONFIG})
        processor.process_mm_data(input_text="test", videos=["dummy.mp4"])

        processor._processor.assert_called_once()
        call_kwargs = processor._processor.call_args.kwargs

        self.assertEqual(call_kwargs.get("num_frames"), self.VIDEO_CONFIG["num_frames"])
        self.assertEqual(call_kwargs.get("fps"), self.VIDEO_CONFIG["fps"])
        self.assertEqual(call_kwargs.get("max_pixels"), self.VIDEO_CONFIG["max_pixels"])

    def test_audio_config_passed_to_hf_processor(self):
        """Test that audio_config is passed to HuggingFace processor in process_mm_data()."""
        processor = self._create_processor({"audio": self.AUDIO_CONFIG})
        processor.process_mm_data(input_text="test", audios=["dummy.wav"])

        processor._processor.assert_called_once()
        call_kwargs = processor._processor.call_args.kwargs

        self.assertEqual(
            call_kwargs.get("sampling_rate"), self.AUDIO_CONFIG["sampling_rate"]
        )


class TestMMProcessConfigIntegration(CustomTestCase):
    """Integration tests using MiniCPM-o-2_6 which supports image, video, and audio."""

    model = "openbmb/MiniCPM-o-2_6"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--enable-multimodal",
                "--disable-cuda-graph",
                "--mm-process-config",
                '{"image": {"max_slice_nums": 2}, "video": {"max_slice_nums": 1}, "audio": {"sampling_rate": 16000}}',
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_image_with_mm_process_config(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                        },
                        {"type": "text", "text": "Describe this image briefly"},
                    ],
                }
            ],
            temperature=0,
        )
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    def test_video_with_mm_process_config(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": VIDEO_JOBS_URL}},
                        {"type": "text", "text": "Describe this video briefly"},
                    ],
                }
            ],
            temperature=0,
        )
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    def test_audio_with_mm_process_config(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": AUDIO_TRUMP_SPEECH_URL},
                        },
                        {"type": "text", "text": "What is being said in this audio?"},
                    ],
                }
            ],
            temperature=0,
        )
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0


if __name__ == "__main__":
    unittest.main()
