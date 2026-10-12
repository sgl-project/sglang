import asyncio
import concurrent.futures
import re
import types
import unittest
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.dots_note_omni import DotsNoteOmniProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_IM_TOKEN_ID = 100
_AUDIO_TOKEN_ID = 200
# The dots.note chat template renders a video content part as this single token.
_VIDEO_PLACEHOLDER = "<|video_pad|>"


class _FakeMultimodalTokens:
    _pattern = re.compile(r"(<image>|<audio>)")

    def get_combined_regex(self):
        return self._pattern

    def get_modality_of_token(self, token):
        return {
            "<image>": Modality.IMAGE,
            "<audio>": Modality.AUDIO,
        }.get(token)


class _FakeTokenizer:
    """Maps each media marker to one pad id and other text to per-char ids."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in _FakeMultimodalTokens._pattern.split(text):
            if part == "<image>":
                ids.append(_IM_TOKEN_ID)
            elif part == "<audio>":
                ids.append(_AUDIO_TOKEN_ID)
            else:
                ids.extend(ord(char) for char in part)
        return ids


def _fake_preprocess_dots_video(raw_video, question, **kwargs):
    """Each video flattens into one frame, one audio segment and the question."""
    return [
        {"type": "image_url", "image_url": {"url": f"{raw_video}-frame"}},
        {"type": "audio_url", "audio_url": {"url": f"{raw_video}-audio"}},
        {"type": "text", "text": question},
    ]


class TestDotsNoteOmniVideoMixing(CustomTestCase):
    def setUp(self):
        self.processor = DotsNoteOmniProcessor.__new__(DotsNoteOmniProcessor)
        self.processor.image_start_token = ""
        self.processor.image_token = "<image>"
        self.processor.image_end_token = ""
        self.processor.audio_start_token = ""
        self.processor.audio_token = "<audio>"
        self.processor.audio_end_token = ""
        self.processor.mm_tokens = _FakeMultimodalTokens()
        self.processor.video_placeholder_regex = re.compile(
            re.escape(_VIDEO_PLACEHOLDER)
        )

    def test_multiple_videos_and_native_media_keep_prompt_order(self):
        prompt = f"{_VIDEO_PLACEHOLDER}<image>between{_VIDEO_PLACEHOLDER}question"
        all_video_media = {}
        contents = [
            [
                {"type": "image_url", "image_url": {"url": "video-0-frame"}},
                {"type": "text", "text": "question"},
            ],
            [
                {"type": "audio_url", "audio_url": {"url": "video-1-audio"}},
                {"type": "text", "text": "question"},
            ],
        ]

        for index, content in enumerate(contents):
            prompt, video_media = self.processor._render_video_content(
                prompt, "question", index, content
            )
            all_video_media.update(video_media)

        prompt, images, audios = self.processor._merge_video_media(
            prompt,
            image_data=["native-image"],
            audio_data=None,
            video_media=all_video_media,
        )

        self.assertEqual(prompt, "<image><image>between<audio>question")
        self.assertEqual(images, ["video-0-frame", "native-image"])
        self.assertEqual(audios, ["video-1-audio"])

    def test_template_without_video_placeholders_inserts_each_video_once(self):
        prompt = "<|user|>question"
        all_video_media = {}

        for index in range(2):
            prompt, video_media = self.processor._render_video_content(
                prompt,
                "question",
                index,
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"video-{index}-frame"},
                    },
                    {"type": "text", "text": "question"},
                ],
            )
            all_video_media.update(video_media)

        prompt, images, audios = self.processor._merge_video_media(
            prompt, None, None, all_video_media
        )

        self.assertEqual(prompt, "<|user|><image><image>question")
        self.assertEqual(images, ["video-0-frame", "video-1-frame"])
        self.assertEqual(audios, [])


class TestDotsNoteOmniProcessMmDataAsync(CustomTestCase):
    """Drive the request entry point that used to reject mixed video inputs."""

    def setUp(self):
        self.processor = DotsNoteOmniProcessor.__new__(DotsNoteOmniProcessor)
        self.processor.image_start_token = ""
        self.processor.image_token = "<image>"
        self.processor.image_end_token = ""
        self.processor.audio_start_token = ""
        self.processor.audio_token = "<audio>"
        self.processor.audio_end_token = ""
        self.processor.mm_tokens = _FakeMultimodalTokens()
        self.processor.video_placeholder_regex = re.compile(
            re.escape(_VIDEO_PLACEHOLDER)
        )
        self.processor.mm_token_ids = {
            "im_start_id": 98,
            "im_token_id": _IM_TOKEN_ID,
            "im_end_id": 99,
            "audio_start_id": 198,
            "audio_token_id": _AUDIO_TOKEN_ID,
            "audio_end_id": 199,
        }
        self.processor._tokenizer = _FakeTokenizer()
        self.processor.audio_processor_config = types.SimpleNamespace(
            sampling_rate=16000
        )
        self.processor.image_preprocessor = types.SimpleNamespace(
            process_images=lambda images: (
                [torch.tensor([float(index)]) for index in range(len(images))],
                [torch.tensor([1, 1, 4]) for _ in images],
                ["<image>" for _ in images],
            )
        )
        self.processor.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        )
        self.addCleanup(self.processor.io_executor.shutdown)

        self.loaded = {}

        async def fake_load_mm_data(prompt, image_data=None, audio_data=None, **kwargs):
            self.loaded["prompt"] = prompt
            self.loaded["image_data"] = list(image_data or [])
            self.loaded["audio_data"] = list(audio_data or [])
            return types.SimpleNamespace(
                input_text=prompt,
                images=list(image_data or []),
                audios=[torch.zeros(4) for _ in audio_data or []],
            )

        self.processor.load_mm_data = fake_load_mm_data

    def _run(self, request_obj, **kwargs):
        with (
            patch(
                "sglang.srt.multimodal.processors.dots_note_omni.preprocess_dots_video",
                _fake_preprocess_dots_video,
            ),
            patch(
                "sglang.srt.multimodal.processors.dots_note_omni.get_audio_token_string",
                lambda *args, **kwargs: "<audio>",
            ),
        ):
            return asyncio.run(
                self.processor.process_mm_data_async(
                    request_obj.text,
                    request_obj,
                    max_req_input_len=4096,
                    **kwargs,
                )
            )

    @staticmethod
    def _request(text, video_data):
        return types.SimpleNamespace(
            text=text,
            video_data=video_data,
            video_config={
                "_question": "question",
                "seq": 131072,
                "audio_cap": 1.0,
                "audio_sr": 16000,
                "k_mode": "eval_ek",
            },
            sampling_params={"max_new_tokens": 16},
            rid="test-rid",
        )

    def test_video_mixed_with_image_and_audio(self):
        request_obj = self._request(
            f"<image>{_VIDEO_PLACEHOLDER}middle<audio>question", ["video-0"]
        )

        output = self._run(
            request_obj, image_data=["native-image"], audio_data=["native-audio"]
        )

        self.assertEqual(
            self.loaded["prompt"], "<image><image><audio>middle<audio>question"
        )
        self.assertEqual(self.loaded["image_data"], ["native-image", "video-0-frame"])
        self.assertEqual(self.loaded["audio_data"], ["video-0-audio", "native-audio"])
        self.assertEqual(
            [item.modality for item in output.mm_items],
            [
                Modality.IMAGE,
                Modality.IMAGE,
                Modality.AUDIO,
                Modality.AUDIO,
            ],
        )

    def test_multiple_videos_mixed_with_image(self):
        request_obj = self._request(
            f"{_VIDEO_PLACEHOLDER}<image>middle{_VIDEO_PLACEHOLDER}question",
            ["video-0", "video-1"],
        )

        output = self._run(request_obj, image_data=["native-image"])

        self.assertEqual(
            self.loaded["prompt"],
            "<image><audio><image>middle<image><audio>question",
        )
        self.assertEqual(
            self.loaded["image_data"],
            ["video-0-frame", "native-image", "video-1-frame"],
        )
        self.assertEqual(self.loaded["audio_data"], ["video-0-audio", "video-1-audio"])
        self.assertEqual(len(output.mm_items), 5)

    def test_extra_native_image_without_placeholder_is_rejected(self):
        request_obj = self._request(f"{_VIDEO_PLACEHOLDER}question", ["video-0"])

        with self.assertRaisesRegex(ValueError, "Image placeholder count"):
            self._run(request_obj, image_data=["native-image"])

    def test_unconsumed_video_placeholder_is_rejected(self):
        request_obj = self._request(
            f"{_VIDEO_PLACEHOLDER}{_VIDEO_PLACEHOLDER}question", ["video-0"]
        )

        with self.assertRaisesRegex(ValueError, "Video placeholder count"):
            self._run(request_obj)


if __name__ == "__main__":
    unittest.main()
