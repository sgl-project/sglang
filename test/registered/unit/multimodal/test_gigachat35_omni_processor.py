"""Unit tests for the GigaChat3.5-Omni placeholder layout, with a fake checkpoint processor."""

import asyncio
import math
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests
import torch

from sglang.srt.configs.gigachat35_omni import GigaChat35OmniConfig
from sglang.srt.multimodal.gigachat35_omni import GigaChat35OmniVideoEmbedding
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.gigachat35_omni import (
    GigaChat35OmniMultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

IMG, VID, AUD, VA = 128004, 128005, 128006, 128007


class TestGigaChat35OmniProcessor(CustomTestCase):
    def setUp(self):
        # What the checkpoint's processor returns per input: image crops, video
        # timestamps and soundtrack samples, audio samples and duration. Mel
        # frames are 160 samples, one token per 16 mel frames.
        self.images = {b"image": 3, b"training_image": 7}
        self.videos = {
            b"no_soundtrack": ([0.0, 1.5], 100),
            b"soundtrack": ([0.0, 1.0, 2.0], 320 * 160),
            b"min_soundtrack": ([0.0], 320),
            b"short_soundtrack": ([0.0], 319),
            # 25 frames sampled from 30 fps at 1 fps, 2499 mel frames.
            b"training_video": ([31 * i / 30 for i in range(25)], 2499 * 160),
        }
        self.audios = {
            b"audio": (2.5 * 16000, 2.5),
            b"training_audio": (7498 * 160, 74.99986394557823),
        }
        self.processor = self._build(video_tokens_per_frame=77)

    def _build(self, video_tokens_per_frame):
        """Builds the in-tree processor around a fake checkpoint processor.

        Args:
            video_tokens_per_frame: Tokens kept per later video frame.

        Returns:
            The processor.
        """
        remote = SimpleNamespace(
            tokenizer=SimpleNamespace(bos_token="<s>", encode=self._encode),
            preprocess_image=lambda image: torch.zeros(self.images[image], 1),
            load_video=lambda source, use_audio: {
                "pixel_values": torch.zeros(len(self.videos[source][0]), 1),
                "timestamps": self.videos[source][0],
                "audio": np.zeros(self.videos[source][1]) if use_audio else None,
            },
            load_audio=lambda source: (
                np.zeros(int(self.audios[source][0])),
                self.audios[source][1],
            ),
            extract_mel=lambda waveform: torch.zeros(len(waveform) // 160, 64),
            num_audio_tokens=lambda num_mel_frames: math.ceil(num_mel_frames / 16),
        )
        config = GigaChat35OmniConfig(
            vision_config={"image_size": 448, "patch_size": 14},
            video_tokens_per_frame=video_tokens_per_frame,
        )

        with patch.object(
            BaseMultimodalProcessor,
            "__init__",
            lambda self, hf_config, server_args, _processor, *args: vars(self).update(
                _processor=_processor, _tokenizer=_processor.tokenizer
            ),
        ):
            processor = GigaChat35OmniMultimodalProcessor(config, None, remote, None)

        processor.io_executor = ThreadPoolExecutor(2)
        self.addCleanup(processor.io_executor.shutdown)
        return processor

    def _encode(self, text, add_special_tokens=False):
        """Tokenizes like the GigaChat3.5 tokenizer: `<s>` and `<img>` are special
        tokens, any other character is one token.

        Args:
            text: Text.
            add_special_tokens: Whether to prepend the BOS.

        Returns:
            Token ids, `0`-`9` = 18-27 and `:` = 28 as in the real tokenizer.
        """
        ids = [1] if add_special_tokens else []

        for piece in re.split("(<s>|<img>)", text):
            ids += {"<s>": [1], "<img>": [IMG]}.get(piece, [ord(c) - 30 for c in piece])

        return ids

    def _process(self, prompt, use_audio=True, **media):
        """Runs the processor on a tokenized prompt.

        Args:
            prompt: Prompt token ids with one placeholder per media input.
            use_audio: The request's `use_audio_in_video`.
            **media: `image_data`, `audio_data` and `video_data`.

        Returns:
            The processor output.
        """
        request = SimpleNamespace(
            video_data=media.get("video_data"), use_audio_in_video=use_audio
        )
        module = "sglang.srt.multimodal.processors.gigachat35_omni"

        with patch(f"{module}.load_image", side_effect=lambda data, **_: (data, None)):
            return asyncio.run(
                self.processor.process_mm_data_async(
                    media.get("image_data"), media.get("audio_data"), prompt, request
                )
            )

    def test_layouts_match_training_before_pruning(self):
        """Stamps, frame runs and the soundtrack q/r split are training's.

        The fixtures are the training processors' layouts of a 1024x664 image,
        a 25 s video with and without its soundtrack and a 75 s audio clip. With
        a budget of 256 tokens per later frame nothing is pruned, so the prompt
        must equal training's token for token.
        """
        self.processor = self._build(video_tokens_per_frame=256)
        video_audio = []

        for i in range(25):
            video_audio += self._encode(f"00:00:{i:02d}") + [VID] * 256
            video_audio += [VA] * (7 if i < 7 else 6)

        audio = self._encode("00:00:00") + [AUD] * 375 + self._encode("00:01:00")
        audio += [AUD] * 94 + self._encode("00:01:14")
        cases = {
            "image": (IMG, {"image_data": [b"training_image"]}, True, [IMG] * 1792),
            "video": (VID, {"video_data": [b"training_video"]}, False, [VID] * 6400),
            "video_audio": (
                VID,
                {"video_data": [b"training_video"]},
                True,
                video_audio,
            ),
            "audio": (AUD, {"audio_data": [b"training_audio"]}, True, audio),
        }

        for name, (token, media, use_audio, ids) in cases.items():
            with self.subTest(name=name):
                result = self._process([1, 1000, token, 2000], use_audio, **media)
                self.assertEqual(result.input_ids, [1, 1000] + ids + [2000])

    def test_audio_offsets_skip_the_minute_stamps(self):
        """Exactly one minute gets one stamp, a clip under one second none."""
        stamp = self._encode
        cases = {
            "under one second": (64, 0.4, [AUD] * 4 + stamp("00:00:00"), [(0, 3)]),
            "one minute": (
                5998,
                59.99,
                stamp("00:00:00") + [AUD] * 375 + stamp("00:00:59"),
                [(8, 382)],
            ),
            "over one minute": (
                7498,
                74.99,
                stamp("00:00:00")
                + [AUD] * 375
                + stamp("00:01:00")
                + [AUD] * 94
                + stamp("00:01:14"),
                [(8, 382), (391, 484)],
            ),
        }

        for name, (num_mel_frames, duration, ids, offsets) in cases.items():
            with self.subTest(name=name):
                input_ids, items = [], []
                mel = torch.zeros(num_mel_frames, 64)
                self.processor.expand_audio(mel, duration, input_ids, items)
                self.assertEqual(input_ids, ids)
                self.assertEqual(items[0].offsets, offsets)

    def test_seconds_to_hms_truncates_to_whole_seconds(self):
        self.assertEqual(self.processor.seconds_to_hms(3725.9), "01:02:05")

    def test_layout_of_every_modality_in_prompt_order(self):
        prompt = [1, IMG, 2, VID, 3, VID, 4, AUD, 5]
        media = dict(
            image_data=[b"image"],
            video_data=[b"no_soundtrack", b"soundtrack"],
            audio_data=[b"audio"],
        )
        result = self._process(prompt, **media)

        # Budget 256 + 77 * (T - 1) split evenly; 20 soundtrack tokens over 3 frames.
        soundtrack_video = []

        for second, frame, audio in zip(range(3), [136, 136, 138], [7, 7, 6]):
            soundtrack_video += (
                self._encode(f"00:00:0{second}") + [VID] * frame + [VA] * audio
            )

        expected = (
            [1]
            + [IMG] * 768
            + [2]
            + [VID] * 333
            + [3]
            + soundtrack_video
            + [4]
            + self._encode("00:00:00")
            + [AUD] * 16
            + self._encode("00:00:02")
            + [5]
        )
        self.assertEqual(result.input_ids, expected)

        items = result.mm_items
        self.assertEqual(
            [item.modality for item in items],
            [
                Modality.IMAGE,
                Modality.VIDEO,
                Modality.VIDEO,
                Modality.AUDIO,
                Modality.AUDIO,
            ],
        )
        self.assertEqual(items[1].thw_grids, [(2, 16, 16)])
        self.assertEqual(items[2].thw_grids, [(3, 16, 16)])
        self.assertEqual(items[3].soundtrack_of, 1)
        self.assertNotIn(
            GigaChat35OmniVideoEmbedding.SOUNDTRACK_OF_KEY, items[4].model_specific_data
        )
        self.assertEqual([e - s + 1 for s, e in items[2].offsets], [136, 136, 138])
        self.assertEqual([e - s + 1 for s, e in items[3].offsets], [7, 7, 6])

        for item, token, count in zip(
            items, [IMG, VID, VID, VA, AUD], [768, 333, 410, 20, 16]
        ):
            covered = [i for start, end in item.offsets for i in range(start, end + 1)]
            self.assertEqual(len(covered), count)
            self.assertEqual({expected[i] for i in covered}, {token})

        # Without use_audio_in_video a video is one frame-token run with no stamps.
        result = self._process(prompt, use_audio=False, **media)
        no_audio = (
            expected[: expected.index(3) + 1]
            + [VID] * 410
            + expected[expected.index(4) :]
        )
        self.assertEqual(result.input_ids, no_audio)
        self.assertEqual(len(result.mm_items), 4)

    def test_soundtrack_shorter_than_one_mel_frame_is_dropped(self):
        for source, has_mel in [
            (b"short_soundtrack", False),
            (b"min_soundtrack", True),
        ]:
            with self.subTest(source=source):
                video = self.processor.load_video(source, use_audio=True)
                self.assertEqual(video["mel"] is not None, has_mel)

    def test_text_prompt_gets_exactly_one_bos(self):
        for prompt in ("<s>a<img>", "a<img>"):
            with self.subTest(prompt=prompt):
                result = self._process(prompt, image_data=[b"image"])
                self.assertEqual(result.input_ids, [1, ord("a") - 30] + [IMG] * 768)

    def test_client_errors_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "More media placeholders"):
            self._process([VID, VID], video_data=[b"soundtrack"])

        with self.assertRaisesRegex(ValueError, "More media items"):
            self._process([1, 2], image_data=[b"image"])

        with self.assertRaisesRegex(ValueError, "Soundtrack placeholders"):
            self._process([1, VA, VID], video_data=[b"soundtrack"])

        refused = requests.exceptions.ConnectionError("refused")

        with patch.object(self.processor._processor, "load_audio", side_effect=refused):
            with self.assertRaisesRegex(ValueError, "Cannot load media: refused"):
                self._process([AUD], audio_data=[b"audio"])


if __name__ == "__main__":
    unittest.main()
