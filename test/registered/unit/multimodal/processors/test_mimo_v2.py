"""Unit tests for the MiMo-V2 multimodal processor."""

import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiMoProcessor(CustomTestCase):
    def test_video_units_allow_audio_to_end_before_video(self):
        torchcodec = ModuleType("torchcodec")
        torchcodec.__path__ = []
        torchcodec_decoders = ModuleType("torchcodec.decoders")
        torchcodec_decoders.AudioDecoder = object
        torchcodec.decoders = torchcodec_decoders
        with patch.dict(
            sys.modules,
            {
                "torchcodec": torchcodec,
                "torchcodec.decoders": torchcodec_decoders,
            },
        ):
            mimo_v2 = importlib.import_module(
                "sglang.srt.multimodal.processors.mimo_v2"
            )

        MiMoProcessor = mimo_v2.MiMoProcessor
        processor = object.__new__(MiMoProcessor)
        processor.temporal_patch_size = 1
        processor.temporal_compression_ratio = 1
        processor.use_video_timestamps = True
        processor.audio_token_per_second = 1
        processor.merge_size = 1

        units = processor._build_video_audio_units(
            thw_grid=(3, 1, 1),
            timestamps=[0.0, 1.0, 2.0],
            video_meta={"segment_end_time": 3.0},
            processed_audio=[42],
            is_tokenized=True,
            audio_token_len=1,
        )

        self.assertEqual([unit["segment_audio_token_len"] for unit in units], [1, 0, 0])
        self.assertEqual([unit["segment_audio"] for unit in units], [[42], None, None])


if __name__ == "__main__":
    unittest.main()
