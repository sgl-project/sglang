from sglang.srt.multimodal.processors.mimo_v2 import MiMoProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMiMoVideoAudioUnits(CustomTestCase):
    def test_audio_shorter_than_video_keeps_video_tail(self):
        processor = object.__new__(MiMoProcessor)
        processor.temporal_patch_size = 2
        processor.temporal_compression_ratio = 1
        processor.merge_size = 2
        processor.use_video_timestamps = True
        processor.audio_token_per_second = 2

        units = processor._build_video_audio_units(
            thw_grid=(3, 4, 4),
            timestamps=[0, 0.5, 1, 1.5, 2, 2.5],
            video_meta={"segment_end_time": 3},
            processed_audio=[10, 11, 12],
            is_tokenized=True,
            audio_token_len=3,
        )

        self.assertEqual([unit["segment_audio_token_len"] for unit in units], [2, 1, 0])
        self.assertEqual([unit["num_video_tokens"] for unit in units], [4, 4, 4])
        self.assertEqual([unit["audio_start_token_idx"] for unit in units], [0, 2, 4])
        self.assertEqual(units[0]["segment_audio"], [10, 11])
        self.assertEqual(units[1]["segment_audio"], [12])
        self.assertIsNone(units[2]["segment_audio"])
