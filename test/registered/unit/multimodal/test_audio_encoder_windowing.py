from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature, hash_mm_item
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.audio_encoder_windowing import (
    AudioEncoderWindowSpec,
    build_audio_encoder_window_items,
    resolve_audio_encoder_window_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FeatureExtractor:
    hop_length = 2
    n_fft = 4
    sampling_rate = 16

    def __call__(self, windows, **kwargs):
        lengths = [max(len(window) // self.hop_length, 1) for window in windows]
        features = torch.zeros(len(windows), 2, max(lengths))
        masks = torch.zeros(len(windows), max(lengths), dtype=torch.long)
        for index, (window, length) in enumerate(zip(windows, lengths)):
            features[index, 0, :length] = torch.from_numpy(
                window[: length * self.hop_length : self.hop_length]
            )
            masks[index, :length] = 1
        return {"input_features": features, "attention_mask": masks}


class TestAudioEncoderWindowing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = SimpleNamespace(
            feature_extractor=_FeatureExtractor(),
            _get_feat_extract_output_lengths=lambda lengths: lengths,
        )
        cls.config = resolve_audio_encoder_window_config(
            AudioEncoderWindowSpec(
                window_frames=8,
                alignment_frames=4,
            ),
            processor=cls.processor,
            output_length_fn=cls.processor._get_feat_extract_output_lengths,
            model_sample_rate=16,
        )

    def _build(self, sample_count):
        return build_audio_encoder_window_items(
            samples=np.arange(sample_count, dtype=np.float32),
            input_ids=torch.tensor([10, 99, 11]),
            placeholder_token_id=99,
            config=self.config,
            processor=self.processor,
            output_length_fn=self.processor._get_feat_extract_output_lengths,
        )

    def test_complete_windows_keep_their_cache_identity_after_append(self):
        first_items, _ = self._build(36)
        appended_items, _ = self._build(40)

        self.assertEqual(
            [item.use_embedding_cache for item in first_items], [True, True, False]
        )
        self.assertEqual(
            [item.hash for item in first_items[:2]],
            [item.hash for item in appended_items[:2]],
        )
        self.assertEqual(
            first_items[0].hash,
            hash_mm_item(
                hash_feature(
                    [
                        first_items[0].feature,
                        first_items[0].model_specific_data["feature_attention_mask"],
                    ]
                ),
                Modality.AUDIO,
                first_items[0].offsets,
            ),
        )

    def test_placeholder_layout_keeps_the_tail_uncached(self):
        items, input_ids = self._build(36)
        tiny_tail_items, _ = self._build(33)

        self.assertEqual(
            [item.offsets for item in items],
            [[(1, 8)], [(9, 16)], [(17, 18)]],
        )
        self.assertEqual(int((input_ids == 99).sum()), 18)
        self.assertEqual(tiny_tail_items[-1].offsets, [(17, 18)])
        self.assertFalse(tiny_tail_items[-1].use_embedding_cache)

    def test_tail_mask_prevents_complete_window_identity_collision(self):
        processor = SimpleNamespace(
            feature_extractor=_FeatureExtractor(),
            _get_feat_extract_output_lengths=lambda lengths: (
                (torch.as_tensor(lengths) + 1) // 2
            ),
        )
        config = resolve_audio_encoder_window_config(
            AudioEncoderWindowSpec(
                window_frames=8,
                alignment_frames=4,
            ),
            processor=processor,
            output_length_fn=processor._get_feat_extract_output_lengths,
            model_sample_rate=16,
        )

        def build(sample_count):
            return build_audio_encoder_window_items(
                samples=np.zeros(sample_count, dtype=np.float32),
                input_ids=torch.tensor([10, 99, 11]),
                placeholder_token_id=99,
                config=config,
                processor=processor,
                output_length_fn=processor._get_feat_extract_output_lengths,
            )[0]

        tail = build(30)[1]
        complete = build(32)[1]

        self.assertTrue(torch.equal(tail.feature, complete.feature))
        self.assertEqual(tail.offsets, complete.offsets)
        self.assertNotEqual(tail.hash, complete.hash)


if __name__ == "__main__":
    unittest.main()
