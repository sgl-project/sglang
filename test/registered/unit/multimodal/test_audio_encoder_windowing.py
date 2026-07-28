from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature, hash_mm_item
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.audio_encoder_windowing import (
    build_audio_encoder_windows,
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
    def test_complete_windows_are_stable_and_tail_is_recomputable(self):
        """A complete window's recomputed features must be byte-stable across
        rolling requests: the embedding cache reuses encoder output purely by
        hash identity, so any drift would silently re-encode (or worse, alias)
        windows. The mutable tail must never claim a cacheable identity."""
        processor = SimpleNamespace(
            feature_extractor=_FeatureExtractor(),
            _get_feat_extract_output_lengths=lambda lengths: lengths,
        )
        config = resolve_audio_encoder_window_config(
            window_frames=8,
            alignment_frames=4,
            processor=processor,
            model_sample_rate=16,
        )

        def build(samples):
            return build_audio_encoder_windows(
                samples=samples,
                input_ids=torch.tensor([10, 99, 11]),
                placeholder_token_id=99,
                config=config,
                processor=processor,
            )

        first_items, first_ids = build(np.arange(36, dtype=np.float32))
        appended_items, _ = build(np.arange(40, dtype=np.float32))
        tiny_tail_items, _ = build(np.arange(33, dtype=np.float32))
        for item in first_items[:2] + appended_items[:2]:
            item.set_pad_value()

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
                hash_feature(first_items[0].feature),
                Modality.AUDIO,
                first_items[0].offsets,
            ),
        )
        self.assertEqual(
            [item.offsets for item in first_items],
            [[(1, 8)], [(9, 16)], [(17, 18)]],
        )
        self.assertEqual(int((first_ids == 99).sum()), 18)
        self.assertEqual(tiny_tail_items[-1].offsets, [(17, 18)])
        self.assertFalse(tiny_tail_items[-1].use_embedding_cache)


if __name__ == "__main__":
    unittest.main()
