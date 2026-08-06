from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from sglang.srt.multimodal.audio_encoder_windowing import (
    AudioEncoderWindowSpec,
    build_audio_encoder_window_items,
    resolve_audio_encoder_window_config,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen3_asr import Qwen3ASRMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FeatureExtractor:
    hop_length = 2
    n_fft = 4
    sampling_rate = 16

    def __call__(self, windows, **kwargs):
        self.last_kwargs = kwargs
        lengths = [max(len(window) // self.hop_length, 1) for window in windows]
        features = torch.zeros(len(windows), 2, max(lengths))
        masks = torch.zeros(len(windows), max(lengths), dtype=torch.long)
        for index, (window, length) in enumerate(zip(windows, lengths)):
            features[index, 0, :length] = torch.from_numpy(
                window[: length * self.hop_length : self.hop_length]
            )
            masks[index, :length] = 1
        return {"input_features": features, "attention_mask": masks}


def _extract_features(processor, windows):
    feature_extractor = processor.feature_extractor
    return feature_extractor(
        windows,
        sampling_rate=feature_extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=False,
        padding="longest",
    )


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
            feature_extractor=cls.processor.feature_extractor,
            output_length_fn=cls.processor._get_feat_extract_output_lengths,
            model_sample_rate=16,
        )

    def _build(self, sample_count):
        return build_audio_encoder_window_items(
            samples=np.arange(sample_count, dtype=np.float32),
            input_ids=torch.tensor([10, 99, 11]),
            placeholder_token_id=99,
            config=self.config,
            extract_features_fn=lambda windows: _extract_features(
                self.processor, windows
            ),
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

    def test_placeholder_layout_keeps_the_tail_uncached(self):
        items, input_ids = self._build(36)
        complete_items, _ = self._build(32)
        tiny_tail_items, _ = self._build(33)

        self.assertEqual(
            [item.offsets for item in items],
            [[(1, 8)], [(9, 16)], [(17, 18)]],
        )
        self.assertEqual(int((input_ids == 99).sum()), 18)
        self.assertEqual(
            [item.offsets for item in tiny_tail_items], [[(1, 8)], [(9, 16)]]
        )
        self.assertEqual(
            [item.use_embedding_cache for item in tiny_tail_items], [True, False]
        )
        self.assertEqual(tiny_tail_items[0].hash, complete_items[0].hash)

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
            feature_extractor=processor.feature_extractor,
            output_length_fn=processor._get_feat_extract_output_lengths,
            model_sample_rate=16,
        )

        def build(sample_count):
            return build_audio_encoder_window_items(
                samples=np.zeros(sample_count, dtype=np.float32),
                input_ids=torch.tensor([10, 99, 11]),
                placeholder_token_id=99,
                config=config,
                extract_features_fn=lambda windows: _extract_features(
                    processor, windows
                ),
                output_length_fn=processor._get_feat_extract_output_lengths,
            )[0]

        tail = build(30)[1]
        complete = build(32)[1]

        self.assertEqual(tail.offsets, complete.offsets)
        self.assertNotEqual(tail.feature.shape, complete.feature.shape)
        self.assertNotEqual(tail.hash, complete.hash)

    def test_complete_window_hash_is_stable_with_real_whisper_frontend(self):
        feature_extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            n_fft=400,
            chunk_length=30,
        )

        def output_lengths(lengths):
            lengths = torch.as_tensor(lengths)
            remainder = lengths % 100
            reduced = (remainder - 1) // 2 + 1
            return ((reduced - 1) // 2 + 1 - 1) // 2 + 1 + lengths // 100 * 13

        config = resolve_audio_encoder_window_config(
            AudioEncoderWindowSpec(window_frames=800, alignment_frames=100),
            feature_extractor=feature_extractor,
            output_length_fn=output_lengths,
            model_sample_rate=16000,
        )
        samples = (
            np.random.default_rng(7)
            .normal(
                0,
                0.1,
                size=3 * config.window_samples + 200,
            )
            .astype(np.float32)
        )

        def build(audio):
            return build_audio_encoder_window_items(
                samples=audio,
                input_ids=torch.tensor([10, 99, 11]),
                placeholder_token_id=99,
                config=config,
                extract_features_fn=lambda windows: feature_extractor(
                    windows,
                    sampling_rate=feature_extractor.sampling_rate,
                    return_attention_mask=True,
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    do_normalize=False,
                ),
                output_length_fn=output_lengths,
            )[0]

        exact_items = build(samples[: 3 * config.window_samples])
        tiny_tail_items = build(samples)

        self.assertEqual(
            [item.use_embedding_cache for item in tiny_tail_items],
            [True, True, False],
        )
        for exact, with_tail in zip(exact_items[:2], tiny_tail_items[:2]):
            self.assertTrue(torch.equal(exact.feature, with_tail.feature))
            self.assertTrue(
                torch.equal(
                    exact.model_specific_data["feature_attention_mask"],
                    with_tail.model_specific_data["feature_attention_mask"],
                )
            )
            self.assertEqual(exact.hash, with_tail.hash)

    def test_qwen_capability_uses_base_builder_and_audio_config(self):
        feature_extractor = _FeatureExtractor()
        processor = Qwen3ASRMultimodalProcessor.__new__(Qwen3ASRMultimodalProcessor)
        processor.hf_config = SimpleNamespace(
            thinker_config=SimpleNamespace(
                audio_config=SimpleNamespace(n_window_infer=8, n_window=2)
            )
        )
        processor._processor = SimpleNamespace(
            feature_extractor=feature_extractor,
            _get_feat_extract_output_lengths=lambda lengths: lengths,
        )
        processor._tokenizer = lambda *args, **kwargs: SimpleNamespace(
            input_ids=torch.tensor([[10, 99, 11]])
        )
        processor.audio_config = {"do_normalize": False}
        processor.use_cuda_ipc = False

        config = processor.resolve_audio_encoder_window_config(16)
        items, input_ids, _ = processor.process_and_combine_mm_data(
            BaseMultiModalProcessorOutput(
                input_text="audio",
                audios=[np.arange(36, dtype=np.float32)],
            ),
            MultimodalSpecialTokens(audio_token_id=99),
            audio_encoder_window_config=config,
        )

        self.assertEqual(
            [item.use_embedding_cache for item in items], [True, True, False]
        )
        self.assertEqual(int((input_ids == 99).sum()), 18)
        self.assertFalse(feature_extractor.last_kwargs["do_normalize"])
        self.assertEqual(feature_extractor.last_kwargs["padding"], "longest")


if __name__ == "__main__":
    unittest.main()
