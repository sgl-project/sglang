import unittest

import numpy as np
import torch

from sglang.srt.configs.fireredasr2 import (
    DEFAULT_ASR_PROMPT,
    FireRedASR2FeatureExtractor,
    FireRedASR2Processor,
    get_audio_token_count,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFireRedASR2Processor(CustomTestCase):
    def test_token_count_matches_convolution_mask(self):
        # The reference counts valid mask entries, not the convolution's
        # receptive-field lengths; every frame residue modulo eight matters.
        for frames in list(range(1, 65)) + [2997, 2998, 3000]:
            mask = torch.arange(3006) < frames
            count = max(1, int(mask[:-2:2][:-2:2].sum()) // 2)
            self.assertEqual(get_audio_token_count(frames), count)

    def test_fbank_pcm_scale_cmvn_and_padding(self):
        # Silence has log(float32 epsilon) in every Kaldi mel bin. This
        # anchors the floor, CMVN precision, frame layout and zero padding.
        extractor = FireRedASR2FeatureExtractor(
            means=[1.0] * 80, inverse_std_variences=[0.5] * 80
        )
        result = extractor([np.zeros(16000, dtype=np.float32)], sampling_rate=16000)
        expected = (np.log(np.finfo(np.float32).eps) - 1) * 0.5
        torch.testing.assert_close(
            result.input_features[0, :98], torch.full((98, 80), float(expected))
        )
        self.assertEqual(result.speech_lengths.tolist(), [98])
        self.assertEqual(result.fake_token_lengths.tolist(), [12])
        self.assertEqual(torch.count_nonzero(result.input_features[0, 98:]), 0)
        with self.assertRaises(ValueError):
            extractor([np.zeros(16000)], sampling_rate=8000)
        with self.assertRaises(ValueError):
            extractor([np.zeros(480001)], sampling_rate=16000)

    def test_pcm_amplitude_before_fbank(self):
        from torchaudio.compliance.kaldi import fbank

        speech = np.sin(np.arange(16000, dtype=np.float32) * 0.1) * 0.2
        extractor = FireRedASR2FeatureExtractor(
            means=[0.0] * 80, inverse_std_variences=[1.0] * 80
        )
        result = extractor([speech], sampling_rate=16000)
        normalized = fbank(torch.from_numpy(speech)[None], num_mel_bins=80, dither=0)
        # Power spectra scale quadratically, so PCM scaling adds 2*log(32768).
        torch.testing.assert_close(
            result.input_features[0, :98, :4],
            normalized[:, :4] + 2 * np.log(32768),
            atol=2e-4,
            rtol=2e-4,
        )

    def test_prompt_expansion_preserves_reference_framing(self):
        class Tokenizer:
            def convert_tokens_to_ids(self, token):
                return 152060

            def __call__(self, text, **kwargs):
                return {"rendered_text": text}

        processor = FireRedASR2Processor(
            FireRedASR2FeatureExtractor(
                means=[0.0] * 80, inverse_std_variences=[1.0] * 80
            ),
            Tokenizer(),
        )
        result = processor(
            text=[DEFAULT_ASR_PROMPT], audios=[np.zeros(16000, dtype=np.float32)]
        )
        self.assertEqual(
            result["rendered_text"],
            [
                "<|im_start|>user\n"
                + "<|AUDIO|>" * 12
                + "请转写音频为文字<|im_end|>\n<|im_start|>assistant\n"
            ],
        )
        with self.assertRaises(ValueError):
            processor(text=[DEFAULT_ASR_PROMPT], audios=[])


if __name__ == "__main__":
    unittest.main()
