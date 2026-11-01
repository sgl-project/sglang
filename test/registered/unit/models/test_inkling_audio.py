import unittest

import torch

from sglang.srt.configs.inkling import InklingAudioConfig
from sglang.srt.models.inkling import InklingAudio
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestInklingAudio(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)
        config = InklingAudioConfig(
            decoder_dmodel=32,
            n_mel_bins=5,
            mel_vocab_size=7,
            use_audio_norm=False,
        )
        self.audio = InklingAudio(config)

    def _reference(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(
            dtype=self.audio.encoder.weight.dtype,
            device=self.audio.encoder.weight.device,
        )
        indices = (
            torch.arange(self.audio.n_mel_bins, device=features.device)
            * self.audio.mel_vocab_size
        ).unsqueeze(0) + features.to(torch.int32)
        return (
            self.audio.encoder(indices.reshape(-1))
            .reshape(features.shape[0], features.shape[1], -1)
            .sum(dim=1)
        )

    def test_chunk_boundaries_match_unchunked_reference(self):
        for num_tokens in (1, 511, 512, 513, 1025):
            with self.subTest(num_tokens=num_tokens):
                features = torch.randint(
                    0,
                    self.audio.mel_vocab_size,
                    (num_tokens, self.audio.n_mel_bins),
                ).float()
                expected = self._reference(features)
                actual = self.audio(features)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
