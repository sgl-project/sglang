import unittest

import torch

from sglang.srt.models.mimo_audio import (
    AudioEncoder,
    MiMoAudioTokenizerConfig,
    group_by_length,
    tokenize_audio_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMiMoAudioBatching(unittest.TestCase):
    def test_groups_respect_frame_budget(self):
        lengths = torch.tensor([2, 3, 7, 0, 4])
        features = torch.arange(16)
        groups, segments = group_by_length(features, lengths, max_length=5)
        self.assertEqual([x.tolist() for x in segments], [[2, 3], [7], [0, 4]])
        self.assertEqual([len(x) for x in groups], [5, 7, 4])
        torch.testing.assert_close(torch.cat(groups), features)
        with self.assertRaisesRegex(ValueError, "Feature size mismatch: 15 vs 16"):
            group_by_length(features[:-1], lengths, max_length=5)

    def test_codes_split_at_mel_boundaries(self):
        for avg_pooler in (1, 3):
            with self.subTest(avg_pooler=avg_pooler), torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                config = MiMoAudioTokenizerConfig(
                    d_model=16,
                    n_mels=8,
                    encoder_layers=0,
                    encoder_attention_heads=2,
                    num_quantizers=2,
                    codebook_size=[32, 32],
                    avg_pooler=avg_pooler,
                )
                encoder = AudioEncoder(config).eval()
                for layer in encoder.quantizer.vq.layers:
                    layer.codebook.normal_()
                mels = [torch.randn(n, 8) for n in (17, 9, 23)]
                lengths = []
                code_sizes = []
                for mel in mels:
                    sizes = [len(segment) for segment in mel.split(8)]
                    lengths.extend(sizes)
                    # Stride-2 convolution, then ceil-pool each segment independently.
                    code_sizes.append(
                        sum(
                            ((n + 1) // 2 + avg_pooler - 1) // avg_pooler for n in sizes
                        )
                    )
                packed, _ = encoder.encode(
                    torch.cat(mels), torch.tensor(lengths), return_codes_only=True
                )
                expected = packed.T.split(code_sizes)
                self.assertGreater(packed.unique().numel(), 1)
                actual = tokenize_audio_batch(mels, encoder, segment_size=8)
                self.assertEqual(len(actual), len(expected))
                for output, reference in zip(actual, expected):
                    torch.testing.assert_close(output, reference)


if __name__ == "__main__":
    unittest.main()
