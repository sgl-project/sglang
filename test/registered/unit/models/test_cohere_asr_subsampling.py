"""cohere-transcribe reserves exactly as many cross-attention KV slots as the encoder emits.

``pad_input_ids`` prepends ``cohere_asr_encoder_seq_len(mel_frames)`` placeholder
tokens, and that count is the cross-attention KV span the scheduler reserves. The
encoder output written into those slots comes from the strided convolutions in
``CohereAsrConvSubsampling``. The two are computed independently, so if they ever
disagree the encoder states land in the wrong slots: fewer frames than slots
leaves stale KV that the decoder attends to, more frames are silently dropped.
"""

import unittest

import torch

from sglang.srt.configs.cohere_asr import cohere_asr_encoder_seq_len
from sglang.srt.models.cohere_asr import CohereAsrConvSubsampling
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Shrunk from the checkpoint (feat_in 128, 256 channels, d_model 1280): output
# length depends only on the conv strides/kernels/padding, not on the widths.
_ENCODER_CONFIG = {
    "feat_in": 16,
    "subsampling_conv_channels": 4,
    "subsampling_factor": 8,
    "feat_out": -1,
    "d_model": 8,
}


class TestCohereAsrEncoderSeqLen(CustomTestCase):
    def test_formula_matches_subsampling_output(self):
        subsampling = CohereAsrConvSubsampling(_ENCODER_CONFIG).eval()
        # Odd and even counts, each side of every stride-2 rounding step, a
        # 1-frame clip, and the 35s maximum clip (3500 frames).
        for mel_frames in (1, 2, 7, 8, 9, 15, 16, 17, 200, 401, 713, 1100, 3500):
            with self.subTest(mel_frames=mel_frames):
                features = torch.randn(1, _ENCODER_CONFIG["feat_in"], mel_frames)
                with torch.no_grad():
                    frames = subsampling(features).shape[1]
                self.assertEqual(cohere_asr_encoder_seq_len(mel_frames), frames)


if __name__ == "__main__":
    unittest.main()
