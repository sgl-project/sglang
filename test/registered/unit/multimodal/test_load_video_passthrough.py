"""Unit tests for ``load_video`` accepting an already-constructed decoder.

``load_video`` returns a ``VideoDecoderWrapper`` for a URL/path/bytes source, so
handing that value back to it must be a no-op rather than an error. Callers that
build the decoder themselves (a custom backend, a caller that already fetched the
bytes under its own network policy) depend on the same behaviour.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.utils.common import load_video
from sglang.test.test_utils import CustomTestCase


class _StubDecoder:
    """Stands in for VideoDecoderWrapper so the test needs no decoder backend."""


class TestLoadVideoAcceptsDecoder(CustomTestCase):
    def test_decoder_instance_passes_through(self):
        decoder = _StubDecoder()
        with patch("sglang.srt.utils.common.VideoDecoderWrapper", _StubDecoder):
            self.assertIs(load_video(decoder), decoder)

    def test_passthrough_is_idempotent(self):
        decoder = _StubDecoder()
        with patch("sglang.srt.utils.common.VideoDecoderWrapper", _StubDecoder):
            self.assertIs(load_video(load_video(decoder)), decoder)

    def test_pre_decoded_frame_types_still_pass_through(self):
        # Regression guard: the new branch must not shadow the existing ones.
        for frames in (
            [1, 2, 3],
            (1, 2, 3),
            np.zeros((2, 4, 4, 3), dtype=np.uint8),
            torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        ):
            with self.subTest(kind=type(frames).__name__):
                self.assertIs(load_video(frames), frames)

    def test_unsupported_type_still_raises(self):
        # The passthrough must not turn every unknown object into a valid input.
        with patch("sglang.srt.utils.common.VideoDecoderWrapper", _StubDecoder):
            with self.assertRaises(ValueError):
                load_video(object())


if __name__ == "__main__":
    unittest.main()
