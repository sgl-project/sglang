"""load_audio bounds decode duration on both backends (decompression-bomb DoS).

Ports vllm-project/vllm#45908: a small compressed payload must not expand into
hours of PCM. Enforced via SGLANG_MAX_AUDIO_DECODE_DURATION_S (default 600s).
"""

import io
import os
import unittest

import numpy as np

import sglang.srt.utils.common as common_utils
from sglang.srt.utils.common import _validate_audio_sample_rate, load_audio
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

SR = 16000
ENV = "SGLANG_MAX_AUDIO_DECODE_DURATION_S"


def _wav_bytes(duration_s: float, sr: int = SR) -> bytes:
    import soundfile as sf

    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


class TestLoadAudioDuration(unittest.TestCase):
    def setUp(self):
        self.short = _wav_bytes(2.0)  # a 2-second clip

    def _run_both_backends(self, check):
        """Run ``check`` against each decode backend explicitly."""
        for backend in ("torchcodec", "soundfile"):
            if backend == "torchcodec":
                try:
                    import torchcodec  # noqa: F401
                except Exception:
                    continue  # torchcodec not available in this env
            original = common_utils._BACKEND
            common_utils._BACKEND = backend
            try:
                with self.subTest(backend=backend):
                    check()
            finally:
                common_utils._BACKEND = original

    def test_under_cap_ok(self):
        def check():
            audio = load_audio(self.short, sr=SR, max_duration_s=10)
            self.assertGreater(len(audio), 0)

        self._run_both_backends(check)

    def test_over_cap_rejected(self):
        def check():
            with self.assertRaises(ValueError):
                load_audio(self.short, sr=SR, max_duration_s=1)

        self._run_both_backends(check)

    def test_nonpositive_cap_disables_check(self):
        def check():
            audio = load_audio(self.short, sr=SR, max_duration_s=0)
            self.assertGreater(len(audio), 0)

        self._run_both_backends(check)

    def test_default_cap_read_from_env(self):
        def check():
            os.environ[ENV] = "1"
            try:
                with self.assertRaises(ValueError):
                    load_audio(self.short, sr=SR)  # no explicit cap
            finally:
                os.environ.pop(ENV, None)

        self._run_both_backends(check)

    def test_nonpositive_requested_sample_rate_rejected(self):
        with self.assertRaises(ValueError):
            load_audio(self.short, sr=0)
        with self.assertRaises(ValueError):
            load_audio(self.short, sr=-1)

    def test_invalid_header_sample_rate_rejected(self):
        for sample_rate in (None, 0, -1):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    _validate_audio_sample_rate(sample_rate, "Audio file sample rate")


if __name__ == "__main__":
    unittest.main()
