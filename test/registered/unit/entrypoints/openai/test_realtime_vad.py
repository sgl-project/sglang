"""Unit tests for srt/entrypoints/openai/realtime/vad.py.

Drives the StreamingVAD state machine with an injected scorer, so no
silero-vad install, model download, or GPU is needed.
"""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.realtime.vad import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    StreamingVAD,
    VADConfig,
    VADEvent,
    offset_to_ms,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FRAME_BYTES = VAD_FRAME_SAMPLES * 2  # PCM16


def make_scorer(frame_probs):
    """Scorer returning scripted per-frame probabilities (0.0 after the
    script runs out)."""
    it = iter(frame_probs)

    def scorer(_frame):
        return next(it, 0.0)

    return scorer


def frames(n):
    """n frames of PCM16 zeros (content is irrelevant with a scripted
    scorer)."""
    return b"\x00" * (_FRAME_BYTES * n)


def ms_to_frames(ms):
    return ms * VAD_SAMPLE_RATE // 1000 // VAD_FRAME_SAMPLES


class TestStreamingVAD(CustomTestCase):
    def test_speech_start_reports_prefix_padded_offset(self):
        cfg = VADConfig(prefix_padding_ms=320)  # 10 frames
        # 20 silence frames, then speech.
        vad = StreamingVAD(cfg, scorer=make_scorer([0.0] * 20 + [1.0] * 5))
        emits = vad.process(frames(25))
        self.assertEqual(len(emits), 1)
        self.assertEqual(emits[0].event_type, VADEvent.SPEECH_STARTED)
        # Speech is detected at frame 21 (samples_consumed = 21 * 512);
        # started offset backs off one frame plus the padding window.
        pad = 320 * VAD_SAMPLE_RATE // 1000
        self.assertEqual(emits[0].sample_offset, 20 * VAD_FRAME_SAMPLES - pad)

    def test_speech_start_offset_clamped_to_zero(self):
        vad = StreamingVAD(VADConfig(), scorer=make_scorer([1.0]))
        emits = vad.process(frames(1))
        self.assertEqual(
            [(e.event_type, e.sample_offset) for e in emits],
            [(VADEvent.SPEECH_STARTED, 0)],
        )

    def test_short_pause_does_not_stop_speech(self):
        cfg = VADConfig(silence_duration_ms=500)
        pause = ms_to_frames(320)  # shorter than 500 ms
        script = [1.0] * 5 + [0.0] * pause + [1.0] * 5
        vad = StreamingVAD(cfg, scorer=make_scorer(script))
        emits = vad.process(frames(len(script)))
        self.assertEqual([e.event_type for e in emits], [VADEvent.SPEECH_STARTED])
        self.assertTrue(vad.is_speech)

    def test_long_silence_stops_speech_at_last_speech_offset(self):
        cfg = VADConfig(silence_duration_ms=500)
        # +1: 500 ms is 15.625 frames, so 15 whole frames (7680 samples)
        # sit just under the 8000-sample threshold.
        silence = ms_to_frames(500) + 1
        script = [1.0] * 5 + [0.0] * silence
        vad = StreamingVAD(cfg, scorer=make_scorer(script))
        emits = vad.process(frames(len(script)))
        self.assertEqual(
            [e.event_type for e in emits],
            [VADEvent.SPEECH_STARTED, VADEvent.SPEECH_STOPPED],
        )
        # Stop offset points at the end of speech, not the end of silence.
        self.assertEqual(emits[1].sample_offset, 5 * VAD_FRAME_SAMPLES)

    def test_threshold_gates_speech(self):
        vad = StreamingVAD(VADConfig(threshold=0.9), scorer=make_scorer([0.7] * 10))
        self.assertEqual(vad.process(frames(10)), [])
        self.assertFalse(vad.is_speech)

    def test_unaligned_appends_buffer_partial_frames(self):
        vad = StreamingVAD(VADConfig(), scorer=make_scorer([0.0] * 100))
        # Feed 1.5 frames, then the missing half plus one more frame.
        self.assertEqual(vad.process(b"\x00" * (_FRAME_BYTES + _FRAME_BYTES // 2)), [])
        self.assertEqual(vad.samples_consumed, VAD_FRAME_SAMPLES)
        vad.process(b"\x00" * (_FRAME_BYTES // 2 + _FRAME_BYTES))
        self.assertEqual(vad.samples_consumed, 3 * VAD_FRAME_SAMPLES)
        self.assertEqual(len(vad.leftover_pcm), 0)

    def test_empty_input_is_noop(self):
        vad = StreamingVAD(VADConfig(), scorer=make_scorer([]))
        self.assertEqual(vad.process(b""), [])
        self.assertEqual(vad.samples_consumed, 0)

    def test_end_utterance_resets_speech_but_keeps_clock(self):
        vad = StreamingVAD(VADConfig(), scorer=make_scorer([1.0] * 5))
        vad.process(frames(5))
        self.assertTrue(vad.is_speech)
        clock = vad.samples_consumed
        vad.end_utterance()
        self.assertFalse(vad.is_speech)
        self.assertEqual(vad.samples_consumed, clock)
        self.assertEqual(len(vad.leftover_pcm), 0)

    def test_two_utterances_emit_two_start_stop_pairs(self):
        cfg = VADConfig(silence_duration_ms=96)  # 3 frames
        script = [1.0] * 3 + [0.0] * 3 + [1.0] * 3 + [0.0] * 3
        vad = StreamingVAD(cfg, scorer=make_scorer(script))
        emits = vad.process(frames(len(script)))
        self.assertEqual(
            [e.event_type for e in emits],
            [
                VADEvent.SPEECH_STARTED,
                VADEvent.SPEECH_STOPPED,
                VADEvent.SPEECH_STARTED,
                VADEvent.SPEECH_STOPPED,
            ],
        )

    def test_offset_to_ms(self):
        self.assertEqual(offset_to_ms(VAD_SAMPLE_RATE), 1000)
        self.assertEqual(offset_to_ms(0), 0)

    def test_missing_silero_raises_import_error_with_hint(self):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "silero_vad":
                raise ImportError("No module named 'silero_vad'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=blocked):
            with self.assertRaisesRegex(ImportError, "pip install silero-vad"):
                StreamingVAD(VADConfig())


if __name__ == "__main__":
    unittest.main()
