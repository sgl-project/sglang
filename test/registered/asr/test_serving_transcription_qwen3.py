"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Qwen3-ASR.

Covers non-streaming and chunk-based streaming transcription.

Usage:
    python3 test_serving_transcription_qwen3.py -v
"""

import unittest

from sglang.test.asr_utils import ASRTestBase, AudioTestCase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")


class TestServingTranscriptionQwen3(ASRTestBase):
    """Test Qwen3-ASR transcription via /v1/audio/transcriptions endpoint."""

    model = "Qwen/Qwen3-ASR-0.6B"
    served_model_name = "Qwen/Qwen3-ASR-0.6B"
    extra_args = ["--trust-remote-code"]
    streaming_exact_match = False
    audio_cases = [
        AudioTestCase(
            url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
            keywords=["listening", "solo", "music", "writing"],
            min_keyword_matches=2,
            local_cache_path="/tmp/test_qwen3_asr_en.wav",
        ),
        AudioTestCase(
            url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
            keywords=["交易", "停滞"],
            min_keyword_matches=1,
            local_cache_path="/tmp/test_qwen3_asr_zh.wav",
        ),
    ]


if __name__ == "__main__":
    unittest.main()
