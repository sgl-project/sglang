"""Dedicated audio OpenAI-server CI for multimodal / VLM coverage.

Addresses #8496 item 3: expand VLM CI beyond vision-only paths.

Usage:
python3 test/registered/vlm/test_audio_openai_server.py TestQwen2AudioServer.test_audio_speech_completion
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import AudioOpenAITestMixin

# Qwen2-Audio launch + speech/ambient/input_audio/streaming/mixed-batch.
register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")


class TestQwen2AudioServer(AudioOpenAITestMixin):
    model = "Qwen/Qwen2-Audio-7B-Instruct"


# Avoid collecting the imported mixin as an independent test case.
del AudioOpenAITestMixin


if __name__ == "__main__":
    unittest.main()
