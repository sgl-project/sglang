import os
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase
from sglang.test.vlm_utils import AUDIO_TRUMP_SPEECH_URL

register_cuda_ci(est_time=317, stage="base-c", runner_config="8-gpu-h200")

MIMO_V2_MODEL = os.environ.get("SGLANG_TEST_MIMO_V2_MODEL", "XiaomiMiMo/MiMo-V2.5")
MIMO_V2_OTHER_ARGS = [
    "--tp",
    "8",
    "--dp",
    "2",
    "--enable-dp-attention",
    "--mm-enable-dp-encoder",
    "--attention-backend",
    "fa3",
    "--mm-attention-backend",
    "fa3",
    "--reasoning-parser",
    "mimo",
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "1.5",
    "--hicache-mem-layout",
    "page_first_direct",
    "--hicache-io-backend",
    "direct",
]
MIMO_V2_MTP_OTHER_ARGS = MIMO_V2_OTHER_ARGS + [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
]


class TestMiMoV2(GSM8KMixin, MMMUServerBase):
    gsm8k_accuracy_thres = 0.75
    gsm8k_accept_length_thres = 2.5
    model = MIMO_V2_MODEL
    mem_fraction_static = 0.65
    server_api_key = None
    other_args = MIMO_V2_MTP_OTHER_ARGS

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            super().setUpClass()

    def test_audio_request_with_dp_attention(self):
        audio_url = os.environ.get("SGLANG_TEST_AUDIO_PATH", AUDIO_TRUMP_SPEECH_URL)
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": audio_url}},
                            {"type": "text", "text": "Transcribe this audio."},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 1,
                "routed_dp_rank": 0,
            },
            timeout=120,
        )

        self.assertEqual(response.status_code, 200, response.text)
        response_json = response.json()
        self.assertEqual(len(response_json["choices"]), 1)

        usage_details = response_json["usage"].get("prompt_tokens_details")
        self.assertIsNotNone(usage_details, "prompt carried no multimodal tokens")
        self.assertGreater(usage_details.get("audio_tokens", 0), 0)


if __name__ == "__main__":
    unittest.main()
