import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase


class TestMiMoV2Flash(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.75
    model = "XiaomiMiMo/MiMo-V2-Flash"

    other_args = [
        "--tp",
        "4",
        "--dp",
        "2",
        "--enable-dp-attention",
        "--trust-remote-code",
        "--tool-call-parser",
        "mimo",
        "--reasoning-parser",
        "qwen3",
        "--attention-backend",
        "fa3",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        "128",
        "--cuda-graph-max-bs",
        "64",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true,"num_threads": 64}',
    ]


if __name__ == "__main__":
    unittest.main()
