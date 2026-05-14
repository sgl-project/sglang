import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase


class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.75
    gsm8k_num_questions = 1319
    gsm8k_parallel = 1319
    model = "XiaomiMiMo/MiMo-V2-Flash"

    other_args = [
        "--tp",
        "4",
        "--dp",
        "2",
        "--enable-dp-attention",
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--max-running-requests",
        "128",
        "--cuda-graph-max-bs",
        "64",
        "--mem-fraction-static",
        "0.75",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-multi-layer-eagle",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true,"num_threads": 64}',
    ]

    bs_1_speed_thres = 170
    accept_length_thres = 3.2


if __name__ == "__main__":
    unittest.main()
