import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

register_cuda_ci(est_time=400, stage="base-c", runner_config="8-gpu-h200")

MIMO_V2_MODEL = "XiaomiMiMo/MiMo-V2.5"
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


if __name__ == "__main__":
    unittest.main()
