import unittest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=350, stage="base-c", runner_config="8-gpu-h200")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.75
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 1319
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
        "--page-size",
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
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.5",
        "--hicache-mem-layout",
        "page_first_direct",
        "--hicache-io-backend",
        "direct",
    ]

    bs_1_speed_thres = 170
    accept_length_thres = 3.2

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            super().setUpClass()


if __name__ == "__main__":
    unittest.main()
