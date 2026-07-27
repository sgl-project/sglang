import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import is_blackwell
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=650, stage="base-c", runner_config="8-gpu-h200")

_COMMON_ARGS = [
    "--tp",
    "4",
    "--dp",
    "2",
    "--enable-dp-attention",
    "--trust-remote-code",
    "--attention-backend",
    "fa4" if is_blackwell() else "fa3",
    "--max-running-requests",
    "128",
    "--cuda-graph-max-bs-decode",
    "64",
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.75",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true,"num_threads": 64}',
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "1.5",
]

_SPEC_ARGS = [
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


class MiMoV2FlashBase(GSM8KMixin, DefaultServerBase):
    """Shared MiMo-V2-Flash fixture; subclasses set ``hicache_args`` /
    ``extra_args`` per parameter combination. Named without a ``Test`` prefix
    so unittest does not collect the base (it has no concrete config).
    """

    model = "XiaomiMiMo/MiMo-V2-Flash"
    gsm8k_accuracy_thres = 0.75

    hicache_args: list[str] = []
    extra_args: list[str] = []

    @classmethod
    def setUpClass(cls):
        cls.other_args = _COMMON_ARGS + cls.extra_args + cls.hicache_args
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            super().setUpClass()


class TestMiMoV2Flash(SpecDecodingMixin, MiMoV2FlashBase):
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 1319

    extra_args = _SPEC_ARGS
    hicache_args = [
        "--hicache-mem-layout",
        "page_first_direct",
        "--hicache-io-backend",
        "direct",
    ]

    bs_1_speed_thres = 170
    accept_length_thres = 3.2


class TestMiMoV2FlashHicacheKernelWriteBack(MiMoV2FlashBase):
    """Regression guard: MiMo-V2 has head_dim != v_head_dim (asymmetric KV
    host pool). With --hicache-io-backend kernel and --hicache-mem-layout
    page_first, unified-radix write-back used to crash at startup with
    "Destination indices must be a CUDA tensor". This config exercises that
    exact write-back path; the direct-io sibling above does not.
    """

    gsm8k_num_questions = 200
    gsm8k_num_threads = 200

    hicache_args = [
        "--hicache-mem-layout",
        "page_first",
        "--hicache-io-backend",
        "kernel",
    ]


if __name__ == "__main__":
    unittest.main()
