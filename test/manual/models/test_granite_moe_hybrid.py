import unittest

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

GRANITE_MOE_HYBRID_MODEL = "ibm-granite/granite-4.0-h-micro"


class TestGraniteMoeHybrid(GSM8KMixin, DefaultServerBase):
    model = GRANITE_MOE_HYBRID_MODEL
    gsm8k_accuracy_thres = 0.78


class TestGraniteMoeHybridExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = GRANITE_MOE_HYBRID_MODEL
    cache_chunk_size = 256
    gsm8k_accuracy_thres = 0.78
    kl_div_thres = 0.002
    kl_div_thres_prefill = 0.02
    other_args = [
        "--mem-fraction-static",
        "0.8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]


if __name__ == "__main__":
    unittest.main()
