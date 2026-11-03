import unittest

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.dwdp_test_utils import (
    GPT_OSS_20B_GSM8K_FLOOR,
    GSM8K_NUM_EXAMPLES,
    GSM8K_NUM_THREADS,
    MEM_FRACTION_STATIC,
    XPU_SERVER_ARGS,
    assert_gsm8k_accuracy,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    configure_nixl_pd_backend,
)

register_xpu_ci(est_time=1200, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)

MODEL = "openai/gpt-oss-20b"
# UCX's shared-memory transport breaks on XPU (docs/hardware-platforms/xpu.mdx)
_UCX_ENV = {"UCX_POSIX_USE_PROC_LINK": "n"}


@unittest.skipUnless(is_xpu(), "requires Intel XPU")
class TestDisaggregationDwdpXpu(PDDisaggregationServerBase):
    """DWDP prefill (2 cards) and TP decode (2 cards) over NIXL."""

    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    extra_prefill_env = _UCX_ENV
    extra_decode_env = _UCX_ENV
    extra_prefill_args = XPU_SERVER_ARGS + [
        "--dwdp-size",
        "2",
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
    ]
    extra_decode_args = XPU_SERVER_ARGS + [
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        configure_nixl_pd_backend(cls)
        cls.model = MODEL
        cls.launch_all()

    def test_gsm8k(self):
        assert_gsm8k_accuracy(
            self,
            model=MODEL,
            base_url=self.lb_url,
            accuracy=GPT_OSS_20B_GSM8K_FLOOR,
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=GSM8K_NUM_THREADS,
        )


if __name__ == "__main__":
    unittest.main()
