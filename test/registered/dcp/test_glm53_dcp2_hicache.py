import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase

register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-b200")


class TestGLM53NVFP4DCP2MTPHiCache(DsaMtpServerBase, UnifiedRadixTreeTestMixin):
    """GLM-5.3 TP4 + DCP2 + MTP with HiCache L2: host-restored KV and index keys must match."""

    model = "RadixArk/GLM-5.3-NVFP4"
    tp_size = 4
    mem_fraction_static = 0.8
    gsm8k_threshold = 0.88
    kl_threshold = 0.01
    extra_server_args = [
        "--quantization",
        "modelopt_fp4",
        "--dcp-size",
        "2",
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.5",
        "--hicache-write-policy",
        "write_through",
        "--random-seed",
        "0",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.input_ids = get_input_ids(cls.model, num_samples=12, trust_remote_code=True)


if __name__ == "__main__":
    unittest.main()
