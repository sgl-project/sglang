import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-b200")

DCP_SIZE = 2
WIDENED_PAGE = 64 * DCP_SIZE


class TestGLM53NVFP4DCP2MTPHiCache(DsaMtpServerBase, UnifiedRadixTreeTestMixin):
    """GLM-5.3 TP4 + DCP2 + MTP with HiCache L2: host-mirrored KV, index keys and draft KV must replay."""

    model = "RadixArk/GLM-5.3-NVFP4"
    tp_size = 4
    mem_fraction_static = 0.8
    gsm8k_threshold = 0.88
    kl_threshold = 0.02
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=WIDENED_PAGE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=WIDENED_PAGE)
    )
    extra_server_args = [
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "bfloat16",
        "--dcp-size",
        str(DCP_SIZE),
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
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
            env={"SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=12, trust_remote_code=True)


if __name__ == "__main__":
    unittest.main()
