"""HiCache L2 under decode context parallelism (DCP) + UnifiedRadixCache.

Under DCP the radix layer allocates widened logical indices while each rank's
buffers hold only its 1/dcp_size shard, so a missing translation makes cache
hits return another rank's KV. The KL cases catch that as a large divergence.

Blackwell-only: the MLA DCP decode path needs ``tokenspeed_mla`` (SM100/12x).
"""

import subprocess
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=270, stage="extra-b", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
DCP_SIZE = 4
PAGE_SIZE = 64
WIDENED_PAGE = PAGE_SIZE * DCP_SIZE
MAX_MAMBA_CACHE_SIZE = 256
# Bound the host pools directly. Sizing them off the device pool would need
# --max-total-tokens, which triggers out-of-range KV writes under DCP.
HICACHE_SIZE_GB = 10


class TestUnifiedKimiLinearDcpHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Kimi Linear + DCP4 + HiCache L2 + UnifiedRadixCache."""

    kl_threshold = 0.01
    gsm8k_threshold = 0.85
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=WIDENED_PAGE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=WIDENED_PAGE)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_LINEAR_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--dcp-size",
                str(DCP_SIZE),
                "--page-size",
                str(PAGE_SIZE),
                "--attention-backend",
                "tokenspeed_mla",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--dcp-comm-backend",
                "a2a",
                "--dcp-replicate-q-proj",
                "--dtype",
                "bfloat16",
                "--random-seed",
                "0",
                "--cuda-graph-max-bs-decode",
                "64",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--mem-fraction-static",
                "0.80",
                "--enable-hierarchical-cache",
                "--hicache-size",
                str(HICACHE_SIZE_GB),
                "--hicache-write-policy",
                "write_through",
                "--max-running-requests",
                "64",
                "--max-mamba-cache-size",
                str(MAX_MAMBA_CACHE_SIZE),
                "--enable-metrics",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
