"""HiCache L2 under decode context parallelism (DCP) + UnifiedRadixCache.

Under DCP the radix/controller layer allocates in a widened logical index
space (pages of ``page_size * dcp_size``) while each rank's device and host
buffers only hold its owned 1/dcp_size token shard. If the host pool hands
those logical indices straight to the transfer kernels, ranks read and write
each other's rows: cache hits then return another rank's KV, which shows up as
silently wrong (sometimes garbage) output once entries are served from L2.

The KL cases compare logprobs for tokens served from cache against
recomputation, so a mistranslated row is a large divergence rather than a
subtle accuracy wobble. ``test_zz_l2_loadback_occurred`` asserts the L2 path
was exercised at all, since the KL cases would otherwise pass on a cache that
is never read back.

Blackwell-only: the DCP decode path for MLA is wired for ``tokenspeed_mla``
(SM100/SM12x); Hopper backends reject or mis-shape it.
"""

import subprocess
import unittest

import requests

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

register_cuda_ci(est_time=1500, stage="extra-b", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
DCP_SIZE = 4
PAGE_SIZE = 64
LOADBACK_TRIGGER_MAX_TOTAL_TOKENS = 8192
WIDENED_PAGE = PAGE_SIZE * DCP_SIZE
MAX_MAMBA_CACHE_SIZE = 256


class TestUnifiedKimiLinearDcpHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Kimi Linear + DCP4 + HiCache L2 + UnifiedRadixCache."""

    kl_threshold = 0.01
    gsm8k_threshold = 0.85
    mmlu_threshold = 0.4
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
                "--hicache-ratio",
                "3",
                "--hicache-write-policy",
                "write_through",
                "--max-total-tokens",
                str(LOADBACK_TRIGGER_MAX_TOTAL_TOKENS),
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

    def _load_back_tokens(self) -> float:
        text = requests.get(self.base_url + "/metrics", timeout=60).text
        total = 0.0
        for line in text.splitlines():
            if line.startswith("sglang:load_back_tokens_total"):
                total += float(line.rsplit(" ", 1)[1])
        return total

    def test_zz_l2_loadback_occurred(self):
        """Host->device load-backs must actually happen under DCP.

        Sorts last so it sees the traffic the KL cases generate; the probe
        below keeps it meaningful when the case is run on its own.
        """
        payload = {
            "input_ids": self.input_ids[:4],
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        }
        for _ in range(3):
            response = requests.post(
                self.base_url + "/generate", json=payload, timeout=600
            )
            response.raise_for_status()
        self.assertGreater(
            self._load_back_tokens(),
            0.0,
            "no host->device load-backs recorded; the L2 path under test was "
            "never exercised",
        )


if __name__ == "__main__":
    unittest.main()
