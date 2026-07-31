"""HiCache L2 under decode context parallelism (DCP) + UnifiedRadixCache.

Under DCP the radix/controller layer allocates in a widened logical index
space (pages of ``page_size * dcp_size``) while each rank's device and host
buffers only hold its owned 1/dcp_size token shard. If the host pool hands
those logical indices straight to the transfer kernels, ranks read and write
each other's rows: cache hits then return another rank's KV, which shows up as
silently wrong (sometimes garbage) output once entries are served from L2.

The KL tests below compare logprobs for tokens served from cache against
recomputation, so a mistranslated row is a large divergence rather than a
subtle accuracy wobble. ``test_zz_l2_loadback_occurred`` guards the guard: with a
correct cache but no eviction pressure nothing would ever be read back from
the host, and the KL tests would pass without exercising the path at all.

Verified against the pre-fix code (4x GB300, Kimi Linear, DCP4): without the
host-pool index translation the L2 path is unusable in both directions --
at this token budget the host pool exposes only its physical slot count to a
radix layer allocating logical ones, nothing is retained and
``test_zz_l2_loadback_occurred`` fails with zero load-backs; with a larger budget
entries are retained but served from the wrong rank's rows, which shows up as
wrong and partly garbage output. All four cases pass with the translation in
place.

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
# Small enough that the device pool evicts during the multi-turn branches, so
# later turns must be served from the host pool (the path under test).
LOADBACK_TRIGGER_MAX_TOTAL_TOKENS = 8192
# Under DCP the radix layer allocates in widened pages, so cache hits commit
# in units of page_size * dcp_size (256 here), not page_size.
WIDENED_PAGE = PAGE_SIZE * DCP_SIZE
# The KDA sidecar host pool is sized device_pool x hicache_ratio and ignores
# --max-running-requests; uncapped it asks for ~300 GB per rank.
MAX_MAMBA_CACHE_SIZE = 256


class TestUnifiedKimiLinearDcpHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Kimi Linear + DCP4 + HiCache L2 + UnifiedRadixCache."""

    # Measured noise floor for this config (fp8 KV + DCP a2a reductions) is
    # ~0.006 with HiCache *disabled*, so the usual 0.003/0.005 thresholds are
    # below the baseline. A mistranslated host row is orders of magnitude
    # larger than this, so 0.01 still fails loudly on the bug it guards.
    kl_threshold = 0.01
    gsm8k_threshold = 0.85
    # simple-evals MMLU runs only 64 examples here, so the score carries a
    # ~6 point standard error; this sits ~3 sigma under the measured 0.56.
    mmlu_threshold = 0.4
    # Cache hits commit whole *widened* pages, so cached_tokens floors to
    # WIDENED_PAGE. Confirmed against a HiCache-disabled control run, which
    # reports the same counts -- this is DCP page granularity, not the L2 path.
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
                # Back every page up immediately instead of waiting for the
                # reuse heuristic, so L2 population does not depend on access
                # order (matches the Mamba HiCache sibling test).
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
        # Kimi Linear ships a custom tokenizer/config, so the KL corpus
        # tokenizer needs the same trust_remote_code as the server.
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

        Without this the KL cases could pass on a cache that is never read
        back from the host, which is exactly the configuration in which a
        broken index translation is invisible. Named to sort last so it sees
        the traffic the KL cases generate; the probe below keeps it meaningful
        when the case is run on its own.
        """
        # Working set sized between the two pools: ~4 x 3k tokens exceeds the
        # 8k device pool (so it must spill) but fits the 3x larger host pool
        # (so it is still there to come back).
        payload = {
            "input_ids": self.input_ids[:4],
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        }
        for _ in range(3):
            response = requests.post(
                self.base_url + "/generate", json=payload, timeout=600
            )
            response.raise_for_status()
        # Cumulative, not a delta: whether this case or an earlier one drove
        # the traffic, what matters is that the L2 path was exercised while
        # the KL cases were asserting losslessness.
        self.assertGreater(
            self._load_back_tokens(),
            0.0,
            "no host->device load-backs recorded; the L2 path under test was "
            "never exercised",
        )


if __name__ == "__main__":
    unittest.main()
