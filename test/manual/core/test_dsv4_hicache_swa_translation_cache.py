"""E2E regression for PR #25889: DSV4 cached_loc stale after HiCache load-back.

Bug (pre-fix):
  DeepSeekV4TokenToKVPool.register_mapping() replaces full_to_swa_index_mapping
  but does NOT clear self.cached_loc. When SGLANG_OPT_CACHE_SWA_TRANSLATION=True,
  set_swa_key_buffer_radix_fused() caches the full→SWA translation across SWA
  layers. After a HiCache commit/load-back that calls register_mapping() with a
  new mapping, subsequent SWA layers reuse the stale cached_loc and write KV to
  wrong SWA pool slots — producing divergent logprobs.

Fix (PR #25889):
  register_mapping() sets self.cached_loc = None so the first SWA layer in the
  next forward pass recomputes the translation from the fresh mapping.

Test strategy:
  Subclass the existing DSV4 Flash HiCache KL suite and activate the SWA
  translation cache via SGLANG_OPT_CACHE_SWA_TRANSLATION=1. The mixin tests
  (test_multiturn_logprobs_match, test_multiturn_prefill_cache_hit_branching,
  test_multiturn_decode_cache_hit_branching) compare logprobs from cold and
  warm radix-cache hits. Without the fix the stale translation corrupts SWA KV
  data and the KL divergence exceeds the threshold; with the fix it stays within.
"""

import unittest

from test_unified_radix_cache_kl_hicache import (
    DSV4_FLASH_LAUNCH_TIMEOUT,
    DSV4_FLASH_MODEL,
    _assert_dsv4_decode_cached_tokens,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDSV4HiCacheSWATranslationCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """DSV4 Flash FP8 + HiCache + SWA translation cache enabled.

    Identical server config to TestUnifiedDeepSeekV4FlashHiCache but with
    SGLANG_OPT_CACHE_SWA_TRANSLATION=1 to activate the cached_loc path.
    Without PR #25889 the KL tests fail; with the fix they pass.
    """

    kl_threshold = 0.005
    sampling_temperature = 0
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5
    decode_cache_assert = staticmethod(_assert_dsv4_decode_cached_tokens)
    gsm8k_threshold = 0.90
    num_gsm8k_questions = 100

    @unittest.skipIf(True, "Covered by test_multiturn_prefill_cache_hit_branching.")
    def test_multiturn_logprobs_match(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--attention-backend",
                "compressed",
                "--page-size",
                "256",
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.9",
                "--disable-shared-experts-fusion",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--swa-full-tokens-ratio",
                "0.25",
                "--max-total-tokens",
                "20000",
                "--max-running-requests",
                "4",
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                # Activate the SWA translation cache — the flag that exposes the bug.
                "SGLANG_OPT_CACHE_SWA_TRANSLATION": "1",
            },
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
