"""HiCache device<->host on Intel XPU with NO Tier-3 storage.

Isolates the plain device<->host copy so a regression there is not attributed to
the storage tier. Both KV-transfer io-backends are covered.
"""

import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.hicache_xpu_common import (
    COMPARE_TOKENS,
    EVICT_DEVICE_POOL_TOKENS,
    EVICT_HICACHE_RATIO,
    XPU_AVAILABLE,
    complete,
    launch_server,
    load_back_tokens,
    prime_cache,
    resolve_base_url,
    shared_prefix_len,
    wait_load_back_tokens,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=400, suite="stage-b-test-1-gpu-xpu")


class _NoStorageServer(CustomTestCase):
    """One HiCache server, no storage tier, configured by the subclass.

    Model and both memory knobs are fixed here so a subclass varies only
    (io_backend, mem_layout); otherwise a failure cannot be attributed to the
    backend rather than to the model or the pool sizing.
    """

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    io_backend = None
    mem_layout = None
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = resolve_base_url()
        cls.process = launch_server(
            cls.model,
            cls.base_url,
            [
                "--tp",
                1,
                "--enable-hierarchical-cache",
                # 1.5B weights plus a 0.4-fraction KV pool leave room for the
                # host pin on a 22 GB Arc.
                "--mem-fraction-static",
                0.4,
                # The fraction cannot size the device pool small enough to evict;
                # pin it in tokens. With no storage tier the host pool is the only
                # place an evicted page can live, so keep it well above the
                # working set or the reload turns into a recompute.
                "--max-total-tokens",
                EVICT_DEVICE_POOL_TOKENS,
                "--hicache-ratio",
                EVICT_HICACHE_RATIO,
                "--page-size",
                16,
                "--hicache-io-backend",
                cls.io_backend,
                "--hicache-mem-layout",
                cls.mem_layout,
                # exposes cached_tokens in the usage block for the reuse assertion
                "--enable-cache-report",
                # exposes load_back_tokens_total, the load-path guard
                "--enable-metrics",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)


class _DeviceHostReuseChecks:
    """Scenario body, mixed into one concrete class per config."""

    def test_repeat_after_eviction_agrees(self):
        """A repeat reloaded from the host tier agrees with the device-tier hit.

        The two runs of the prompt are separated by enough filler to evict it,
        so the second one is served by the host->device load kernel rather than
        by the radix tree; load_back_tokens_total is what tells the two apart.
        """
        # Radix reuse is page-granular, so the prompt must span several
        # --page-size 16 pages; an 11-token prompt reports 0 cached whatever the
        # KV path does.
        prompt = (
            "The answer to life, the universe, and everything has been debated "
            "by philosophers, novelists, and astronomers alike. " * 8
        ) + " The most widely cited answer is"
        prime_cache(self.base_url, self.model, prompt, max_tokens=COMPARE_TOKENS)
        first, first_cached = complete(
            self.base_url,
            self.model,
            prompt,
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        # 191-token prompt + 10 x 128 tokens of filler against a 1024-token
        # device pool, so the prompt's pages are evicted to the host tier.
        for i in range(10):
            complete(
                self.base_url,
                self.model,
                f"Unrelated filler request {i}: " + "lorem ipsum " * 60,
                max_tokens=16,
            )
        loaded_before = load_back_tokens(self.base_url)
        second, cached = complete(
            self.base_url,
            self.model,
            prompt,
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        loaded_after = wait_load_back_tokens(self.base_url, loaded_before)
        self.assertGreater(
            loaded_after,
            loaded_before,
            "no tokens loaded back from the host tier -> the repeat was served "
            "from the device pool, so the assertions below say nothing about the "
            "device<->host path",
        )
        self.assertGreater(
            cached, 0, "repeat reported 0 cached tokens -> no prefix reuse"
        )
        self.assertEqual(
            cached,
            first_cached,
            f"reuse boundary moved between the two repeats "
            f"({first_cached} then {cached} cached tokens)",
        )
        spl = shared_prefix_len(second, first)
        self.assertEqual(
            second,
            first,
            f"repeat diverged after {spl} chars "
            f"(first={first!r} second={second!r}) -> device<->host reuse "
            f"corrupted the KV",
        )


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestNoStorageDirectLayerFirst(_NoStorageServer, _DeviceHostReuseChecks):
    """direct/layer_first: device<->host reuse with no storage tier."""

    io_backend = "direct"
    mem_layout = "layer_first"


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestNoStorageKernelPageFirst(_NoStorageServer, _DeviceHostReuseChecks):
    """kernel/page_first: device<->host reuse with no storage tier."""

    io_backend = "kernel"
    mem_layout = "page_first"


if __name__ == "__main__":
    unittest.main(verbosity=2)
