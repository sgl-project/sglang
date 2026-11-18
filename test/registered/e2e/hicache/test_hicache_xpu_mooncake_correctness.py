"""XPU HiCache evict/restore correctness with Mooncake as the storage backend.

A restore must reproduce the golden greedy continuation over COMPARE_TOKENS.
Two tiers are exercised: a host (L2) reload after the device pool overflows,
and a Mooncake (L3) fetch after flush_cache has emptied the device and host
tiers.

Mooncake runs over TCP, so no RDMA NIC is needed; HiCache hands it host-pool
pages, so the transfer engine never touches XPU memory.
"""

import time
import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.hicache_xpu_common import (
    COMPARE_TOKENS,
    DETERMINISTIC_DECODE_ARGS,
    EVICT_DEVICE_POOL_TOKENS,
    EVICT_HICACHE_RATIO,
    XPU_AVAILABLE,
    complete,
    flush_cache,
    launch_server,
    load_back_tokens,
    prime_cache,
    resolve_base_url,
    shared_prefix_len,
    wait_load_back_tokens,
)
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    terminate_and_kill_process_tree,
)

register_xpu_ci(
    est_time=400,
    suite="nightly-kernel-main-test-1-gpu-xpu",
    nightly=True,
    disabled=(
        "XPU HiCache storage coverage in CI is NIXL-POSIX "
        "(test_hicache_xpu_posix_correctness); run this locally against a "
        "Mooncake build with -DUSE_XPU=ON -DUSE_TENT=ON -DWITH_STORE=ON"
    ),
)

try:
    import mooncake.store  # noqa: F401

    MOONCAKE_AVAILABLE = True
except ImportError:
    MOONCAKE_AVAILABLE = False

# 1 GiB of store segment is well above both scenarios' working sets.
_STORE_SEGMENT_BYTES = 1 << 30


def _wait_master_metric(mooncake, name, above, timeout_s=30.0) -> float:
    """Poll a Mooncake master counter until it passes `above`, then return it.

    Storage backup runs on the cache controller's thread after the request
    returns, so a read straight after the completion races the write.
    """
    deadline = time.time() + timeout_s
    seen = mooncake.master_metric(name)
    while seen <= above and time.time() < deadline:
        time.sleep(0.5)
        seen = mooncake.master_metric(name)
    return seen


class _MooncakeCorrectnessServer(CustomTestCase):
    """One HiCache server backed by a local Mooncake store, plus its scenarios.

    Each scenario uses its own prompts and starts from flush_cache; storage is
    not flushed, so a prompt reused across scenarios would hit L3.
    """

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    io_backend = None
    mem_layout = None
    process = None
    mooncake = None

    @classmethod
    def setUpClass(cls):
        if cls is _MooncakeCorrectnessServer:
            raise unittest.SkipTest(
                "abstract base; concrete subclasses set io_backend and mem_layout"
            )
        cls.mooncake = MooncakeTestServices(
            protocol="tcp", store_segment_size=_STORE_SEGMENT_BYTES
        )
        cls.mooncake.start()
        cls.base_url = resolve_base_url()
        cls.process = launch_server(
            cls.model,
            cls.base_url,
            [
                "--tp",
                1,
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.5,
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
                "--hicache-storage-backend",
                "mooncake",
                # A restore must not be served before the L3 fetch lands.
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--enable-cache-report",
                "--enable-metrics",
                *DETERMINISTIC_DECODE_ARGS,
            ],
            env_extra=cls.mooncake.server_env(),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        if cls.mooncake is not None:
            cls.mooncake.stop()

    def _golden(self, prompt):
        prime_cache(self.base_url, self.model, prompt)
        golden, golden_cached = complete(
            self.base_url,
            self.model,
            prompt,
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        self.assertGreater(len(golden.strip()), 0, "golden output empty")
        self.assertGreater(
            golden_cached,
            0,
            "golden reported 0 cached tokens -> device tier did not retain the "
            "primed prefix, so it is not a tier-1 baseline",
        )
        return golden, golden_cached

    def _assert_restore_matches(self, prompt, golden, golden_cached, tier):
        restored, restored_cached = complete(
            self.base_url,
            self.model,
            prompt,
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        self.assertEqual(
            restored_cached,
            golden_cached,
            f"{tier} restore reused {restored_cached} tokens vs the golden's "
            f"{golden_cached} -> the tiers disagree on the prefix boundary",
        )
        spl = shared_prefix_len(restored, golden)
        self.assertEqual(
            restored,
            golden,
            f"{tier} restore diverged after {spl} chars "
            f"(golden={golden!r} restored={restored!r}) -> KV corrupted in transit",
        )

    def test_evict_restore_agrees(self):
        """A prefix evicted from the device pool reloads from host intact."""
        prompt = (
            "In a distant galaxy, a lone cartographer charted forgotten star "
            "systems and recorded their histories. "
        ) * 30 + " The most important discovery was"

        flush_cache(self.base_url)
        keys_before = self.mooncake.master_metric("master_key_count")
        golden, golden_cached = self._golden(prompt)

        # 12 x 421 tokens of filler against a 1024-token device pool pushes
        # the 576-token golden out of the device pool.
        for i in range(12):
            filler = (
                f"Unrelated chronicle number {i}: the archivist catalogued "
                f"maps and ledgers across the ages. "
            ) * 20
            complete(self.base_url, self.model, filler, max_tokens=8)

        loaded_before = load_back_tokens(self.base_url)
        self._assert_restore_matches(prompt, golden, golden_cached, tier="host")
        self.assertGreater(
            wait_load_back_tokens(self.base_url, loaded_before),
            loaded_before,
            "no tokens loaded back from the host tier -> the restore was a "
            "device-tier hit and did not exercise the load kernel",
        )
        self.assertGreater(
            _wait_master_metric(self.mooncake, "master_key_count", keys_before),
            keys_before,
            "no keys written to Mooncake -> host<->storage tier idle",
        )

    def test_storage_restore_agrees(self):
        """A prefix held only in Mooncake is fetched back intact."""
        # Picked for its top-1 margin: the golden never comes within 0.25
        # logprob of a tie over COMPARE_TOKENS, so page-order rounding cannot
        # flip it. A near-tie prompt diverges at token 11 on a bit-exact restore.
        prompt = (
            "The lighthouse keeper climbed the spiral stairs each evening to "
            "trim the wick and polish the great lens. "
        ) * 25 + " The keeper's most important duty was"

        flush_cache(self.base_url)
        keys_before = self.mooncake.master_metric("master_key_count")
        golden, golden_cached = self._golden(prompt)
        self.assertGreater(
            _wait_master_metric(self.mooncake, "master_key_count", keys_before),
            keys_before,
            "prefix was never backed up to Mooncake, so a flush would lose it",
        )

        # Drops the device and host tiers; only Mooncake still holds the prefix.
        flush_cache(self.base_url)
        gets_before = self.mooncake.master_metric(
            "master_batch_get_replica_list_items_total"
        )
        self._assert_restore_matches(prompt, golden, golden_cached, tier="storage")
        self.assertGreater(
            self.mooncake.master_metric("master_batch_get_replica_list_items_total"),
            gets_before,
            "no Mooncake reads after the flush -> the restore did not come from "
            "the storage tier",
        )


_SKIP_REASON = "Intel XPU and mooncake-transfer-engine required"


@unittest.skipUnless(XPU_AVAILABLE and MOONCAKE_AVAILABLE, _SKIP_REASON)
class TestMooncakeCorrectnessDirect(_MooncakeCorrectnessServer):
    io_backend = "direct"
    mem_layout = "page_first"


@unittest.skipUnless(XPU_AVAILABLE and MOONCAKE_AVAILABLE, _SKIP_REASON)
class TestMooncakeCorrectnessKernel(_MooncakeCorrectnessServer):
    io_backend = "kernel"
    mem_layout = "page_first"


if __name__ == "__main__":
    unittest.main(verbosity=2)
