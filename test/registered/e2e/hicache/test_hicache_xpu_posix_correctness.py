"""XPU<->host KV-transfer correctness via HiCache evict/restore over NIXL-POSIX.

A restore must reproduce the golden greedy continuation over COMPARE_TOKENS.
cached_tokens > 0 keeps that agreement from being satisfied by silent
recomputation, and load_back_tokens_total rising keeps it from being satisfied by
a prefix that never left the device -- the device pool is pinned to
EVICT_DEVICE_POOL_TOKENS so the fillers genuinely overflow it. See
hicache_xpu_common for why the golden is a device-tier hit rather than a cold
prefill (prime_cache) and why the window is bounded (COMPARE_TOKENS).
"""

import os
import shutil
import tempfile
import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.hicache_xpu_common import (
    COMPARE_TOKENS,
    EVICT_DEVICE_POOL_TOKENS,
    EVICT_HICACHE_RATIO,
    NIXL_AVAILABLE,
    XPU_AVAILABLE,
    complete,
    count_storage_files,
    flush_cache,
    launch_server,
    load_back_tokens,
    nixl_posix_config,
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

register_xpu_ci(est_time=500, suite="stage-b-test-1-gpu-xpu")


class _PosixCorrectnessServer(CustomTestCase):
    """One HiCache server over NIXL-POSIX, configured by the subclass."""

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    io_backend = None
    mem_layout = None
    process = None
    tmp_dir = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = resolve_base_url()
        cls.tmp_dir = tempfile.mkdtemp(prefix=f"hc_xpu_posix_correct_{cls.io_backend}_")
        cls.storage_dir = os.path.join(cls.tmp_dir, "storage")
        os.makedirs(cls.storage_dir, exist_ok=True)
        cls.process = launch_server(
            cls.model,
            cls.base_url,
            [
                "--tp",
                1,
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.5,
                # The fraction cannot size the device pool small enough to evict;
                # pin it in tokens, and give the host tier room for the fillers.
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
                "nixl",
                "--hicache-storage-backend-extra-config",
                nixl_posix_config(),
                # exposes cached_tokens in the usage block for the assertion
                "--enable-cache-report",
                # exposes load_back_tokens_total, the load-path guard
                "--enable-metrics",
            ],
            env_extra={"SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR": cls.storage_dir},
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        if cls.tmp_dir is not None:
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)


class _EvictRestoreChecks:
    """Scenario bodies, mixed into one concrete class per config.

    Both start from flush_cache, so they are order-independent and can share
    the subclass's single server launch.
    """

    def test_evict_restore_agrees(self):
        """One long prefix evicted to host/storage then restored."""
        # Unique long prefix so no stale cache interferes and it spans many pages.
        prefix = (
            "In a distant galaxy, a lone cartographer charted forgotten star "
            "systems and recorded their histories. "
        ) * 30
        prompt = prefix + " The most important discovery was"

        # 1) GOLDEN served from the device tier. The priming call is the one that
        # computes on device; the golden must itself be a cache hit so that only
        # the tier, not the prefill path, differs from the restore (see
        # prime_cache).
        flush_cache(self.base_url)
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

        # 2) EVICT: 12 x 421 tokens of filler against a 1024-token device pool
        # pushes the 576-token golden out of the device pool into host + storage.
        for i in range(12):
            filler = (
                f"Unrelated chronicle number {i}: the archivist catalogued "
                f"maps and ledgers across the ages. "
            ) * 20
            complete(self.base_url, self.model, filler, max_tokens=8)

        # 3) RESTORE: identical prompt -> KV restored from host/storage.
        loaded_before = load_back_tokens(self.base_url)
        restored, restored_cached = complete(
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
            "no tokens loaded back from the host tier -> the restore was served "
            "without running the load kernel, so the comparison below cannot "
            "distinguish it from a device-tier hit",
        )
        self.assertGreater(
            restored_cached,
            0,
            "restore reported 0 cached tokens -> KV was recomputed, not restored",
        )
        self.assertEqual(
            restored_cached,
            golden_cached,
            f"restore reused {restored_cached} tokens vs the golden's "
            f"{golden_cached} -> the tiers disagree on the prefix boundary, so "
            f"the comparison below would not be like for like",
        )
        spl = shared_prefix_len(restored, golden)
        self.assertEqual(
            restored,
            golden,
            f"restored continuation diverged after {spl} chars "
            f"(golden={golden!r} restored={restored!r}) -> XPU<->host KV "
            f"transfer corrupted the cache",
        )
        self.assertGreater(
            count_storage_files(self.storage_dir),
            0,
            "no NIXL storage files written -> host<->storage tier idle",
        )

    def test_multi_prompt_restore(self):
        """Four unrelated prefixes round-tripped, guarding against aliasing."""
        # Several *distinct* prompts (no shared template), each round-tripped
        # through evict/restore. Distinct bodies make each restore an independent
        # full-prefix cache hit, guarding against cross-page/index aliasing.
        bodies = [
            "The marine biologist documented bioluminescent plankton drifting "
            "through the midnight zone of the trench. ",
            "The locomotive engineer inspected the mountain railway's brakes "
            "before the long descent through the alpine tunnels. ",
            "The pastry chef folded laminated dough for the morning croissants "
            "while the ovens warmed the empty kitchen. ",
            "The radio astronomer aligned the dish toward the pulsar and "
            "logged the timing of each sweeping pulse. ",
        ]
        tails = [
            " The decisive observation was",
            " The critical adjustment turned out to be",
            " The essential technique was",
            " The surprising measurement showed",
        ]
        prompts = [(bodies[k] * 30) + tails[k] for k in range(len(bodies))]

        flush_cache(self.base_url)
        goldens = []
        for k, p in enumerate(prompts):
            # Each golden is itself a device-tier hit, so only the tier differs
            # from its restore below (see prime_cache).
            prime_cache(self.base_url, self.model, p)
            g, cached = complete(
                self.base_url,
                self.model,
                p,
                max_tokens=COMPARE_TOKENS,
                want_cached=True,
            )
            self.assertGreater(len(g.strip()), 0)
            self.assertGreater(
                cached,
                0,
                f"golden {k} reported 0 cached tokens -> device tier did not "
                f"retain the primed prefix",
            )
            goldens.append((g, cached))

        for i in range(12):
            complete(
                self.base_url, self.model, (f"Filler passage {i}. " * 40), max_tokens=8
            )

        loaded_before = load_back_tokens(self.base_url)
        any_cached = 0
        for p, (g, g_cached) in zip(prompts, goldens):
            r, c = complete(
                self.base_url,
                self.model,
                p,
                max_tokens=COMPARE_TOKENS,
                want_cached=True,
            )
            any_cached = max(any_cached, c)
            spl = shared_prefix_len(r, g)
            self.assertEqual(
                c,
                g_cached,
                f"restore reused {c} tokens vs the golden's {g_cached} -> the "
                f"tiers disagree on the prefix boundary",
            )
            self.assertEqual(
                r,
                g,
                f"multi-prompt restore diverged after {spl} chars "
                f"(golden={g!r} restored={r!r}) -> KV aliasing/corruption",
            )
        self.assertGreater(
            any_cached,
            0,
            "no prompt hit the cache on restore; evict/restore not exercised",
        )
        loaded_after = wait_load_back_tokens(self.base_url, loaded_before)
        self.assertGreater(
            loaded_after,
            loaded_before,
            "no tokens loaded back from the host tier across four restores -> "
            "every prompt was still resident on the device",
        )


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestPosixCorrectnessDirect(_PosixCorrectnessServer, _EvictRestoreChecks):
    """direct io backend: evict/restore agreement + cached_tokens>0."""

    io_backend = "direct"
    mem_layout = "page_first"


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestPosixCorrectnessKernel(_PosixCorrectnessServer, _EvictRestoreChecks):
    """kernel io backend: evict/restore agreement + cached_tokens>0."""

    io_backend = "kernel"
    mem_layout = "page_first"


if __name__ == "__main__":
    unittest.main(verbosity=2)
