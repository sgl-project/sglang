"""MHA KV-transfer kernel matrix on Intel XPU with NIXL-POSIX storage attached.

Drives every MHA load/backup kernel through prefill -> host/storage offload ->
reload and asserts the reloaded continuation still agrees with the first plus
Tier-3 files written. The device pool is pinned to EVICT_DEVICE_POOL_TOKENS and
the reload is gated on load_back_tokens_total rising, so a run where nothing was
evicted fails instead of passing on a device-tier hit. See prime_cache in
hicache_xpu_common for why the baseline is a device-tier hit rather than a cold
prefill.

Kernel selection (see pool_host/mha.py load_to_device_per_layer /
backup_from_device_all_layer):

    io_backend  mem_layout    load kernel                         backup kernel
    ----------  ------------  ----------------------------------  ----------------------------------
    kernel      layer_first   transfer_kv_per_layer               transfer_kv_all_layer
    kernel      page_first    transfer_kv_per_layer_pf_lf         transfer_kv_all_layer_lf_pf
    kernel      page_head     transfer_kv_per_layer_ph_lf         transfer_kv_all_layer_lf_ph
    direct      layer_first   transfer_kv_direct                  transfer_kv_direct
    direct      page_first*   transfer_kv_per_layer_direct_pf_lf  transfer_kv_all_layer_direct_lf_pf

    (*) direct + page_first auto-resolves to page_first_direct in server_args.

The table holds on XPU because can_use_jit is gated to (_is_cuda or _is_hip) at
mha.py:123, so the jit_transfer_hicache_* branch preceding each transfer_kv_*
call never fires here. It is documentation, not an assertion: the server runs in
another process, so the test cannot observe which symbol dispatched.
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

register_xpu_ci(est_time=700, suite="stage-b-test-1-gpu-xpu")


class _NixlTransferServer(CustomTestCase):
    """One HiCache server over NIXL-POSIX, configured by the subclass."""

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    io_backend = None
    mem_layout = None
    process = None
    tmp_dir = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = resolve_base_url()
        cls.tmp_dir = tempfile.mkdtemp(
            prefix=f"hc_xpu_nixl_{cls.io_backend}_{cls.mem_layout}_"
        )
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
                # Pin the device pool in tokens; the fraction alone leaves it far
                # too large for the prompts below to evict anything.
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
                # exposes cached_tokens in the usage block for the reload assertion
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


class _OffloadReloadCheck:
    """Scenario body, mixed into one concrete class per kernel pair."""

    def test_offload_reload_round_trip(self):
        """Evict a long prefix to host/storage, reload it, and compare greedily.

        One method rather than several: the storage-file and cached-token
        assertions are only meaningful after the eviction has run, and unittest
        orders methods alphabetically, not causally.
        """
        # 444-token shared prefix, then 8 x 128 tokens of filler: 1468 tokens
        # against a 1024-token device pool, so the prefix is evicted to
        # host/storage and the reload has to run the load kernel.
        prefix = "The history of computing is a long and fascinating story. " * 40
        prime_cache(
            self.base_url,
            self.model,
            prefix + " In summary,",
            max_tokens=COMPARE_TOKENS,
        )
        out1, out1_cached = complete(
            self.base_url,
            self.model,
            prefix + " In summary,",
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )
        for i in range(8):
            complete(
                self.base_url,
                self.model,
                f"Unrelated filler request {i}: " + "lorem ipsum " * 60,
                max_tokens=16,
            )
        loaded_before = load_back_tokens(self.base_url)
        out2, cached = complete(
            self.base_url,
            self.model,
            prefix + " In summary,",
            max_tokens=COMPARE_TOKENS,
            want_cached=True,
        )

        tag = f"[{self.io_backend}/{self.mem_layout}]"
        loaded_after = wait_load_back_tokens(self.base_url, loaded_before)
        self.assertGreater(
            loaded_after,
            loaded_before,
            f"{tag} no tokens loaded back from the host tier -> the reload was "
            f"served without running the load kernel, so the assertions below "
            f"cannot distinguish it from a device-tier hit",
        )
        self.assertGreater(len(out1.strip()), 0, f"{tag} empty output")
        self.assertGreater(
            cached, 0, f"{tag} reload reported 0 cached tokens -> KV was recomputed"
        )
        self.assertEqual(
            cached,
            out1_cached,
            f"{tag} reload reused {cached} tokens vs the baseline's "
            f"{out1_cached} -> the tiers disagree on the prefix boundary",
        )
        spl = shared_prefix_len(out2, out1)
        self.assertEqual(
            out2,
            out1,
            f"{tag} reloaded continuation diverged after {spl} chars "
            f"(first={out1!r} reloaded={out2!r}) -> host/storage round trip "
            f"corrupted the KV",
        )
        self.assertGreater(
            count_storage_files(self.storage_dir),
            0,
            f"{tag} no NIXL storage files written -> host<->storage tier idle",
        )


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestNixlDirectLayerFirst(_NixlTransferServer, _OffloadReloadCheck):
    """direct/layer_first: transfer_kv_direct both ways."""

    io_backend = "direct"
    mem_layout = "layer_first"


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestNixlDirectPageFirstDirect(_NixlTransferServer, _OffloadReloadCheck):
    """direct/page_first (auto-resolved to page_first_direct)."""

    io_backend = "direct"
    mem_layout = "page_first"


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestNixlKernelLayerFirst(_NixlTransferServer, _OffloadReloadCheck):
    """kernel/layer_first: transfer_kv_per_layer / transfer_kv_all_layer."""

    io_backend = "kernel"
    mem_layout = "layer_first"


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestNixlKernelPageFirst(_NixlTransferServer, _OffloadReloadCheck):
    """kernel/page_first: transfer_kv_per_layer_pf_lf / transfer_kv_all_layer_lf_pf."""

    io_backend = "kernel"
    mem_layout = "page_first"


@unittest.skipUnless(XPU_AVAILABLE and NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestNixlKernelPageHead(_NixlTransferServer, _OffloadReloadCheck):
    """kernel/page_head: transfer_kv_per_layer_ph_lf / transfer_kv_all_layer_lf_ph."""

    io_backend = "kernel"
    mem_layout = "page_head"


if __name__ == "__main__":
    unittest.main(verbosity=2)
