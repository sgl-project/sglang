"""CPU decision table for `_should_elide_dsa_index_k`.

Skip-topk DSA layers build no Indexer, so the forward never writes their
index-K and the pool can give them a 0-row buffer. This gate decides when that
is safe; the sizing path (`pool_configurator`) and the allocation path
(`KVCacheConfigurator`) must both read the same answer, so pin it here.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers import hisparse_coordinator
from sglang.srt.mem_cache import kv_cache_configurator as kvcc
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestDSAIndexKElisionGate(CustomTestCase):
    def _elide(
        self,
        *,
        is_draft=False,
        disabled=False,
        disagg="null",
        hisparse=False,
        hierarchical=False,
        external_linker=False,
        is_npu=False,
    ):
        memory = SimpleNamespace(
            enable_hisparse=hisparse,
            enable_hierarchical_cache=hierarchical,
            enable_unified_cache_external_linker=external_linker,
        )
        fake_envs = SimpleNamespace(
            SGLANG_DISABLE_DSA_INDEXK_ELISION=SimpleNamespace(get=lambda: disabled),
        )
        with (
            mock.patch.object(kvcc, "get_memory", lambda: memory),
            mock.patch.object(
                kvcc,
                "get_disagg",
                lambda: SimpleNamespace(disaggregation_mode=disagg),
            ),
            mock.patch.object(kvcc, "envs", fake_envs),
            mock.patch.object(kvcc, "_is_npu", is_npu),
        ):
            return kvcc._should_elide_dsa_index_k(is_draft_worker=is_draft)

    def test_draft_worker_never_elides(self):
        # The MTP/NextN draft pool keeps its own indexer on every layer.
        self.assertFalse(self._elide(is_draft=True))

    def test_kill_switch_disables(self):
        self.assertFalse(self._elide(disabled=True))
        self.assertFalse(self._elide(disabled=True, disagg="decode", hisparse=True))

    def test_external_linker_never_elides(self):
        # The linker hands rows to an out-of-tree owner that assumes a dense
        # per-layer layout.
        self.assertFalse(self._elide(external_linker=True))

    def test_npu_pd_never_elides(self):
        # Ascend PD slices the KV list positionally (index-K rides in it rather
        # than in StateType.DSA), so a compact layout would mis-slice.
        for disagg in ("prefill", "decode"):
            with self.subTest(disagg=disagg):
                self.assertFalse(self._elide(is_npu=True, disagg=disagg))
        self.assertTrue(self._elide(is_npu=True, disagg="null"))

    def test_npu_hierarchical_never_elides(self):
        # The NPU HiCache host pool has no compact index-K layout.
        self.assertFalse(self._elide(is_npu=True, hierarchical=True))

    def test_plain_dsa_elides_in_every_role(self):
        # Without HiSparse the engine owns both sides of its index-K, so the
        # saving is unconditional -- including on a PD prefill or decode worker
        # and with hierarchical cache enabled.
        for disagg in ("null", "prefill", "decode"):
            for hierarchical in (False, True):
                with self.subTest(disagg=disagg, hierarchical=hierarchical):
                    self.assertTrue(
                        self._elide(disagg=disagg, hierarchical=hierarchical)
                    )

    def test_hisparse_elides_only_on_a_decode_engine(self):
        # A HiSparse prefill peer writes its own index-K and must keep it; only
        # the decode engine, which never runs the indexer for skip layers, can
        # drop the rows.
        self.assertTrue(self._elide(disagg="decode", hisparse=True))
        self.assertFalse(self._elide(disagg="prefill", hisparse=True))
        self.assertFalse(self._elide(disagg="null", hisparse=True))


class TestDSAIndexKElisionVisibility(CustomTestCase):
    def test_shared_layer_access_raises(self):
        # A caller on a skip-topk layer must not read or write the 0-row buffer.
        cache = IndexKeyCache.__new__(IndexKeyCache)
        cache.pool = SimpleNamespace(skip_topk_layers=[False, True], start_layer=2)
        cache._check_has_index_k(2)
        with self.assertRaisesRegex(RuntimeError, "Layer 3 is a skip-topk layer"):
            cache.get_k_and_scale(3, None, None, 0, 0)
        with self.assertRaisesRegex(RuntimeError, "Layer 3 is a skip-topk layer"):
            cache.store_quantized(3, None, None, None)

    def test_startup_log_reports_saving_and_host_pool(self):
        # GLM-5.2 at host_to_device_ratio=10: 656 B latent and 132 B index-K
        # per layer, 21 of 78 layers keep an indexer. The per-row cost drops
        # from 154,128 to 78,888 B, so the budget behind 338,752 rows held
        # ~173,376 rows (page-floored) before elision.
        device_pool = SimpleNamespace(
            skip_topk_layers=[i >= 21 for i in range(78)],
            index_bytes_per_token=132,
            index_kpool=1,
            size=338_752,
            index_buf_size=3_387_520,
            bytes_per_token=656,
            page_size=64,
            host_capacity_tokens=None,
        )
        with self.assertLogs(hisparse_coordinator.logger, level="INFO") as logs:
            hisparse_coordinator._log_index_k_elision(device_pool, 10)
        message = logs.output[0]
        self.assertIn("57 of 78 layers", message)
        self.assertIn("saving 7524 B per logical slot x 3387520 slots", message)
        self.assertIn("from ~173376 to 338752 tokens", message)
        # 3,387,584 host tokens (one padding page) x 51,168 B, as allocated.
        self.assertIn("from ~88.71 GB to 173.34 GB", message)
        self.assertIn(
            "--max-total-tokens N (device KV tokens; the host pool needs "
            "N x 511680 B, so N <= host_bytes / 511680)",
            message,
        )
        self.assertIn("SGLANG_DISABLE_DSA_INDEXK_ELISION=1", message)

    def test_startup_log_survives_all_shared_layers(self):
        # The saving must not depend on a live buffer, so an all-shared pool
        # still reports the bytes it would otherwise hold.
        device_pool = SimpleNamespace(
            skip_topk_layers=[True] * 8,
            index_bytes_per_token=132,
            index_kpool=1,
            size=1_000_000,
            index_buf_size=10_000_000,
            bytes_per_token=656,
            page_size=64,
            host_capacity_tokens=None,
        )
        with self.assertLogs(hisparse_coordinator.logger, level="INFO") as logs:
            hisparse_coordinator._log_index_k_elision(device_pool, 10)
        self.assertIn(
            "saving 1056 B per logical slot x 10000000 slots = 10.56 GB",
            logs.output[0],
        )


if __name__ == "__main__":
    unittest.main()
