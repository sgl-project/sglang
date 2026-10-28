"""CPU decision table for `_should_elide_dsa_index_k`.

Skip-topk DSA layers build no Indexer, so the forward never writes their
index-K and the pool can give them a 0-row buffer. This gate decides when that
is safe; the sizing path (`pool_configurator`) and the allocation path
(`KVCacheConfigurator`) must both read the same answer, so pin it here.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache import kv_cache_configurator as kvcc
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


if __name__ == "__main__":
    unittest.main()
