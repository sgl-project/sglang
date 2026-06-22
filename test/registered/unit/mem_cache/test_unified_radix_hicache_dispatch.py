import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _STRATEGIES,
    StackBuildResult,
    StackStrategy,
    _apply_stack_result,
    _DeepSeekV4Strategy,
    _DsaStrategy,
    _MambaStrategy,
    _PlainKvStrategy,
    _select_strategy,
    _SwaStrategy,
    register_stack_strategy,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _mock_kvcache(cls):
    return MagicMock(spec=cls)


FULL = ComponentType.FULL
SWA = ComponentType.SWA
MAMBA = ComponentType.MAMBA


class TestUnifiedRadixHiCacheDispatch(unittest.TestCase):
    def test_strategy_registry_ordering(self):
        order = [type(s) for s in _STRATEGIES]
        # DeepSeekV4 inherits from SWAKVPool, so it must resolve before _SwaStrategy.
        self.assertLess(order.index(_DeepSeekV4Strategy), order.index(_SwaStrategy))
        self.assertEqual(order[-1], _PlainKvStrategy)

    def test_deepseek_v4_full_swa(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = _mock_kvcache(DeepSeekV4TokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _DeepSeekV4Strategy)

    def test_mamba(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = _mock_kvcache(HybridLinearKVPool)
        strategy = _select_strategy(kvcache, {FULL, MAMBA})
        self.assertIsInstance(strategy, _MambaStrategy)

    def test_swa(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = _mock_kvcache(SWAKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _SwaStrategy)

    def test_dsa(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kvcache = _mock_kvcache(DSATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _DsaStrategy)

    def test_plain_kv_fallback(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        kvcache = _mock_kvcache(MHATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_mla_routes_to_plain(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        kvcache = _mock_kvcache(MLATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_unknown_combo_raises(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        for cls in (SWAKVPool, DeepSeekV4TokenToKVPool):
            kvcache = _mock_kvcache(cls)
            with self.assertRaises(AssertionError) as cm:
                _select_strategy(kvcache, {FULL})
            self.assertIn("No matching HiCache strategy", str(cm.exception))

    def test_register_custom_strategy_takes_precedence(self):
        class _CustomStrategy(StackStrategy):
            def matches(self, kvcache, components):
                return components == {FULL}

            def build(self, **_):
                raise NotImplementedError

        custom = _CustomStrategy()
        original = list(hybrid_pool_assembler._STRATEGIES)
        try:
            register_stack_strategy(custom)
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            kvcache = _mock_kvcache(MHATokenToKVPool)
            self.assertIs(_select_strategy(kvcache, {FULL}), custom)
        finally:
            hybrid_pool_assembler._STRATEGIES[:] = original


class TestApplyStackResult(unittest.TestCase):
    @staticmethod
    def _fake_cache(component_types):
        cache = MagicMock()
        cache.components = {ct: MagicMock() for ct in component_types}
        return cache

    def test_wires_components_sidecars_and_counters(self):
        full_host, swa_host, mamba_host = MagicMock(), MagicMock(), MagicMock()
        cache = self._fake_cache([FULL, SWA, MAMBA])
        kvcache = MagicMock()
        params = MagicMock()
        controller = MagicMock()
        sidecar = SidecarPoolSpec(
            pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV
        )
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=controller,
            component_host_pools={FULL: full_host, SWA: swa_host, MAMBA: mamba_host},
            sidecars=[sidecar],
            register_req_to_token_counter=True,
            transfer_layer_num=8,
            pools_desc="KV + SWA + MAMBA",
        )

        _apply_stack_result(cache, kvcache, params, result)

        self.assertIs(cache.host_pool_group, result.host_pool_group)
        self.assertIs(cache.cache_controller, controller)
        self.assertIs(cache.full_kv_pool_host, full_host)
        self.assertIs(cache.swa_kv_pool_host, swa_host)
        self.assertIs(cache.mamba_pool_host, mamba_host)
        self.assertIs(cache.components[FULL]._full_kv_pool_host, full_host)
        self.assertIs(cache.components[SWA]._swa_kv_pool_host, swa_host)
        self.assertIs(cache.components[MAMBA]._mamba_pool_host, mamba_host)
        cache.register_sidecar_pool.assert_called_once_with(sidecar)
        kvcache.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )
        params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )

    def test_skips_req_to_token_counter_when_flag_false(self):
        cache = self._fake_cache([FULL])
        kvcache = MagicMock()
        params = MagicMock()
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=MagicMock(),
            component_host_pools={FULL: MagicMock()},
            sidecars=[],
            register_req_to_token_counter=False,
            transfer_layer_num=1,
            pools_desc="KV",
        )

        _apply_stack_result(cache, kvcache, params, result)

        kvcache.register_layer_transfer_counter.assert_called_once()
        params.req_to_token_pool.register_layer_transfer_counter.assert_not_called()
        cache.register_sidecar_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
