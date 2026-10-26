"""CPU-only tests for the DeepEP v2 ElasticBuffer ownership facade."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher import deepep_v2
from sglang.srt.runtime_context import get_resources, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeGroup:
    pass


class _FakeBuffer:
    instances = []
    num_rdma_ranks = 1

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.num_bytes = 1 << 20
        self.num_rdma_ranks = type(self).num_rdma_ranks
        type(self).instances.append(self)


class TestDeepEPv2BufferLifecycle(CustomTestCase):
    def setUp(self):
        reset_context()
        _FakeBuffer.instances = []
        _FakeBuffer.num_rdma_ranks = 1
        self._patches = [
            patch.object(deepep_v2, "use_deepep_v2", True),
            patch.object(deepep_v2, "ElasticBuffer", _FakeBuffer, create=True),
            patch.object(deepep_v2.dist, "get_world_size", return_value=8),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self):
        reset_context()
        for item in reversed(self._patches):
            item.stop()

    def _get(self, group=None, **overrides):
        kwargs = {
            "group": group or _FakeGroup(),
            "hidden_size": 4096,
            "router_topk": 8,
            "num_max_dispatch_tokens_per_rank": 128,
            "use_fp8_dispatch": True,
            "allow_hybrid_mode": False,
        }
        kwargs.update(overrides)
        return deepep_v2.DeepEPv2Buffer.get_buffer(**kwargs)

    def test_same_key_reuses_buffer(self):
        group = _FakeGroup()
        first = self._get(group)
        second = self._get(group)
        self.assertIs(first, second)
        self.assertEqual(len(_FakeBuffer.instances), 1)

    def test_constructor_inputs_participate_in_key(self):
        group = _FakeGroup()
        first = self._get(group)
        second = self._get(group, num_max_dispatch_tokens_per_rank=256)
        third = self._get(
            group,
            num_max_dispatch_tokens_per_rank=256,
            allow_hybrid_mode=True,
        )
        self.assertIsNot(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(len(_FakeBuffer.instances), 3)

    def test_draft_topk_shares_target_buffer(self):
        """Alternating target/draft layers with different top-k must not rebuild.

        A rebuild is a collective ElasticBuffer construction, so one per
        target/draft switch stalls every DSpark step (and breaks graph replay).
        """
        group = _FakeGroup()
        target = self._get(group, router_topk=6)
        for _ in range(3):
            self.assertIs(self._get(group, router_topk=3), target)
            self.assertIs(self._get(group, router_topk=6), target)
        self.assertEqual([b.kwargs["num_topk"] for b in _FakeBuffer.instances], [6])

    def test_larger_topk_rebuilds_once_at_the_max(self):
        """A smaller top-k built first is outgrown once, then stays shared."""
        group = _FakeGroup()
        self._get(group, router_topk=3)
        grown = self._get(group, router_topk=6)
        self.assertIs(self._get(group, router_topk=3), grown)
        self.assertEqual([b.kwargs["num_topk"] for b in _FakeBuffer.instances], [3, 6])

    def test_multi_node_without_ibstat_uses_explicit_sms(self):
        """DeepEP's own SM sizing divides by zero without `ibstat` once RDMA is used."""
        _FakeBuffer.num_rdma_ranks = 2
        with patch.object(deepep_v2.shutil, "which", return_value=None):
            self._get()
            self.assertEqual(deepep_v2.DeepEPv2Buffer.num_sms(), 32)
            deepep_v2.DeepEPv2Buffer.destroy()
            with envs.SGLANG_DEEPEP_V2_NUM_SMS.override(24):
                self._get()
                self.assertEqual(deepep_v2.DeepEPv2Buffer.num_sms(), 24)
        deepep_v2.DeepEPv2Buffer.destroy()
        _FakeBuffer.num_rdma_ranks = 1
        with patch.object(deepep_v2.shutil, "which", return_value=None):
            self._get()
            self.assertEqual(deepep_v2.DeepEPv2Buffer.num_sms(), 0)

    def test_deepep_2_5_epbuffer_is_preferred(self):
        """deep_ep_v25 (sgl-deep-ep-v25) wins, then deep_ep's EPBuffer, then 2.1's ElasticBuffer."""
        import importlib
        import sys
        import types

        def _module(name, **names):
            module = types.ModuleType(name)
            module.__dict__.update(names)
            return module

        def _reload_with(v25, **names):
            # None in sys.modules makes `import deep_ep_v25` raise ModuleNotFoundError.
            modules = {"deep_ep_v25": v25, "deep_ep": _module("deep_ep", **names)}
            with patch.dict(sys.modules, modules):
                return importlib.reload(deepep_v2)

        class _V25:
            pass

        class _EP25:
            pass

        class _Elastic21:
            pass

        try:
            v25 = _module("deep_ep_v25", EPBuffer=_V25)
            module = _reload_with(v25, ElasticBuffer=_Elastic21)
            self.assertIs(module.ElasticBuffer, _V25)
            module = _reload_with(None, EPBuffer=_EP25, ElasticBuffer=_Elastic21)
            self.assertIs(module.ElasticBuffer, _EP25)
            module = _reload_with(None, ElasticBuffer=_Elastic21)
            self.assertIs(module.ElasticBuffer, _Elastic21)
            # An installed deep_ep_v25 without EPBuffer falls back to deep_ep.
            module = _reload_with(_module("deep_ep_v25"), ElasticBuffer=_Elastic21)
            self.assertIs(module.ElasticBuffer, _Elastic21)
            self.assertTrue(module.use_deepep_v2)
        finally:
            importlib.reload(deepep_v2)

    def test_masked_capacity_defaults_to_and_is_capped_by_dispatch(self):
        """Decode slabs are sized by the masked capacity, never above the buffer's."""
        from sglang.srt.layers.moe.utils import get_deepep_v2_masked_tokens_per_rank

        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(8192):
            self.assertEqual(get_deepep_v2_masked_tokens_per_rank(), 8192)
            with envs.SGLANG_DEEPEP_V2_NUM_MAX_MASKED_TOKENS_PER_RANK.override(768):
                self.assertEqual(get_deepep_v2_masked_tokens_per_rank(), 768)
            with envs.SGLANG_DEEPEP_V2_NUM_MAX_MASKED_TOKENS_PER_RANK.override(10000):
                self.assertEqual(get_deepep_v2_masked_tokens_per_rank(), 8192)

    def test_masked_dispatch_rejects_rows_over_masked_capacity(self):
        """A decode batch above the masked capacity must fail, not overflow the slab."""
        impl = object.__new__(deepep_v2._DeepEPv2Impl)
        impl._handle = None
        impl.num_max_dispatch_tokens_per_rank = 64
        impl.num_max_masked_tokens_per_rank = 8
        impl.hidden_size = 128
        impl.activation_scale_block_size = 128
        impl.router_topk = 2
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        out = StandardTopKOutput(
            topk_weights=torch.ones(9, 2),
            topk_ids=torch.zeros(9, 2, dtype=torch.int32),
            router_logits=None,
        )
        with patch.object(deepep_v2, "get_is_extend_in_batch", return_value=False):
            with self.assertRaisesRegex(ValueError, "NUM_MAX_MASKED_TOKENS_PER_RANK"):
                impl.dispatch(torch.empty(9, 128), out)

    def test_key_keeps_process_group_object(self):
        group = _FakeGroup()
        self._get(group)
        state = get_resources().buffers[deepep_v2.DeepEPv2Buffer._STATE_KEY]
        self.assertIs(state.key[0], group)

    def test_distinct_process_group_rebuilds(self):
        first = self._get(_FakeGroup())
        second = self._get(_FakeGroup())
        self.assertIsNot(first, second)
        self.assertEqual(len(_FakeBuffer.instances), 2)

    def test_state_lives_in_runtime_resources(self):
        self._get()
        self.assertIn(
            deepep_v2.DeepEPv2Buffer._STATE_KEY,
            get_resources().buffers,
        )

    def test_reset_context_drops_state_and_rebuilds(self):
        group = _FakeGroup()
        self._get(group)
        reset_context()
        self.assertNotIn(
            deepep_v2.DeepEPv2Buffer._STATE_KEY,
            get_resources().buffers,
        )
        self._get(group)
        self.assertEqual(len(_FakeBuffer.instances), 2)

    def test_failed_constructor_is_not_published(self):
        class _FailingBuffer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("construct failed")

        with patch.object(deepep_v2, "ElasticBuffer", _FailingBuffer):
            with self.assertRaisesRegex(RuntimeError, "construct failed"):
                self._get()

        state = get_resources().buffers[deepep_v2.DeepEPv2Buffer._STATE_KEY]
        self.assertIsNone(state.buffer)
        self.assertIsNone(state.key)
        self._get()
        self.assertEqual(len(_FakeBuffer.instances), 1)

    def test_destroy_clears_facade_state(self):
        group = _FakeGroup()
        first = self._get(group)
        deepep_v2.DeepEPv2Buffer.destroy()
        state = get_resources().buffers[deepep_v2.DeepEPv2Buffer._STATE_KEY]
        self.assertIsNone(state.buffer)
        self.assertIsNone(state.key)
        second = self._get(group)
        self.assertIsNot(first, second)

    def test_unavailable_deepep_fails_before_state_creation(self):
        with patch.object(deepep_v2, "use_deepep_v2", False):
            with self.assertRaisesRegex(ImportError, "Install sgl-deep-ep-v25"):
                self._get()
        self.assertNotIn(
            deepep_v2.DeepEPv2Buffer._STATE_KEY,
            get_resources().buffers,
        )

    def test_dispatch_capacity_guard_uses_actual_input_rows(self):
        impl = object.__new__(deepep_v2._DeepEPv2Impl)
        impl.num_max_dispatch_tokens_per_rank = 4
        impl.hidden_size = 128
        impl.activation_scale_block_size = 128
        impl.router_topk = 2
        impl._validate_common(torch.empty(4, 128), torch.zeros(4, 2))
        with self.assertRaisesRegex(ValueError, "per-rank buffer capacity"):
            impl._validate_common(torch.empty(5, 128), torch.zeros(5, 2))


if __name__ == "__main__":
    unittest.main()
