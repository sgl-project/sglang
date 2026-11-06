"""CPU-only tests for the DeepEP v2 ElasticBuffer ownership facade."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher import deepep_v2
from sglang.srt.runtime_context import get_resources, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeGroup:
    pass


class _FakeBuffer:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.num_bytes = 1 << 20
        type(self).instances.append(self)


class TestDeepEPv2BufferLifecycle(CustomTestCase):
    def setUp(self):
        reset_context()
        _FakeBuffer.instances = []
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
            with self.assertRaisesRegex(ImportError, "github.com/deepseek-ai/DeepEP"):
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

    def test_masked_capacity_covers_source_rows_and_graph_padding(self):
        impl = object.__new__(deepep_v2._DeepEPv2Impl)
        impl.group = _FakeGroup()
        impl.num_max_dispatch_tokens_per_rank = 1152
        parallel = SimpleNamespace(
            attn_dp_size=1,
            attn_cp_size=1,
            tp_size=4,
            tp_group=SimpleNamespace(device_group=impl.group),
        )
        with patch.object(deepep_v2, "get_parallel", return_value=parallel):
            for rows, expected in ((0, 128), (160, 768), (264, 1152), (1152, 4608)):
                with self.subTest(rows=rows):
                    capacity = impl._masked_token_capacity(rows, 4)
                    self.assertEqual(capacity, expected)
                    self.assertGreaterEqual(capacity, min(4608, rows * 4 + 3))

            # Rank-local rows cannot bound other sources in these topologies.
            for overrides in (
                dict(attn_dp_size=2),
                dict(attn_cp_size=2),
                dict(tp_size=8),
                dict(tp_group=None),
                dict(tp_group=SimpleNamespace(device_group=_FakeGroup())),
            ):
                with self.subTest(overrides=overrides):
                    topology = SimpleNamespace(**(vars(parallel) | overrides))
                    with patch.object(deepep_v2, "get_parallel", return_value=topology):
                        self.assertEqual(impl._masked_token_capacity(160, 4), 4608)


if __name__ == "__main__":
    unittest.main()
