import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.expert_distribution import (
    _DeepepLowLatencySinglePassGatherer,
)
from sglang.srt.layers.moe.fused_moe_triton import layer
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPDispatcher,
    _DeepEPDispatcherImplLowLatency,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Backend:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            return lambda: attr[3:] == self.name
        raise AttributeError(attr)


def _gatherer(width, *, elastic=False):
    gatherer = object.__new__(_DeepepLowLatencySinglePassGatherer)
    gatherer._data = torch.zeros((1, width), dtype=torch.int32)
    gatherer._elastic_ep_enabled = elastic
    return gatherer


def _runner_config(num_fused_shared_experts=2):
    return SimpleNamespace(
        top_k=8,
        num_experts=16,
        num_local_experts=6,
        hidden_size=128,
        params_dtype=torch.bfloat16,
        num_fused_shared_experts=num_fused_shared_experts,
    )


class TestDeepEPLowLatencySharedSlots(CustomTestCase):
    def test_recorder_drops_trailing_shared_slots(self):
        gatherer = _gatherer(2)

        gatherer.on_deepep_dispatch_low_latency(
            0,
            torch.tensor([3, 5, 101, 103], dtype=torch.int32),
            num_trailing_shared_slots=2,
        )

        torch.testing.assert_close(
            gatherer._data, torch.tensor([[3, 5]], dtype=torch.int32)
        )

    def test_routed_length_is_derived_for_each_elastic_callback(self):
        gatherer = _gatherer(2, elastic=True)

        gatherer.on_deepep_dispatch_low_latency(
            0,
            torch.tensor([1, 2, 99], dtype=torch.int32),
            num_trailing_shared_slots=1,
        )
        gatherer.on_deepep_dispatch_low_latency(
            0,
            torch.tensor([3, 4, 5, 99], dtype=torch.int32),
            num_trailing_shared_slots=1,
        )

        torch.testing.assert_close(
            gatherer._data, torch.tensor([[4, 6]], dtype=torch.int32)
        )

    def test_nonelastic_unexplained_shape_mismatch_fails_closed(self):
        gatherer = _gatherer(2)

        with self.assertRaisesRegex(RuntimeError, "expected 2, got 3"):
            gatherer.on_deepep_dispatch_low_latency(
                0,
                torch.tensor([1, 2, 3], dtype=torch.int32),
            )

    def test_invalid_shared_slot_count_is_rejected(self):
        for value, exception in (
            (True, TypeError),
            (1.5, TypeError),
            (-1, ValueError),
            (4, ValueError),
        ):
            with self.subTest(value=value), self.assertRaises(exception):
                _gatherer(2).on_deepep_dispatch_low_latency(
                    0,
                    torch.tensor([1, 2, 3], dtype=torch.int32),
                    num_trailing_shared_slots=value,
                )

    def test_deepep_requires_the_layout_contract(self):
        param = inspect.signature(DeepEPDispatcher.__init__).parameters[
            "num_trailing_shared_slots"
        ]
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(param.default, inspect.Parameter.empty)

    def test_deepep_dispatch_forwards_the_layout_contract(self):
        dispatcher = object.__new__(_DeepEPDispatcherImplLowLatency)
        dispatcher.return_recv_hook = True
        dispatcher.num_trailing_shared_slots = 2
        recorder = Mock()
        masked_m = torch.tensor([3, 5, 101, 103], dtype=torch.int32)

        with patch(
            "sglang.srt.layers.moe.token_dispatcher.deepep."
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            dispatcher.dispatch_b(
                hidden_states=torch.empty((0, 8)),
                topk_ids=torch.empty((0, 1), dtype=torch.int64),
                topk_weights=torch.empty((0, 1)),
                masked_m=masked_m,
                expected_m=0,
                event=None,
                hook=lambda: None,
            )

        recorder.on_deepep_dispatch_low_latency.assert_called_once_with(
            masked_m,
            num_trailing_shared_slots=2,
        )

    def test_dispatcher_funnel_only_passes_contract_to_deepep(self):
        for backend_name, expected in (
            ("deepep", 2),
            ("mooncake", None),
            ("mori", None),
            ("nixl", None),
            ("pplx", None),
        ):
            with (
                self.subTest(backend=backend_name),
                patch.object(
                    layer, "get_moe_a2a_backend", return_value=_Backend(backend_name)
                ),
                patch.object(layer, "_get_deepep_comm_group", return_value=object()),
                patch.object(layer, "get_deepep_mode", return_value=object()),
                patch.object(layer, "MaybeTboDeepEPDispatcher") as dispatcher,
            ):
                layer.create_moe_dispatcher(_runner_config())
                kwargs = dispatcher.call_args.kwargs
                if expected is None:
                    self.assertNotIn("num_trailing_shared_slots", kwargs)
                else:
                    self.assertEqual(kwargs["num_trailing_shared_slots"], expected)


if __name__ == "__main__":
    import unittest

    unittest.main()
