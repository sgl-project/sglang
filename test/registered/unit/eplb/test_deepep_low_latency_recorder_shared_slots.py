import inspect
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.expert_distribution import (
    _DeepepLowLatencySinglePassGatherer,
    _DeepepNormalSinglePassGatherer,
    _ExpertDistributionRecorderReal,
)
from sglang.srt.layers.moe.fused_moe_triton import layer
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPDispatcher,
    _DeepEPDispatcherImplLowLatency,
    _DeepEPDispatcherImplNormal,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    MoriEPDispatcher,
    _MoriEPDispatcherImplLowLatency,
    _MoriEPDispatcherImplNormal,
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


def _normal_gatherer():
    gatherer = object.__new__(_DeepepNormalSinglePassGatherer)
    gatherer._expert_location_metadata = SimpleNamespace(num_layers=1)
    gatherer._objects_of_layer = {}
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

    def test_normal_recorder_drops_trailing_shared_slots(self):
        gatherer = _normal_gatherer()

        gatherer.on_deepep_dispatch_normal(
            0,
            [3, 5, 101, 103],
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=None,
            num_trailing_shared_slots=2,
        )

        self.assertEqual(gatherer._objects_of_layer, {0: [3, 5]})

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

    def test_active_dispatchers_require_the_layout_contract(self):
        for dispatcher_cls in (DeepEPDispatcher, MoriEPDispatcher):
            with self.subTest(dispatcher=dispatcher_cls.__name__):
                param = inspect.signature(dispatcher_cls.__init__).parameters[
                    "num_trailing_shared_slots"
                ]
                self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(param.default, inspect.Parameter.empty)

    def test_recorder_forwards_the_layout_contract_to_the_active_gatherer(self):
        gatherer = Mock()
        recorder = object.__new__(_ExpertDistributionRecorderReal)
        recorder._disable_all = False
        recorder._recording = True
        recorder._current_debug_name = SimpleNamespace(value="default")
        recorder._current_layer_idx = SimpleNamespace(value=3)
        recorder._accumulator = Mock()
        recorder._accumulator.get_single_pass_gatherer_key.return_value = "gatherer"
        recorder._single_pass_gatherers = {"gatherer": gatherer}
        local_count = torch.tensor([3, 5, 101, 103], dtype=torch.int32)

        recorder.on_deepep_dispatch_low_latency(
            local_count,
            num_trailing_shared_slots=2,
        )

        gatherer.on_deepep_dispatch_low_latency.assert_called_once_with(
            layer_idx=3,
            local_physical_count_of_layer=local_count,
            num_trailing_shared_slots=2,
        )

        recorder.on_deepep_dispatch_normal(
            [3, 5, 101, 103],
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=None,
            num_trailing_shared_slots=2,
        )
        gatherer.on_deepep_dispatch_normal.assert_called_once_with(
            layer_idx=3,
            local_physical_count_of_layer=[3, 5, 101, 103],
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=None,
            num_trailing_shared_slots=2,
        )

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

    def test_deepep_normal_dispatch_forwards_the_layout_contract(self):
        dispatcher = object.__new__(_DeepEPDispatcherImplNormal)
        dispatcher.async_finish = False
        dispatcher.num_experts = 4
        dispatcher.num_trailing_shared_slots = 2
        local_count = [3, 5, 101, 103]
        event = object()
        buffer = Mock()
        buffer.get_dispatch_layout.return_value = (
            None,
            None,
            None,
            None,
            None,
        )
        buffer.dispatch.return_value = (
            torch.empty((0, 8)),
            torch.empty((0, 1), dtype=torch.int64),
            torch.empty((0, 1)),
            local_count,
            object(),
            event,
        )
        dispatcher._get_buffer = Mock(return_value=buffer)
        recorder = Mock()

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep."
                "_deepep_precompile_tp_barrier"
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep."
                "DeepEPConfig.get_instance",
                return_value=SimpleNamespace(normal_dispatch_config=object()),
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep."
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            dispatcher._dispatch_core(
                x=torch.empty((0, 8)),
                topk_ids=torch.empty((0, 1), dtype=torch.int64),
                topk_weights=torch.empty((0, 1)),
                previous_event=None,
            )

        recorder.on_deepep_dispatch_normal.assert_called_once_with(
            local_count,
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=None,
            num_trailing_shared_slots=2,
        )

    def test_mori_low_latency_dispatch_forwards_the_layout_contract(self):
        masked_m = torch.tensor([3, 5, 101, 103], dtype=torch.int32)
        async_ll = object()
        dispatcher = object.__new__(_MoriEPDispatcherImplLowLatency)
        dispatcher.num_trailing_shared_slots = 2
        dispatcher._mori_op = SimpleNamespace(
            config=SimpleNamespace(kernel_type=async_ll),
            dispatch_recv=Mock(),
            local_expert_count=masked_m,
        )
        fake_mori = SimpleNamespace(
            ops=SimpleNamespace(
                EpDispatchCombineKernelType=SimpleNamespace(AsyncLL=async_ll)
            )
        )
        recorder = Mock()
        with (
            patch.dict(sys.modules, {"mori": fake_mori}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moriep."
                "_should_record_expert_distribution",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moriep."
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            dispatcher.dispatch_b(
                hidden_states=torch.empty((0, 8)),
                recv_topk_weights=torch.empty((0, 1)),
                recv_topk_ids=torch.empty((0, 1), dtype=torch.int64),
                recv_scales=None,
                packed_recv_count=masked_m,
                topk_weights=torch.empty((0, 1)),
                topk_ids=torch.empty((0, 1), dtype=torch.int64),
                output_dtype=torch.bfloat16,
            )
        recorder.on_deepep_dispatch_low_latency.assert_called_once_with(
            masked_m,
            num_trailing_shared_slots=2,
        )

    def test_mori_normal_dispatch_forwards_the_layout_contract(self):
        local_count = torch.tensor([3, 5, 101, 103], dtype=torch.int32)
        dispatcher = object.__new__(_MoriEPDispatcherImplNormal)
        dispatcher.num_trailing_shared_slots = 2
        dispatcher._comm_stream = None
        dispatcher._mori_op = SimpleNamespace(
            dispatch=Mock(
                return_value=(
                    torch.empty((0, 8)),
                    torch.empty((0, 1)),
                    None,
                    torch.empty((0, 1), dtype=torch.int64),
                    local_count,
                )
            ),
            local_expert_count=local_count,
        )
        recorder = Mock()

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moriep."
                "_should_record_expert_distribution",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moriep."
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            dispatcher._dispatch_core(
                hidden_states=torch.empty((0, 8)),
                topk_weights=torch.empty((0, 1)),
                topk_ids=torch.empty((0, 1), dtype=torch.int64),
            )

        recorder.on_deepep_dispatch_low_latency.assert_called_once_with(
            local_count,
            num_trailing_shared_slots=2,
        )

    def test_dispatcher_funnel_only_passes_contract_to_active_recorders(self):
        for backend_name, expected in (
            ("deepep", 2),
            ("mooncake", None),
            ("mori", 2),
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
