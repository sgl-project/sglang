"""The radix-4 router hands its fused MoE sort to AITER only on the batch sizes
where AITER's fused_moe will consume it: those whose resolved config is the
BM=16 aux-sort pipeline."""

import sys
import unittest
from enum import Enum, IntEnum
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe import route_quant_handoff, route_sort
from sglang.srt.layers.moe.moe_runner import aiter_route_sort
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS = 896
TOPK = 16
MODEL_DIM = 64
INTER_DIM = 32


class _ActivationType(IntEnum):
    Silu = 0
    Situv2 = 7


class _QuantType(IntEnum):
    No = 0
    per_1x32 = 5


class _GateMode(str, Enum):
    SEPARATED = "separated"
    INTERLEAVE = "interleave"


def _metadata(block_m, output_aux=True):
    return SimpleNamespace(block_m=block_m, output_aux=output_aux, full_impl=None)


def _fake_cfg_api(get_2stage_cfgs):
    return SimpleNamespace(
        ActivationType=_ActivationType,
        QuantType=_QuantType,
        GateMode=_GateMode,
        get_2stage_cfgs=get_2stage_cfgs,
        get_inter_dim=lambda w1_shape, w2_shape: (
            w1_shape[0],
            w2_shape[1],
            w2_shape[2],
        ),
        get_padded_M=lambda m: 1 << (m - 1).bit_length(),
        normalize_activation_params=lambda activation, beta, linear_beta, limit: (
            1.0 if beta is None else beta,
            1.0 if linear_beta is None else linear_beta,
            None,
        ),
        quant_remap={},
        resolve_activation_dtype=lambda *args, **kwargs: torch.bfloat16,
        presort_block_m=16,
    )


def _fused_moe_call(m, **overrides):
    """The keyword arguments the AITER runner passes fused_moe for a K3 layer."""
    call = dict(
        hidden_states=torch.zeros((m, MODEL_DIM), dtype=torch.bfloat16),
        w1=torch.zeros((NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM // 2)),
        w2=torch.zeros((NUM_EXPERTS, MODEL_DIM, INTER_DIM // 2)),
        topk_weight=torch.zeros((m, TOPK)),
        topk_ids=torch.zeros((m, TOPK), dtype=torch.int32),
        quant_type=_QuantType.per_1x32,
        activation=_ActivationType.Situv2,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
        bias1=None,
        bias2=None,
        expert_mask=None,
        doweight_stage1=False,
        hidden_pad=0,
        intermediate_pad=0,
        gate_mode=_GateMode.SEPARATED.value,
        beta=1.5,
        linear_beta=7.0,
        moe_sorting_dispatch_policy=0,
    )
    call.update(overrides)
    return call


def _bm16_up_to_64(**kwargs):
    return _metadata(16 if kwargs["token"] <= 64 else 32)


def _patch_cfg_api(lookup):
    return patch.object(
        aiter_route_sort, "_cfg_api", return_value=_fake_cfg_api(lookup)
    )


def _route(consumer, m, registrations):
    """One routing call of a layer that staged consumer, with the kernel stubbed."""
    ids = torch.zeros((m, TOPK), dtype=torch.int32)
    weights = torch.zeros((m, TOPK))
    aux = {"M": m, "topk_ids_ptr": ids.data_ptr()}
    route_with_sort = MagicMock(return_value=(weights, ids, aux))
    sort_ws = SimpleNamespace(get=lambda device: torch.zeros(4, dtype=torch.int32))
    with (
        patch.object(aiter_route_sort, "_registrar", return_value=registrations.append),
        patch.object(route_sort, "get_buffer", return_value=sort_ws),
        patch.object(
            route_sort.moe_route_radix4, "route_radix4_with_sort", route_with_sort
        ),
    ):
        route_sort.stage(consumer)
        try:
            route_quant_handoff.bind_select_experts(None, fused_routing_ok=True)
            hit = route_sort.try_route_sort_fused(
                torch.zeros((m, NUM_EXPERTS)),
                torch.zeros(NUM_EXPERTS),
                topk=TOPK,
                renormalize=True,
                routed_scaling_factor=2.5,
            )
        finally:
            route_quant_handoff.clear()
    return hit, route_with_sort.call_count


class TestAiterRouteSortConsumer(CustomTestCase):
    def setUp(self):
        self.lookup = MagicMock(side_effect=_bm16_up_to_64)
        patcher = _patch_cfg_api(self.lookup)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_registers_the_sort_only_where_aiter_picks_bm16(self):
        """AITER sorts again on a BM=32 batch, so the fused sort is skipped there."""
        consumer = aiter_route_sort.AiterRouteSortConsumer()
        consumer.observe(_fused_moe_call(1))
        registrations = []
        hit, launches = _route(consumer, 16, registrations)
        self.assertIsNotNone(hit)
        self.assertEqual(launches, 1)
        hit, launches = _route(consumer, 128, registrations)
        self.assertIsNone(hit)
        self.assertEqual(launches, 0)
        self.assertEqual([r["M"] for r in registrations], [16])

    def test_declines_before_it_has_seen_the_layers_fused_moe_call(self):
        registrations = []
        hit, _ = _route(aiter_route_sort.AiterRouteSortConsumer(), 16, registrations)
        self.assertIsNone(hit)
        self.assertEqual(registrations, [])
        self.lookup.assert_not_called()

    def test_asks_aiter_once_per_batch_size(self):
        """A host-only cached answer per M, so a captured graph keeps its own."""
        consumer = aiter_route_sort.AiterRouteSortConsumer()
        consumer.observe(_fused_moe_call(4))
        consumer.observe(_fused_moe_call(8, a1_scale=torch.ones(1)))  # first call wins
        self.assertEqual(
            [consumer.takes_sort(m) for m in (16, 16, 128, 128, 16)],
            [True, True, False, False, True],
        )
        self.assertEqual(
            [c.kwargs["token"] for c in self.lookup.call_args_list], [16, 128]
        )
        self.assertFalse(self.lookup.call_args.kwargs["has_activation_scales"])

    def test_calls_aiter_would_not_take_a_presort_on_decline(self):
        for overrides in (
            {"expert_mask": torch.ones(NUM_EXPERTS)},
            {"block_size_M": 32},
            {"activation": _ActivationType.Silu, "swiglu_limit": 7.0},
        ):
            with self.subTest(overrides=sorted(overrides)):
                consumer = aiter_route_sort.AiterRouteSortConsumer()
                consumer.observe(_fused_moe_call(1, **overrides))
                self.assertFalse(consumer.takes_sort(1))
        self.lookup.assert_not_called()


class TestAiterRouteSortConsumerAiterAnswers(CustomTestCase):
    def test_bm16_config_without_the_aux_sort_pipeline_declines(self):
        """Only the aux-sort pipeline reads a presort: AITER's A16W4 SiTUv2 rows
        are BM=16 without it, and a full_impl config returns before any sort."""
        for metadata in (
            _metadata(16, output_aux=False),
            SimpleNamespace(block_m=16, output_aux=True, full_impl=object()),
        ):
            with (
                self.subTest(metadata=metadata),
                _patch_cfg_api(MagicMock(return_value=metadata)),
            ):
                consumer = aiter_route_sort.AiterRouteSortConsumer()
                consumer.observe(_fused_moe_call(1))
                self.assertFalse(consumer.takes_sort(1))

    def test_a_config_aiter_rejects_turns_the_layer_off(self):
        lookup = MagicMock(side_effect=NotImplementedError("no backend"))
        with _patch_cfg_api(lookup):
            consumer = aiter_route_sort.AiterRouteSortConsumer()
            consumer.observe(_fused_moe_call(1))
            self.assertFalse(consumer.takes_sort(1))
            self.assertFalse(consumer.takes_sort(2))
        self.assertEqual(lookup.call_count, 1)

    def test_wrong_lookup_arguments_are_not_swallowed(self):
        lookup = MagicMock(side_effect=TypeError("unexpected keyword argument"))
        with _patch_cfg_api(lookup):
            consumer = aiter_route_sort.AiterRouteSortConsumer()
            consumer.observe(_fused_moe_call(1))
            with self.assertRaises(TypeError):
                consumer.takes_sort(1)

    def test_aiter_without_the_config_lookup_is_unsupported(self):
        """An AITER with register_presorted_moe but no get_2stage_cfgs cannot say
        which batches use the sort, so no layer asks for it."""
        fake_aiter = ModuleType("aiter")
        fake_aiter.__path__ = []
        fake_aiter.ActivationType = _ActivationType
        fake_aiter.QuantType = _QuantType
        fake_fused_moe = ModuleType("aiter.fused_moe")
        fake_fused_moe.register_presorted_moe = lambda aux: None
        with patch.dict(
            sys.modules, {"aiter": fake_aiter, "aiter.fused_moe": fake_fused_moe}
        ):
            self.assertIsNone(aiter_route_sort._cfg_api.__wrapped__())
        with (
            patch.object(aiter_route_sort, "_registrar", return_value=lambda aux: None),
            patch.object(aiter_route_sort, "_cfg_api", return_value=None),
        ):
            self.assertFalse(aiter_route_sort.available())


if __name__ == "__main__":
    unittest.main()
