"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe, mega_moe_sm90

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepGemmMegaMoeApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()

    def tearDown(self):
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        super().tearDown()

    def test_mxf4_buffer_uses_typed_api(self):
        deep_gemm = ModuleType("deep_gemm")
        expected_buffer = object()
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(return_value=expected_buffer)
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                mega_moe,
                "_mega_moe_mma_type",
                return_value="mxf4xmxf4",
                create=True,
            ),
        ):
            actual_buffer = mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
            )

        self.assertIs(actual_buffer, expected_buffer)
        call = deep_gemm.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp8_dispatch", call.kwargs)

    def test_server_flag_selects_mxf4_mma_type(self):
        for enabled, expected in ((False, "fp8xfp4"), (True, "mxf4xmxf4")):
            with self.subTest(enabled=enabled):
                config = SimpleNamespace(
                    moe=SimpleNamespace(enable_w4a4_mxfp4_megamoe=enabled)
                )
                with patch.object(mega_moe, "get_exec", return_value=config):
                    self.assertEqual(mega_moe._mega_moe_mma_type(), expected)

    def test_buffer_cache_separates_mma_types(self):
        deep_gemm = ModuleType("deep_gemm")
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                mega_moe,
                "_mega_moe_mma_type",
                side_effect=("fp8xfp4", "mxf4xmxf4"),
            ),
        ):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                )
                for _ in range(2)
            )

        self.assertEqual(actual_buffers, expected_buffers)
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["mma_type"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["fp8xfp4", "mxf4xmxf4"],
        )

    def test_mxf4_weight_transform_uses_matching_mma_type(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.transform_sf_into_required_layout = MagicMock(
            side_effect=lambda _sf, mn, k, recipe, num_groups, disable_ue8m0_cast: torch.zeros(
                (num_groups, mn, max(1, k // 32)), dtype=torch.int32
            )
        )
        deep_gemm.transform_weights_for_mega_moe = MagicMock(
            side_effect=lambda l1, l2, **_kwargs: (l1, l2)
        )
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_platform",
                return_value=SimpleNamespace(is_sm90=False),
            ),
        ):
            method.process_weights_after_loading(layer)

        call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")

    def test_sm90_mxf4_weight_transform_uses_native_fp4_contract(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        deep_gemm.mega_moe_pre_dispatch_sm90 = MagicMock()
        deep_gemm._C = SimpleNamespace(fp8_fp4_mega_moe_sm90=MagicMock())
        l1 = (torch.ones((1,), dtype=torch.int8), torch.ones((1,), dtype=torch.int32))
        l2 = (torch.ones((1,), dtype=torch.int8), torch.ones((1,), dtype=torch.int32))
        deep_gemm.transform_weights_for_mega_moe_sm90_fp4 = MagicMock(
            return_value=(l1, l2)
        )
        deep_gemm.transform_sf_into_required_layout = MagicMock()

        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.full((1, 32, 4), 127, dtype=torch.uint8),
                requires_grad=False,
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.full((1, 32, 4), 127, dtype=torch.uint8),
                requires_grad=False,
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_platform",
                return_value=SimpleNamespace(is_sm90=True),
            ),
        ):
            method.process_weights_after_loading(layer)

        transform = deep_gemm.transform_weights_for_mega_moe_sm90_fp4
        transform.assert_called_once()
        l1_input, l2_input = transform.call_args.args
        self.assertEqual(l1_input[0].dtype, torch.int8)
        self.assertEqual(l2_input[0].dtype, torch.int8)
        self.assertTrue(torch.all(l1_input[1] == 1.0))
        self.assertTrue(torch.all(l2_input[1] == 1.0))
        deep_gemm.transform_sf_into_required_layout.assert_not_called()
        self.assertTrue(layer._mega_moe_sm90_fp4_weights)
        self.assertTrue(layer._mega_moe_weights_built)

    def test_sm90_fp4_compat_transform_preserves_packed_byte_layout(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        deep_gemm.mega_moe_pre_dispatch_sm90 = MagicMock()
        deep_gemm._C = SimpleNamespace(fp8_fp4_mega_moe_sm90=MagicMock())
        transform = mega_moe_sm90._resolve_sm90_fp4_weight_transform(deep_gemm)

        l1_weight = torch.arange(32 * 4, dtype=torch.uint8).reshape(1, 32, 4)
        l2_weight = torch.arange(8 * 4, dtype=torch.uint8).reshape(1, 8, 4)
        l1_scale = (
            torch.pow(2.0, torch.arange(32, dtype=torch.float32))
            .reshape(1, 32, 1)
            .expand(-1, -1, 4)
            .contiguous()
        )
        l2_scale = (
            torch.pow(2.0, torch.arange(8, dtype=torch.float32))
            .reshape(1, 8, 1)
            .expand(-1, -1, 4)
            .contiguous()
        )

        (l1_weight_out, l1_scale_out), (l2_weight_out, l2_scale_out) = transform(
            (l1_weight, l1_scale), (l2_weight, l2_scale)
        )

        row_order = torch.tensor(
            list(range(0, 8))
            + list(range(16, 24))
            + list(range(8, 16))
            + list(range(24, 32))
        )
        torch.testing.assert_close(
            l1_weight_out, l1_weight.index_select(1, row_order).view(torch.int8)
        )
        torch.testing.assert_close(l2_weight_out, l2_weight.view(torch.int8))
        torch.testing.assert_close(
            l1_scale_out.view(torch.uint8),
            (127 + row_order)
            .to(torch.uint8)
            .reshape(1, 32, 1)
            .expand(-1, -1, 4),
        )
        torch.testing.assert_close(
            l2_scale_out.view(torch.uint8),
            (127 + torch.arange(8))
            .to(torch.uint8)
            .reshape(1, 8, 1)
            .expand(-1, -1, 4),
        )
        self.assertEqual(l1_scale_out.dtype, torch.int32)
        self.assertEqual(l2_scale_out.dtype, torch.int32)
        self.assertTrue(l1_scale_out.is_contiguous())
        self.assertTrue(l2_scale_out.is_contiguous())

    def test_mxf4_pre_dispatch_uses_typed_api(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=True,
        )
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                swiglu_limit=None,
            ),
            experts=experts,
            gate=MagicMock(return_value=torch.empty((1, 8))),
            topk=MagicMock(return_value=topk_output),
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=0,
            routed_scaling_factor=1.0,
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(
                mega_moe,
                "_get_mega_moe_symm_buffer",
                return_value=buffer,
            ),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((1, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=1,
            )

        self.assertTrue(deep_gemm.mega_moe_pre_dispatch.called)
        call = deep_gemm.mega_moe_pre_dispatch.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp4_acts", call.kwargs)

    def test_mxf4_l1_uses_packed_gate_up_interleave(self):
        source = torch.arange(32).reshape(1, 32)
        expected = torch.tensor(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
            ]
        ).reshape(1, 32)

        actual = mega_moe._interleave_mega_moe_gate_up(source, gran=16)

        torch.testing.assert_close(actual, expected)

    def test_sm90_fp4_availability_requires_marker_and_native_kernel(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        deep_gemm.mega_moe_pre_dispatch_sm90 = MagicMock()
        deep_gemm._C = SimpleNamespace(fp8_fp4_mega_moe_sm90=MagicMock())
        experts = SimpleNamespace(_mega_moe_sm90_fp4_weights=True)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe_sm90, "_device_sm", 90),
        ):
            self.assertTrue(mega_moe_sm90.is_sm90_fp4_mega_moe_available(experts))
            del deep_gemm._C.fp8_fp4_mega_moe_sm90
            self.assertFalse(mega_moe_sm90.is_sm90_fp4_mega_moe_available(experts))

    def test_sm90_fp4_dispatch_uses_fp8_fp4_kernel(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch_sm90 = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        deep_gemm.fp8_mega_moe = MagicMock()
        experts = SimpleNamespace(
            _mega_moe_sm90_fp4_weights=True,
            should_fuse_routed_scaling_factor_in_topk=True,
            mega_l1_weights=(torch.empty(0), torch.empty(0)),
            mega_l2_weights=(torch.empty(0), torch.empty(0)),
        )
        moe = SimpleNamespace(
            experts=experts,
            config=SimpleNamespace(hidden_size=4, swiglu_limit=None),
            routed_scaling_factor=1.0,
        )
        buffer = SimpleNamespace(
            x=torch.empty((1, 4)),
            x_sf=torch.empty((1, 1)),
            topk_idx=torch.empty((1, 1), dtype=torch.int32),
            topk_weights=torch.empty((1, 1)),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                mega_moe_sm90,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(enable_w4a4_mxfp4_megamoe=False)
                ),
            ),
        ):
            output = mega_moe_sm90.run_sm90_mega_routed(
                moe,
                torch.ones((1, 4), dtype=torch.bfloat16),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.ones((1, 1)),
                buffer,
                1,
            )

        self.assertEqual(output.shape, (1, 4))
        deep_gemm.mega_moe_pre_dispatch_sm90.assert_called_once()
        deep_gemm.fp8_fp4_mega_moe.assert_called_once()
        self.assertEqual(
            deep_gemm.fp8_fp4_mega_moe.call_args.kwargs["recipe"], (1, 1, 32)
        )
        deep_gemm.fp8_mega_moe.assert_not_called()

    def test_sm90_layer_probes_bracket_dispatch_and_routed_kernel(self):
        events = []
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch_sm90 = MagicMock(
            side_effect=lambda *_args, **_kwargs: events.append("pre_dispatch")
        )
        deep_gemm.fp8_fp4_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: events.append("routed")
        )
        deep_gemm.fp8_mega_moe = MagicMock()
        experts = SimpleNamespace(
            _mega_moe_sm90_fp4_weights=True,
            should_fuse_routed_scaling_factor_in_topk=True,
            mega_l1_weights=(torch.empty(0), torch.empty(0)),
            mega_l2_weights=(torch.empty(0), torch.empty(0)),
        )
        moe = SimpleNamespace(
            experts=experts,
            config=SimpleNamespace(hidden_size=4, swiglu_limit=None),
            routed_scaling_factor=1.0,
        )
        buffer = SimpleNamespace(
            x=torch.empty((1, 4)),
            x_sf=torch.empty((1, 1)),
            topk_idx=torch.empty((1, 1), dtype=torch.int32),
            topk_weights=torch.empty((1, 1)),
        )
        forward_batch = object()

        def record_probe(actual_batch, checkpoint):
            self.assertIs(actual_batch, forward_batch)
            events.append(checkpoint)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                mega_moe_sm90,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(enable_w4a4_mxfp4_megamoe=False)
                ),
            ),
            patch.object(
                mega_moe_sm90,
                "maybe_sync_eagle_cuda_debug",
                side_effect=record_probe,
            ),
        ):
            mega_moe_sm90.run_sm90_mega_routed(
                moe,
                torch.ones((1, 4), dtype=torch.bfloat16),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.ones((1, 1)),
                buffer,
                1,
                forward_batch=forward_batch,
            )

        self.assertEqual(
            events,
            [
                "pre_dispatch",
                "after_megamoe_pre_dispatch",
                "routed",
                "after_megamoe_routed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
