"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDeepGemmMegaMoeApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        self.deep_gemm = ModuleType("deep_gemm")
        self.deep_gemm.num_sms = 132
        self.deep_gemm.get_num_sms = MagicMock(
            side_effect=lambda: self.deep_gemm.num_sms
        )
        self.deep_gemm.set_num_sms = MagicMock(
            side_effect=lambda num_sms: setattr(self.deep_gemm, "num_sms", num_sms)
        )
        max_num_sms = patch.object(mega_moe, "_mega_moe_max_num_sms", return_value=130)
        self.max_num_sms = max_num_sms.start()
        self.addCleanup(max_num_sms.stop)

    def tearDown(self):
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        super().tearDown()

    def test_mxf4_buffer_uses_typed_api(self):
        deep_gemm = self.deep_gemm
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
        deep_gemm = self.deep_gemm
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

    def test_buffer_cache_uses_effective_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                num_sms=deep_gemm.get_num_sms()
            )
        )
        group = object()
        buffers = []

        for current_num_sms, expected_num_sms in (
            (132, 130),
            (131, 130),
            (96, 96),
            (97, 96),
            (132, 130),
        ):
            with self.subTest(current_num_sms=current_num_sms):
                deep_gemm.set_num_sms(current_num_sms)
                buf = self._get_test_buffer(group)
                buffers.append(buf)
                self.assertEqual(buf.num_sms, expected_num_sms)
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)
                with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
                    self.assertEqual(buf.num_sms, deep_gemm.get_num_sms())
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)

        self.assertIs(buffers[0], buffers[1])
        self.assertIs(buffers[0], buffers[4])
        self.assertIs(buffers[2], buffers[3])
        self.assertIsNot(buffers[0], buffers[2])
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)

    def test_buffer_allocation_without_sm_override(self):
        deep_gemm = self.deep_gemm
        self.max_num_sms.return_value = None
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: deep_gemm.get_num_sms()
        )

        buf = self._get_test_buffer(object())
        self.assertEqual(buf, 132)
        deep_gemm.set_num_sms.assert_not_called()

    def test_buffer_allocation_failure_restores_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=RuntimeError("allocation failed")
        )

        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self._get_test_buffer(object())

        self.assertEqual(deep_gemm.get_num_sms(), 132)
        self.assertEqual(deep_gemm.set_num_sms.call_args_list, [call(130), call(132)])
        self.assertEqual(mega_moe._MEGA_MOE_SYMM_BUFFER, {})

    def test_buffer_cache_separates_native_shared_expert_count(self):
        deep_gemm = self.deep_gemm
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
        ):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                    num_shared_experts=num_shared_experts,
                )
                for num_shared_experts in (0, 1)
            )

        self.assertEqual(actual_buffers, expected_buffers)
        self.assertEqual(
            [
                call.kwargs["num_shared_experts"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            [0, 1],
        )

    def test_builds_native_shared_weights_without_routed_slot_remap(self):
        deep_gemm = ModuleType("deep_gemm")
        transformed_l1 = object()
        transformed_l2 = object()
        deep_gemm.transform_weights_for_mega_moe = MagicMock(
            return_value=(transformed_l1, transformed_l2)
        )
        gate_up = SimpleNamespace(
            weight=torch.nn.Parameter(torch.zeros((32, 16)), requires_grad=False),
            weight_scale_inv=torch.nn.Parameter(
                torch.ones((1, 1)), requires_grad=False
            ),
        )
        down = SimpleNamespace(
            weight=torch.nn.Parameter(torch.zeros((16, 16)), requires_grad=False),
            weight_scale_inv=torch.nn.Parameter(
                torch.ones((1, 1)), requires_grad=False
            ),
        )
        moe = SimpleNamespace(
            is_deepseek_v4=True,
            num_fused_shared_experts=0,
            n_shared_experts=1,
            shared_experts=SimpleNamespace(gate_up_proj=gate_up, down_proj=down),
            shared_experts_is_fp8=True,
            shared_experts_weight_block_size=[128, 128],
        )
        requant_results = ((object(), object()), (object(), object()))

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(
                mega_moe,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_megamoe=lambda: True),
            ),
            patch.object(
                mega_moe,
                "_prepare_mega_moe_shared_weight_ue8m0",
                side_effect=requant_results,
            ) as requant,
        ):
            built = mega_moe.build_mega_moe_shared_expert_weights(moe)

        self.assertTrue(built)
        self.assertEqual(requant.call_count, 2)
        self.assertIs(moe.mega_shared_l1_weights, transformed_l1)
        self.assertIs(moe.mega_shared_l2_weights, transformed_l2)
        self.assertEqual(moe.num_fused_shared_experts, 0)
        transform_call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(transform_call.args, requant_results)
        self.assertEqual(transform_call.kwargs, {"mma_type": "fp8xfp4"})

    def test_native_shared_weight_expands_k128_scale_for_k32_consumption(self):
        weight = torch.zeros((128, 512), dtype=torch.float8_e4m3fn)
        raw_scale = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        packed_scale = torch.zeros((128, 1), dtype=torch.int32)
        packed_scale.format_ue8m0 = True
        unpacked_scale = raw_scale.clone()
        requantized_weight = torch.zeros_like(weight)
        requantized_scale = torch.ones((128, 1), dtype=torch.int32)
        native_scale = torch.empty_strided((128, 4), (1, 128), dtype=torch.int32)

        for scale, should_unpack in (
            (raw_scale, False),
            (packed_scale, True),
        ):
            with (
                self.subTest(should_unpack=should_unpack),
                patch(
                    "sglang.srt.layers.quantization.fp8_utils.inverse_transform_scale_ue8m0",
                    return_value=unpacked_scale,
                ) as unpack,
                patch(
                    "sglang.srt.layers.quantization.fp8_utils.requant_weight_ue8m0",
                    return_value=(requantized_weight, requantized_scale),
                ) as requant,
                patch(
                    "sglang.srt.layers.quantization.fp8_utils.transform_scale_ue8m0",
                    return_value=native_scale,
                ) as transform,
            ):
                actual = mega_moe._prepare_mega_moe_shared_weight_ue8m0(
                    weight, scale, [128, 128]
                )

            self.assertIs(actual[0], weight if should_unpack else requantized_weight)
            self.assertIs(actual[1], native_scale)
            if should_unpack:
                requant.assert_not_called()
                unpack.assert_called_once_with(packed_scale, mn=128)
            else:
                requant.assert_called_once_with(weight, raw_scale, [128, 128])
                unpack.assert_called_once_with(requantized_scale, mn=128)
            transform.assert_called_once()
            torch.testing.assert_close(
                transform.call_args.args[0],
                torch.tensor(
                    [
                        [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            2.0,
                            2.0,
                            2.0,
                            2.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                        ]
                    ]
                ),
            )
            self.assertEqual(transform.call_args.kwargs, {"mn": 128})

    def test_native_shared_weight_drops_packed_k_padding_before_expansion(self):
        weight = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)
        packed_scale = torch.zeros((128, 1), dtype=torch.int32)
        packed_scale.format_ue8m0 = True
        unpacked_with_padding = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        native_scale = torch.empty_strided((128, 2), (1, 128), dtype=torch.int32)

        with (
            patch(
                "sglang.srt.layers.quantization.fp8_utils.inverse_transform_scale_ue8m0",
                return_value=unpacked_with_padding,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils.transform_scale_ue8m0",
                return_value=native_scale,
            ) as transform,
        ):
            mega_moe._prepare_mega_moe_shared_weight_ue8m0(
                weight, packed_scale, [128, 128]
            )

        torch.testing.assert_close(
            transform.call_args.args[0],
            torch.tensor([[1.0] * 4 + [2.0] * 4]),
        )

    def test_native_shared_weights_reject_w4a4_routed_mode(self):
        moe = SimpleNamespace(
            is_deepseek_v4=True,
            num_fused_shared_experts=0,
            n_shared_experts=1,
            shared_experts=object(),
            shared_experts_is_fp8=True,
            shared_experts_weight_block_size=[128, 128],
        )

        with (
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(
                mega_moe,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_megamoe=lambda: True),
            ),
        ):
            self.assertFalse(mega_moe.build_mega_moe_shared_expert_weights(moe))

        self.assertFalse(hasattr(moe, "_mega_moe_shared_weights_built"))

    def test_mxf4_weight_transform_uses_matching_mma_type(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.transform_sf_into_required_layout = MagicMock(
            side_effect=lambda _sf, mn, k, recipe, num_groups, disable_ue8m0_cast: (
                torch.zeros((num_groups, mn, max(1, k // 32)), dtype=torch.int32)
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
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
        ):
            method.process_weights_after_loading(layer)

        call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")

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
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
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
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=object())
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

    def test_run_mega_routed_experts_generic_entry(self):
        deep_gemm = ModuleType("deep_gemm")
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
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        hidden_states = torch.zeros((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(mega_moe, "mega_moe_pre_dispatch") as pre_dispatch,
            patch.object(
                mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
            ) as get_buffer,
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
            ),
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                hidden_states,
                topk_ids,
                topk_weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                activation_clamp=7.0,
                routed_scaling_factor=1.0,
            )

        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, torch.bfloat16)
        buf_call = get_buffer.call_args
        self.assertEqual(buf_call.kwargs.get("num_topk"), 2)
        self.assertEqual(buf_call.kwargs.get("hidden"), 4)
        self.assertEqual(buf_call.kwargs.get("intermediate_hidden"), 8)
        # The kernel wants int32 ids and fp32 weights regardless of the router dtype.
        ids_arg, weights_arg = pre_dispatch.call_args.args[1:3]
        self.assertEqual(ids_arg.dtype, torch.int32)
        self.assertEqual(weights_arg.dtype, torch.float32)
        mega_call = deep_gemm.fp8_fp4_mega_moe.call_args
        self.assertIs(mega_call.args[1], experts.mega_l1_weights)
        self.assertIs(mega_call.args[2], experts.mega_l2_weights)
        self.assertEqual(mega_call.kwargs.get("activation_clamp"), 7.0)

    def test_shape_check_rejects_unaligned_intermediate(self):
        # Qwen3-30B-A3B: intermediate 768 leaves a 24-byte scale row.
        with self.assertRaisesRegex(ValueError, "multiples of 512"):
            mega_moe.check_mega_moe_shapes(2048, 768, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1536, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1024, "mxf4xmxf4")
        # 768 is a multiple of 256, so the NVFP4 (g16) rule accepts it.
        mega_moe.check_mega_moe_shapes(2048, 768, "nvfp4xnvfp4")
        with self.assertRaisesRegex(ValueError, "multiples of 256"):
            mega_moe.check_mega_moe_shapes(2048, 384, "nvfp4xnvfp4")

    def test_native_shared_path_wires_scale_layout_weights_and_routed_scaling(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.get_block_m_for_mega_moe = MagicMock(return_value=32)
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        shared_l1_sf = object()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
            shared_l1_acts_sf=shared_l1_sf,
            num_max_tokens_per_rank=64,
        )
        device_group = SimpleNamespace(size=lambda: 2)
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=False,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
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
            ),
            experts=experts,
            gate=MagicMock(return_value=torch.empty((1, 8))),
            topk=MagicMock(return_value=topk_output),
            is_hash=False,
            num_fused_shared_experts=0,
            n_shared_experts=1,
            layer_id=0,
            routed_scaling_factor=2.0,
            _mega_moe_shared_weights_built=True,
            mega_shared_l1_weights=object(),
            mega_shared_l2_weights=object(),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(
                mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
            ) as get_buffer,
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            patch.object(mega_moe, "mega_moe_pre_dispatch") as pre_dispatch,
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=device_group)
                ),
            ),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((1, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=1,
            )

        self.assertEqual(get_buffer.call_args.kwargs["num_shared_experts"], 1)
        deep_gemm.get_block_m_for_mega_moe.assert_called_once_with(
            2, 8, 64, 1, 2, "fp8xfp4"
        )
        self.assertIs(pre_dispatch.call_args.kwargs["shared_l1_acts_sf"], shared_l1_sf)
        self.assertEqual(pre_dispatch.call_args.kwargs["shared_block_m"], 32)
        torch.testing.assert_close(
            pre_dispatch.call_args.args[2], torch.tensor([[1.2, 0.8]])
        )
        kernel_call = deep_gemm.fp8_fp4_mega_moe.call_args
        self.assertIs(
            kernel_call.kwargs["shared_l1_weights"],
            moe.mega_shared_l1_weights,
        )
        self.assertIs(
            kernel_call.kwargs["shared_l2_weights"],
            moe.mega_shared_l2_weights,
        )

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

    def _get_test_buffer(self, group):
        with (
            patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
        ):
            return mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
            )


if __name__ == "__main__":
    unittest.main()
