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

from sglang.srt.layers.moe import mega_moe

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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

    def test_model_adapter_reads_minimax_moe_contract(self):
        expected_logits = torch.tensor([[1.0, 2.0]])
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_act="swigluoai",
                swiglu_alpha=1.702,
                swiglu_limit=7.0,
                intermediate_size=384,
            ),
            _compute_router_logits=MagicMock(return_value=expected_logits),
            routed_scaling_factor=2.0,
            experts=SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=True,
            ),
            topk=SimpleNamespace(
                topk_config=SimpleNamespace(apply_routed_scaling_factor_on_output=True)
            ),
        )

        actual_logits = mega_moe._compute_mega_moe_router_logits(
            moe, torch.zeros(1, 3), forward_batch=object()
        )

        self.assertIs(actual_logits, expected_logits)
        self.assertEqual(mega_moe._get_moe_intermediate_size(moe.config), 384)
        self.assertEqual(
            mega_moe._get_mega_moe_activation_params(moe.config),
            ("swigluoai", 1.702, 1.0, 7.0),
        )
        self.assertEqual(mega_moe._get_mega_moe_routed_scaling_factor(moe), 1.0)

    def test_deepseek_expert_scaling_contract_remains_authoritative(self):
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=False,
            ),
            routed_scaling_factor=2.5,
            topk=SimpleNamespace(
                topk_config=SimpleNamespace(apply_routed_scaling_factor_on_output=True)
            ),
        )

        self.assertEqual(mega_moe._get_mega_moe_routed_scaling_factor(moe), 2.5)

    def test_non_sm90_rejects_parameterized_swiglu(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                hidden_act="swigluoai",
                swiglu_alpha=1.702,
                swiglu_limit=7.0,
                num_experts_per_tok=2,
                intermediate_size=8,
            ),
            experts=SimpleNamespace(
                num_experts=8,
                mega_l1_weights=object(),
                mega_l2_weights=object(),
                should_fuse_routed_scaling_factor_in_topk=True,
            ),
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
            topk=SimpleNamespace(
                topk_config=SimpleNamespace(apply_routed_scaling_factor_on_output=True)
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer),
            patch.object(mega_moe, "mega_moe_pre_dispatch"),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
            self.assertRaisesRegex(RuntimeError, "only supported on SM90"),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((0, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=0,
            )

        deep_gemm.fp8_fp4_mega_moe.assert_not_called()

    def test_sm90_forwards_parameterized_swiglu(self):
        from sglang.srt.layers.moe import mega_moe_sm90

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch_sm90 = MagicMock()
        deep_gemm.fp8_mega_moe = MagicMock()
        experts = SimpleNamespace(
            mega_l1_weights=object(),
            mega_l2_weights=object(),
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            experts=experts,
        )
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            mega_moe_sm90.run_sm90_mega_routed(
                moe,
                torch.zeros((1, 4)),
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([[0.6, 0.4]]),
                buffer,
                num_tokens=1,
                routed_scaling_factor=1.0,
                activation="swigluoai",
                activation_alpha=1.702,
                activation_up_bias=1.0,
                activation_clamp=7.0,
            )

        call = deep_gemm.fp8_mega_moe.call_args
        self.assertEqual(call.kwargs["activation"], "swigluoai")
        self.assertEqual(call.kwargs["activation_alpha"], 1.702)
        self.assertEqual(call.kwargs["activation_up_bias"], 1.0)
        self.assertEqual(call.kwargs["activation_clamp"], 7.0)

    def test_minimax_megamoe_fails_closed(self):
        from sglang.srt.models import minimax_m3

        backend = SimpleNamespace(
            is_megamoe=lambda: True,
            is_deepep=lambda: False,
        )
        moe = object.__new__(minimax_m3.MiniMaxM3MoE)

        with (
            patch.object(minimax_m3, "get_moe_a2a_backend", return_value=backend),
            patch.object(mega_moe, "should_use_mega_moe", return_value=False),
            self.assertRaisesRegex(RuntimeError, "refusing to fall back"),
        ):
            moe.forward(torch.zeros(1, 1), forward_batch=object())


if __name__ == "__main__":
    unittest.main()
