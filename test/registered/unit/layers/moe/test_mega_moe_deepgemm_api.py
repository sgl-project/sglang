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

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend, mega_moe
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_exec, get_flags, get_parallel

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

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffer = mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="mxf4xmxf4",
            )

        self.assertIs(actual_buffer, expected_buffer)
        call = deep_gemm.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp8_dispatch", call.kwargs)

    def test_buffer_cache_separates_mma_types(self):
        deep_gemm = self.deep_gemm
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                    mma_type=mma_type,
                )
                for mma_type in ("fp8xfp4", "mxf4xmxf4")
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
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
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

    def test_draft_layers_keep_their_own_w4a4_choice(self):
        """Draft MegaMoE layers must not follow the target's MMA type once built.

        The draft forward runs outside draft_model_build_scope, so a layer that
        re-read the global flag would pair draft weights with the wrong kernel.
        """
        for target_flag, draft_flag, expected_target, expected_draft in (
            (True, False, "mxf4xmxf4", "fp8xfp4"),
            (True, None, "mxf4xmxf4", "mxf4xmxf4"),
            (False, True, "fp8xfp4", "mxf4xmxf4"),
            (False, None, "fp8xfp4", "fp8xfp4"),
        ):
            with self.subTest(
                enable_w4a4_mxfp4_megamoe=target_flag,
                speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
            ):
                with get_context().override_server_args(
                    model_path="dummy",
                    enable_w4a4_mxfp4_megamoe=target_flag,
                    speculative_enable_w4a4_mxfp4_megamoe=draft_flag,
                ):
                    target = self._build_fused_moe()
                    with draft_model_build_scope():
                        draft = self._build_fused_moe()

                    self.assertEqual(
                        get_exec().moe.enable_w4a4_mxfp4_megamoe, target_flag
                    )
                    self.assertEqual(
                        mega_moe._mega_moe_mma_type(target), expected_target
                    )
                    self.assertEqual(mega_moe._mega_moe_mma_type(draft), expected_draft)

    def test_draft_build_scope_restores_w4a4_on_exception(self):
        """A failed draft build must not leave the target on the draft's MMA type."""
        with get_context().override_server_args(
            model_path="dummy",
            enable_w4a4_mxfp4_megamoe=False,
            speculative_enable_w4a4_mxfp4_megamoe=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "draft build failed"):
                with draft_model_build_scope():
                    self.assertTrue(get_exec().moe.enable_w4a4_mxfp4_megamoe)
                    raise RuntimeError("draft build failed")

            self.assertFalse(get_exec().moe.enable_w4a4_mxfp4_megamoe)
            target = self._build_fused_moe()
            self.assertEqual(mega_moe._mega_moe_mma_type(target), "fp8xfp4")

    def _build_fused_moe(self):
        method = UnquantizedFusedMoEMethod()
        with (
            patch.object(method, "create_weights"),
            patch.object(method, "create_moe_runner"),
            patch.object(
                fused_moe_layer_module,
                "create_moe_dispatcher",
                return_value=SimpleNamespace(),
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.AUTO,
                a2a_backend=MoeA2ABackend.MEGAMOE,
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            return fused_moe_layer_module.FusedMoE(
                num_experts=2,
                hidden_size=4,
                intermediate_size=8,
                layer_id=0,
                quant_method=method,
            )

    def test_minimax_replicated_shared_experts_need_no_fused_weight_attrs(self):
        hidden_states = torch.ones((2, 4))
        routed_output = torch.full((2, 4), 2.0)
        shared_output = torch.full((2, 4), 3.0)
        expected = routed_output + shared_output
        moe = SimpleNamespace(
            alt_stream=None,
            num_fused_shared_experts=0,
            _forward_shared_experts=MagicMock(return_value=shared_output),
        )

        with patch.object(
            mega_moe, "_run_mega_routed", return_value=routed_output
        ) as run_routed:
            actual = mega_moe.forward_mega_moe(
                moe, hidden_states, forward_batch=object()
            )

        torch.testing.assert_close(actual, expected)
        moe._forward_shared_experts.assert_called_once_with(hidden_states)
        run_routed.assert_called_once()

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
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
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
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            mega_moe_sm90.run_sm90_mega_routed(
                experts,
                torch.zeros((1, 4)),
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([[0.6, 0.4]]),
                buffer,
                num_tokens=1,
                hidden_size=4,
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

    def _get_test_buffer(self, group):
        with patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}):
            return mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
                mma_type="fp8xfp4",
            )


if __name__ == "__main__":
    unittest.main()
