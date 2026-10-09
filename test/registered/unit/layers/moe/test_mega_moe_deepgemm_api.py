"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe import mega_gate as mega_gate_runtime
from sglang.srt.layers.moe import mega_moe
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.topk import TopKConfig
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
            gate=MagicMock(
                return_value=torch.empty((1, 8)), e_score_correction_bias_vl=None
            ),
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

    def test_v41_megamoe_selects_image_bias_on_local_rows(self):
        """Image rows must retain their router bias after attention-TP sharding.

        Equal logits make the two biases select disjoint experts. A numerical
        expert boundary exposes accidentally routing image rows as text, and
        padded rows must contribute nothing.
        """
        for fused_shared in (0, 1):
            for scale_in_topk in (False, True):
                with self.subTest(
                    fused_shared=fused_shared, scale_in_topk=scale_in_topk
                ):
                    moe = self._v41_moe(fused_shared, scale_in_topk)
                    ids = torch.tensor([7, 99, 99])
                    hidden = torch.ones((3, 4))
                    batch = SimpleNamespace(
                        moe_num_token_non_padded=lambda: torch.tensor(
                            2, dtype=torch.int32
                        )
                    )

                    def run_experts(_experts, x, topk_ids, weights, **kwargs):
                        # Routed experts multiply x by their physical ID + 1;
                        # the rank-1 shared slot (ID 5) is an identity expert.
                        factors = (topk_ids + 1).float()
                        factors[topk_ids == 5] = 1.0
                        self.assertTrue((topk_ids[2] == -1).all())
                        return x * (
                            (factors * weights).sum(-1, keepdim=True)
                            * kwargs["routed_scaling_factor"]
                        )

                    with (
                        get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
                        get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
                        patch.object(
                            mega_moe.ExpertLocationDispatchInfo,
                            "init_new",
                            return_value=None,
                        ),
                        patch.object(mega_moe, "run_mega_routed_experts", run_experts),
                        # The fused padded-row fill requires Triton/GPU.
                        patch(
                            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                            return_value=False,
                        ),
                        patch.object(topk_module, "_is_cuda", False),
                        patch.object(
                            topk_module, "_can_fuse_padded_region", return_value=False
                        ),
                    ):
                        out = mega_moe._run_mega_routed(
                            moe, hidden, batch, ids, num_tokens=3
                        )
                        text_out = mega_moe._run_mega_routed(
                            moe, hidden, batch, None, num_tokens=3
                        )
                    # Text selects logical 0/1; image selects logical 2/3. With
                    # a shared slot, rank-1 routed IDs shift by one (2/3 -> 3/4).
                    expected = torch.tensor(
                        [3.0 + fused_shared, 7.0 + 3.0 * fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(out, expected)
                    text_expected = torch.tensor(
                        [3.0 + fused_shared, 3.0 + fused_shared, 0.0]
                    )[:, None].expand_as(hidden)
                    torch.testing.assert_close(text_out, text_expected)

    def test_v41_megamoe_empty_rank_still_participates(self):
        """An idle DP rank must reach A2A combine without running its router."""
        moe = self._v41_moe(0, False)
        hidden = torch.empty((0, 4))

        def run_experts(_experts, x, ids, weights, **kwargs):
            self.assertIsNone(ids)
            self.assertIsNone(weights)
            # The collective boundary owns the result even on an idle rank.
            return x.new_empty((0, kwargs["hidden_size"]))

        with patch.object(mega_moe, "run_mega_routed_experts", run_experts):
            out = mega_moe._run_mega_routed(
                moe, hidden, None, torch.empty(0, dtype=torch.long), num_tokens=0
            )
        self.assertEqual(out.shape, (0, 4))
        moe.gate.assert_not_called()

    @staticmethod
    def _mega_gate_reference(x, weight, top_k, **kwargs):
        """Numerical substitute at the external DeepGEMM boundary only."""
        if not torch.isfinite(x).all():
            raise ValueError("The pinned MegaGate rejects nonfinite GEMM inputs")
        scores = torch.log1p(torch.exp(x.float() @ weight.float().T)).sqrt()
        if kwargs["unmapped_topk_idx"] is not None:
            ids = kwargs["unmapped_topk_idx"]
        else:
            bias = kwargs["bias"]
            if kwargs["image_bias"] is not None:
                bias = torch.where(
                    kwargs["image_token_mask"][:, None], kwargs["image_bias"], bias
                )
            ids = (scores + bias).topk(top_k, dim=-1).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * kwargs["routed_scaling_factor"]
        if kwargs["mask"] is not None:
            weights = weights.masked_fill(~kwargs["mask"][:, None], 0)
            ids = ids.masked_fill(~kwargs["mask"][:, None], -1)
        kwargs["out"][0].copy_(ids)
        kwargs["out"][1].copy_(weights)

    @classmethod
    def _mega_gate_moe(cls, fused_shared, scale_in_topk, hash_routing=False):
        moe = cls._v41_moe(fused_shared, scale_in_topk)
        moe.config.model_type = "deepseek_v41"
        moe.config.hidden_size = 256
        moe.gate.weight = torch.zeros((8, 256), dtype=torch.bfloat16)
        moe.gate.e_score_correction_bias = torch.tensor(
            [4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        moe.gate.e_score_correction_bias_vl = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]
        )
        moe.topk.topk_config.scoring_func = "sqrtsoftplus"
        if hash_routing:
            moe.is_hash = True
            moe.gate.e_score_correction_bias = None
            moe.topk = SimpleNamespace(
                topk=2 + fused_shared,
                num_fused_shared_experts=fused_shared,
                routed_scaling_factor=2.0,
                apply_routed_scaling_factor_on_output=scale_in_topk,
                score_func="sqrtsoftplus",
                tid2eid=torch.tensor([[6, 7]] * 100, dtype=torch.int32),
            )
        return moe

    def test_mega_gate_preserves_vision_hash_shared_scaling_and_padding(self):
        """A fused gate must preserve expert contributions across EP layouts.

        Nonfinite padding and out-of-vocabulary padded IDs must reach the
        expert boundary with zero weight; routed scaling must apply once and
        the home-rank shared expert must contribute exactly one identity.
        """
        deep_gemm = self.deep_gemm
        deep_gemm.bf16_mega_gate = self._mega_gate_reference
        for is_hash in (False, True):
            for fused_shared in (0, 1):
                for scale_in_topk in (False, True):
                    for valid_rows in (0, 2, 32):
                        with self.subTest(
                            is_hash=is_hash,
                            fused_shared=fused_shared,
                            scale_in_topk=scale_in_topk,
                            valid_rows=valid_rows,
                        ):
                            moe = self._mega_gate_moe(
                                fused_shared, scale_in_topk, is_hash
                            )
                            hidden = torch.ones((32, 256), dtype=torch.bfloat16)
                            hidden[valid_rows:] = float("nan")
                            input_ids = torch.tensor([7, 99] * 16)
                            input_ids[valid_rows:] = 1000
                            batch = SimpleNamespace(
                                moe_num_token_non_padded=lambda: torch.tensor(
                                    valid_rows
                                )
                            )

                            def run_experts(_experts, x, ids, weights, **kwargs):
                                self.assertTrue((ids[valid_rows:] == -1).all())
                                self.assertTrue((weights[valid_rows:] == 0).all())
                                factors = (ids + 1).float()
                                factors[ids == 9] = 1.0  # rank-1 shared slot
                                return (factors * weights).sum(
                                    -1, keepdim=True
                                ) * kwargs["routed_scaling_factor"]

                            with (
                                get_context().override_server_args(
                                    model_path="dummy",
                                    tp_size=2,
                                    ep_size=2,
                                    enable_deterministic_inference=False,
                                ),
                                envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
                                patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                                patch.object(mega_moe, "_device_sm", 100),
                                get_flags().moe.override(
                                    a2a_backend=MoeA2ABackend.MEGAMOE
                                ),
                                get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
                                patch.object(
                                    mega_moe.ExpertLocationDispatchInfo,
                                    "init_new",
                                    return_value=None,
                                ),
                                patch.object(
                                    mega_moe, "run_mega_routed_experts", run_experts
                                ),
                                patch.object(topk_module, "_is_cuda", True),
                                patch.object(
                                    topk_module,
                                    "_can_fuse_padded_region",
                                    return_value=False,
                                ),
                            ):
                                out = mega_moe._run_mega_routed(
                                    moe, hidden, batch, input_ids, num_tokens=32
                                )
                            image_value = 15.0 + 3 * fused_shared
                            text_value = image_value if is_hash else 3.0 + fused_shared
                            expected = torch.tensor([text_value, image_value] * 16)
                            expected[valid_rows:] = 0.0
                            torch.testing.assert_close(out[:, 0], expected)
                            self.assertTrue(torch.isnan(hidden[valid_rows:]).all())

    def test_mega_gate_admission_preserves_unsupported_router_fallbacks(self):
        """Do not silently change routing when the fused API cannot express it."""
        moe = self._mega_gate_moe(0, False)
        hidden = torch.zeros((32, 256), dtype=torch.bfloat16)
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.bf16_mega_gate = self._mega_gate_reference
        with (
            get_context().override_server_args(
                model_path="dummy", enable_deterministic_inference=False
            ),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True),
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
        ):
            self.assertTrue(
                mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
            )
            for tokens in (0, 1, 16):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(
                        moe, hidden[:tokens], 100, None
                    )
                )
            self.assertTrue(
                mega_gate_runtime.should_use_mega_gate(moe, hidden[:17], 103, None)
            )
            for sm in (90, 120):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(moe, hidden, sm, None)
                )
            for name, value in (
                ("renormalize", False),
                ("use_grouped_topk", True),
                ("scoring_func", "sigmoid"),
                ("custom_routing_function", object()),
                ("torch_native", True),
            ):
                with patch.object(moe.topk.topk_config, name, value):
                    self.assertFalse(
                        mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
                    )
            for algorithm in ("lp", "fake"):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(
                        moe,
                        hidden,
                        100,
                        SimpleNamespace(ep_dispatch_algorithm=algorithm),
                    )
                )
            with patch.object(moe.topk, "enable_waterfill", True, create=True):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
                )
            with envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(True):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
                )
            with get_exec().deterministic.override(enable_deterministic_inference=True):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
                )
            with envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(False):
                self.assertFalse(
                    mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
                )
            del deep_gemm.bf16_mega_gate
            self.assertFalse(
                mega_gate_runtime.should_use_mega_gate(moe, hidden, 100, None)
            )

    def test_mega_gate_maps_eplb_before_shared_slots_and_records_routed_ids(self):
        """Redundant-expert maps index routed IDs before per-rank slot insertion.

        Shared IDs must not index the EPLB map or enter expert statistics, and
        padding must not be counted as the map's last physical expert.
        """
        moe = self._mega_gate_moe(1, True)
        hidden = torch.ones((32, 256), dtype=torch.bfloat16)
        hidden[2:] = float("nan")
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.bf16_mega_gate = self._mega_gate_reference
        # Reverse routed placement. Physical expert count includes redundancy.
        dispatch_info = mega_moe.ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=torch.tensor(
                [9, 8, 7, 6, 5, 4, 3, 2], dtype=torch.int32
            ),
            partial_logical_to_all_physical_map=None,
            partial_logical_to_all_physical_map_num_valid=None,
            num_physical_experts=10,
        )
        recorded = []
        recorder = SimpleNamespace(
            on_select_experts=lambda **kw: recorded.append(kw["topk_ids"].clone())
        )
        with (
            get_context().override_server_args(
                model_path="dummy", tp_size=2, ep_size=2
            ),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
            get_parallel().override(moe_ep_size=2, moe_ep_rank=1),
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "_can_fuse_padded_region", return_value=False),
            patch.object(
                mega_gate_runtime,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            weights, ids = mega_gate_runtime.run_mega_gate(
                moe,
                hidden,
                torch.tensor([7, 99] + [1000] * 30),
                torch.tensor(2),
                dispatch_info,
            )
        # Rank 1's redundant-routed layout has shared ID 11. Logical 0/1
        # maps to physical 9/8, then shared-slot insertion shifts to 10/9.
        torch.testing.assert_close(ids[:2], torch.tensor([[10, 9, 11], [2, 3, 11]]))
        torch.testing.assert_close(weights[:2], torch.ones((2, 3)))
        self.assertTrue((ids[2:] == -1).all())
        self.assertTrue((weights[2:] == 0).all())
        self.assertEqual(len(recorded), 1)
        torch.testing.assert_close(recorded[0][:2], torch.tensor([[9, 8], [2, 3]]))
        self.assertTrue((recorded[0][2:] == -1).all())

    @staticmethod
    def _v41_moe(fused_shared, scale_in_topk):
        return SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                image_token_id=99,
            ),
            experts=SimpleNamespace(
                should_fuse_routed_scaling_factor_in_topk=scale_in_topk,
                moe_runner_config=SimpleNamespace(swiglu_limit=None),
            ),
            gate=MagicMock(
                return_value=torch.zeros((3, 4)),
                e_score_correction_bias=torch.tensor([4.0, 3.0, 0.0, 0.0]),
                e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 3.0, 4.0]),
            ),
            topk=SimpleNamespace(
                topk_config=TopKConfig(
                    top_k=2 + fused_shared,
                    num_fused_shared_experts=fused_shared,
                    renormalize=True,
                    routed_scaling_factor=2.0,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                )
            ),
            is_hash=False,
            num_fused_shared_experts=fused_shared,
            layer_id=0,
            routed_scaling_factor=2.0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
        )

    def test_shape_check_uses_pinned_fp8_fp4_alignment(self):
        # Packed SF groups need 128 elements, not a 16-byte token SF row.
        mega_moe.check_mega_moe_shapes(2048, 768, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(5120, 2304, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(512, 128, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 128"):
            mega_moe.check_mega_moe_shapes(2048, 736, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(2016, 768, "fp8xfp4")
        with self.assertRaisesRegex(ValueError, "hidden_size.*multiple of 512"):
            mega_moe.check_mega_moe_shapes(256, 128, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1536, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1024, "mxf4xmxf4")
        # 768 is a multiple of 256, so the NVFP4 (g16) rule accepts it.
        mega_moe.check_mega_moe_shapes(2048, 768, "nvfp4xnvfp4")
        with self.assertRaisesRegex(ValueError, "intermediate_size.*multiple of 256"):
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
