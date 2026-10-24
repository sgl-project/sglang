"""Routing follows the owning layer and active batch across target/draft and TBO."""

import inspect
import sys
import unittest
from enum import IntEnum
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.operations import (
    ExecutionOperation,
    _StageExecutor,
)
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe import hash_topk as hash_topk_module
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import (
    StandardTopKOutputPacked,
    TopK,
    TopKOutputFormat,
    fused_topk_torch_native,
)
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    draft_model_build_scope,
    initialize_moe_config,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
)
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_model,
    get_parallel,
    get_resources,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _batch(real):
    empty = torch.empty(0, dtype=torch.int32)
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        input_ids=torch.arange(4),
        req_pool_indices=empty,
        seq_lens=empty,
        out_cache_loc=empty,
        seq_lens_sum=0,
    )
    batch.num_token_non_padded = torch.tensor(real, dtype=torch.int32)
    batch.global_num_token_non_padded = batch.num_token_non_padded
    batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    batch.global_dp_buffer_len = batch.tbo_padded_len = 4
    return batch


def _placement(num_layers=1):
    logical_to_physical = torch.tensor([[2, 0, 3, 1]], dtype=torch.int64).repeat(
        num_layers, 1
    )
    physical_to_logical = logical_to_physical.argsort(dim=-1)
    all_physical = torch.full((num_layers, 4, 4), -1, dtype=torch.int64)
    all_physical[:, :, 0] = logical_to_physical
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical,
        physical_to_logical_map_cpu=physical_to_logical.clone(),
        logical_to_all_physical_map=all_physical,
        logical_to_all_physical_map_cpu=all_physical.clone(),
        logical_to_all_physical_map_num_valid=torch.ones(
            (num_layers, 4), dtype=torch.int64
        ),
        ep_size=1,
        logical_to_rank_dispatch_physical_map=logical_to_physical,
    )


class TestTopKRoutingInputs(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = get_context().override_server_args(
            ep_dispatch_algorithm="static", enable_eplb=True
        )
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        topology = get_parallel().override(
            attn_dp_size=1, attn_cp_size=1, attn_tp_size=1, attn_tp_rank=0
        )
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        for name in ("_is_cuda", "_is_hip", "_use_aiter"):
            scope = patch.object(topk_module, name, False)
            scope.start()
            self.addCleanup(scope.stop)
        self.metadata = _placement()
        get_resources().expert_location_metadata = self.metadata
        self.addCleanup(setattr, get_resources(), "expert_location_metadata", None)
        self.hidden = torch.ones((4, 2))
        self.logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]]).repeat(4, 1)

    def _router(self, **kwargs):
        shared_slots = kwargs.pop("num_fused_shared_experts", 0)
        layer_id = kwargs.pop("layer_id", 0)
        with get_model().override(
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=shared_slots)
        ):
            return TopK(
                top_k=2 - shared_slots,
                layer_id=layer_id,
                custom_routing_function=fused_topk_torch_native,
                **kwargs,
            )

    def test_target_and_draft_remap_only_the_target_and_keep_mask(self):
        """Draft routes must not use the target placement or lose every live row."""
        target = self._router()
        with draft_model_build_scope():
            draft = self._router()
        batch = _batch(2)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=batch
        ):
            target_ids = target.forward_native(self.hidden, self.logits).topk_ids
            draft_ids = draft.forward_native(self.hidden, self.logits).topk_ids
        self.assertEqual(target_ids.tolist(), [[2, 0], [2, 0], [-1, -1], [-1, -1]])
        self.assertEqual(draft_ids.tolist(), [[0, 1], [0, 1], [-1, -1], [-1, -1]])
        self.assertFalse(draft.topk_config.allow_routed_experts_capture)
        self.assertTrue(target.topk_config.allow_routed_experts_capture)

    def test_late_preparation_and_in_place_rebalance(self):
        """Forward preparation and rebalance must be observed after scope entry."""
        router = self._router()
        batch = _batch(1)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=batch
        ):
            before = router.forward_native(self.hidden, self.logits).topk_ids
            batch.num_token_non_padded = torch.tensor(3)
            self.metadata.logical_to_rank_dispatch_physical_map.copy_(
                torch.tensor([[1, 3, 0, 2]])
            )
            after = router.forward_native(self.hidden, self.logits).topk_ids
        self.assertEqual(before[0].tolist(), [2, 0])
        self.assertEqual(before[1].tolist(), [-1, -1])
        self.assertEqual(after.tolist(), [[1, 3], [1, 3], [1, 3], [-1, -1]])

    def test_nested_scope_restores_and_retains_replay_buffer_identity(self):
        router = self._router()
        parent, child = _batch(3), _batch(1)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=parent
        ):
            original = get_forward_context()
            bound, _ = router.routing_inputs()
            self.assertIs(bound, parent.num_token_non_padded)
            with self.assertRaisesRegex(RuntimeError, "child failed"):
                with forward_context(
                    ForwardContext(attn_backend=object()), forward_batch=child
                ):
                    self.assertIs(
                        router.routing_inputs()[0], child.num_token_non_padded
                    )
                    raise RuntimeError("child failed")
            self.assertIs(get_forward_context(), original)
            parent.num_token_non_padded.fill_(2)
            self.assertIs(router.routing_inputs()[0], bound)
            self.assertEqual(
                router.forward_native(self.hidden, self.logits).topk_ids[-2:].tolist(),
                [[-1, -1], [-1, -1]],
            )
        self.assertFalse(has_forward_context())

    def test_tbo_children_bind_bounds_without_tbo_attention_backend(self):
        router = self._router()
        seen = []

        def read_bound(state, forward_batch):
            seen.append(int(router.routing_inputs()[0]))
            return {"forward_batch": forward_batch}

        stages = [[ExecutionOperation("route", read_bound)]] * 2
        parent, child_a, child_b = _batch(4), _batch(1), _batch(3)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=parent
        ):
            a = _StageExecutor("a", stages, {"forward_batch": child_a})
            b = _StageExecutor("b", stages, {"forward_batch": child_b})
            a.next()
            b.next()
            a.next()
            b.next()
            self.assertEqual(int(router.routing_inputs()[0]), 4)
        self.assertEqual(seen, [1, 3, 1, 3])

    def test_hash_router_uses_same_placement_and_bound(self):
        router = HashTopK(2, 4, vocab_size=4, scoring_func="softmax", layer_id=0)
        with envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = router(self.hidden, self.logits, torch.arange(4))
        self.assertEqual(output.topk_ids.tolist(), [[2, 0], [0, 3], [-1, -1], [-1, -1]])

    def test_vision_router_maps_routed_columns_and_masks_shared_rows_last(self):
        router = self._router(
            num_fused_shared_experts=1,
            routed_scaling_factor=2.0,
            fused_shared_experts_scaling_factor=0.25,
        )
        moe = SimpleNamespace(
            topk=router,
            layer_id=0,
            config=SimpleNamespace(image_token_id=99),
            gate=SimpleNamespace(
                e_score_correction_bias=torch.zeros(4),
                e_score_correction_bias_vl=torch.zeros(4),
            ),
        )
        with patch(
            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False
        ):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = vision_topk(moe, self.logits, torch.tensor([1, 99, 1, 99]))
        self.assertEqual(output.topk_ids.tolist(), [[2, 4], [2, 4], [-1, -1], [-1, -1]])
        torch.testing.assert_close(output.topk_weights[:2, -1], torch.full((2,), 0.125))
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))

    def test_vision_packed_output_follows_built_runner(self):
        """A requested MXFP4 fallback must not emit an unused packed-ID buffer."""
        with get_context().override_server_args(moe_runner_backend="flashinfer_mxfp4"):
            initialize_moe_config()
            for backend in (
                None,
                MoeRunnerBackend.TRITON,
                MoeRunnerBackend.FLASHINFER_MXFP4,
            ):
                with self.subTest(backend=backend):
                    router = self._router(is_draft=True)
                    router.bind_runner_backend(backend)
                    moe = SimpleNamespace(
                        topk=router,
                        layer_id=0,
                        config=SimpleNamespace(image_token_id=99),
                        gate=SimpleNamespace(
                            e_score_correction_bias=torch.zeros(4),
                            e_score_correction_bias_vl=torch.zeros(4),
                        ),
                    )
                    with (
                        patch(
                            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                            return_value=True,
                        ),
                        patch(
                            "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate",
                            return_value=(
                                torch.full((4, 2), 0.5),
                                torch.tensor([[0, 1]], dtype=torch.int32).repeat(4, 1),
                            ),
                        ) as gate,
                    ):
                        output = vision_topk(moe, self.logits, input_ids=None)
                    packed = gate.call_args.kwargs["packed_out"]
                    expected_packed = backend == MoeRunnerBackend.FLASHINFER_MXFP4
                    self.assertEqual(packed is not None, expected_packed)
                    self.assertEqual(
                        isinstance(output, StandardTopKOutputPacked), expected_packed
                    )

    def test_vision_fake_algorithm_neutralizes_both_biases(self):
        with get_exec().moe.override(ep_dispatch_algorithm="fake"):
            router = self._router()
            moe = SimpleNamespace(
                topk=router,
                layer_id=0,
                config=SimpleNamespace(image_token_id=99),
                gate=SimpleNamespace(
                    e_score_correction_bias=torch.tensor([0.0, 0.0, 1000.0, 0.0]),
                    e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 0.0, 1000.0]),
                ),
            )
            with patch(
                "sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False
            ):
                with forward_context(
                    ForwardContext(attn_backend=object()), forward_batch=_batch(4)
                ):
                    logits = self.logits.clone()
                    output = vision_topk(moe, logits, torch.tensor([1, 99, 1, 99]))
            logical = logits.topk(2, dim=-1).indices
            expected = self.metadata.logical_to_all_physical_map[0, logical, 0]
            self.assertTrue(torch.equal(output.topk_ids.long(), expected))

    def test_draft_without_input_ids_retains_prepared_live_width(self):
        """Deferred draft IDs must not turn a live draft's padding bound into zero."""
        batch = _batch(0)
        batch.input_ids = None
        batch.spec_info = EagleDraftInput(num_tokens_per_req=2)
        with get_parallel().override(tp_size=4, attn_tp_size=4, moe_ep_size=4):
            batch._init_num_token_non_padded(device="cpu", is_draft_worker=True)
        self.assertEqual(batch.global_num_token_non_padded_cpu, 8)
        self.assertEqual(int(batch.global_num_token_non_padded), 8)
        batch.forward_mode = ForwardMode.IDLE
        with get_parallel().override(tp_size=4, attn_tp_size=4, moe_ep_size=4):
            batch._init_num_token_non_padded(device="cpu", is_draft_worker=True)
        self.assertEqual(int(batch.global_num_token_non_padded), 0)

    def test_npu_routes_keep_valid_ids_and_zero_only_padded_contributions(self):
        """NPU padding must stay inert without inventing invalid or duplicate routes."""
        router = self._router(num_fused_shared_experts=1)
        router.topk_config.scoring_func = "sqrtsoftplus"
        logical = torch.tensor([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]]).repeat(4, 1)

        class Capture:
            def capture(self, *, layer_id, topk_indices):
                self.ids = topk_indices.clone()

        capture = Capture()
        get_resources().experts_capturer = capture
        self.addCleanup(setattr, get_resources(), "experts_capturer", None)
        _, info = router.routing_inputs()
        with patch.object(
            torch.ops.custom,
            "npu_moe_gating_top_k",
            create=True,
            return_value=(weights, logical, None),
        ):
            output = fused_topk_npu(
                hidden_states=self.hidden,
                router_logits=self.logits,
                topk_config=router.topk_config,
                num_token_non_padded=torch.tensor(2),
                expert_location_dispatch_info=info,
                layer_id=0,
            )
        self.assertEqual(output.topk_ids.tolist(), [[2, 4], [0, 4], [3, 4], [1, 4]])
        torch.testing.assert_close(
            output.topk_weights[:2], torch.tensor([[0.75, 0.25]]).repeat(2, 1)
        )
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))
        self.assertEqual(capture.ids.tolist(), [[0], [1], [2], [3]])

    def test_in_kernel_formats_do_not_silently_ignore_expert_placement(self):
        router = self._router(output_format=TopKOutputFormat.TRITON_KERNEL)
        with self.assertRaisesRegex(
            NotImplementedError, "placement remapping requires"
        ):
            router.forward_cuda(self.hidden, self.logits)

        router = self._router(output_format=TopKOutputFormat.BYPASSED)
        output = router.forward_cuda(self.hidden, self.logits)
        with self.assertRaisesRegex(
            NotImplementedError, "placement remapping requires"
        ):
            _ = output.router_logits

    def test_bypassed_materialization_preserves_placement_and_padding(self):
        """A dispatcher can remap lazy routes before sending physical expert IDs."""
        router = self._router(output_format=TopKOutputFormat.BYPASSED)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=_batch(2)
        ):
            output = router.forward_cuda(self.hidden, self.logits)
            materialized = output.to_standard(layer_id=0)
        self.assertEqual(
            materialized.topk_ids.tolist(), [[2, 0], [2, 0], [-1, -1], [-1, -1]]
        )
        self.assertIs(materialized.router_logits, self.logits)
        layer = SimpleNamespace(
            _use_ascend_fuseep=False,
            forward_impl=lambda hidden, routes, **kwargs: routes.to_standard(
                layer_id=0
            ),
        )
        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.is_in_tc_piecewise_cuda_graph",
            return_value=True,
        ):
            piecewise = FusedMoE.forward(layer, self.hidden, output)
        torch.testing.assert_close(piecewise.topk_ids, materialized.topk_ids)

    def test_piecewise_lazy_routing_keeps_a_bound_without_placement(self):
        """A logits-only opaque op must not drop a row bound on graph entry."""
        router = self._router(output_format=TopKOutputFormat.BYPASSED)
        with get_exec().moe.override(ep_dispatch_algorithm=None):
            output = router.forward_cuda(
                self.hidden, self.logits, num_token_non_padded=torch.tensor(2)
            )
        self.assertIsNone(output.expert_location_dispatch_info)
        layer = SimpleNamespace(
            _use_ascend_fuseep=False,
            forward_impl=lambda hidden, routes, **kwargs: routes.to_standard(),
        )
        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.is_in_tc_piecewise_cuda_graph",
            return_value=True,
        ):
            materialized = FusedMoE.forward(layer, self.hidden, output)
        self.assertEqual(
            materialized.topk_ids.tolist(), [[0, 1], [0, 1], [-1, -1], [-1, -1]]
        )

    def test_fp8_consumers_keep_lazy_padding_and_placement(self):
        """Both FP8 formats consume physical IDs and ignore poisoned inactive scores."""
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.utils import RoutingMethodType

        provider = ModuleType("flashinfer.fused_moe")
        provider.Fp8QuantizationType = IntEnum(
            "Fp8QuantizationType", {"DeepSeekFp8": 0, "MxFp8": 1}
        )

        def score_router(**kwargs):
            weights, ids = fused_topk_torch_native(**kwargs)
            return weights.to(score_dtype), ids

        def routed_kernel(*, topk_ids, output, routing_method_type, **kwargs):
            ids, weights = topk_ids
            self.assertEqual(routing_method_type, RoutingMethodType.TopK)
            self.assertEqual(kwargs["top_k"], ids.shape[1])
            self.assertEqual(
                weights.dtype,
                torch.float32 if score_dtype == torch.float16 else score_dtype,
            )
            self.assertTrue(ids.is_contiguous() and weights.is_contiguous())
            self.assertTrue(bool((ids[real:] == -1).all()))
            self.assertEqual(ids[:real].tolist(), [expected_ids] * real)
            values = (ids.float() + 1) * torch.where(ids >= 0, weights, 0)
            output.copy_(values.sum(-1, keepdim=True).expand_as(output))

        provider.trtllm_fp8_block_scale_routed_moe = routed_kernel
        provider.trtllm_fp8_per_tensor_scale_routed_moe = routed_kernel

        def logits_kernel(*, routing_logits, output=None, **kwargs):
            weights, ids = routing_logits.float().softmax(-1).topk(2)
            weights = weights / weights.sum(-1, keepdim=True) * 2.5
            values = ((ids.float() + 1) * weights).sum(-1, keepdim=True)
            result = values.expand_as(self.hidden)
            if output is None:
                return result.to(torch.bfloat16)
            output.copy_(result)
            return output

        provider.trtllm_fp8_block_scale_moe = logits_kernel
        provider.trtllm_fp8_per_tensor_scale_moe = logits_kernel
        for block_quant in (True, False):
            quant_info = flashinfer_trtllm.FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=torch.empty((4, 4, 2)),
                w2_weight=torch.empty((4, 2, 2)),
                global_num_experts=4,
                local_expert_offset=0,
                local_num_experts=4,
                intermediate_size=2,
                routing_method_type=RoutingMethodType.RenormalizeNaive,
                block_quant=block_quant,
                weight_block_k=128,
                w13_weight_scale_inv=torch.ones((4, 1, 1)),
                w2_weight_scale_inv=torch.ones((4, 1, 1)),
                w13_input_scale=torch.ones(1),
                output1_scales_scalar=torch.ones(4),
                output1_scales_gate_scalar=torch.ones(4),
                output2_scales_scalar=torch.ones(4),
            )
            for score_dtype in (torch.float16, torch.bfloat16, torch.float32):
                for placement in (False, True):
                    expected_ids = [2, 0] if placement else [0, 1]
                    for real in (3, 1, 0, 4):
                        with self.subTest(
                            block_quant=block_quant,
                            score_dtype=score_dtype,
                            placement=placement,
                            real=real,
                        ):
                            hidden = self.hidden.clone()
                            hidden[real:] = float("nan")
                            logits = self.logits.clone()
                            logits[real:] = float("nan")
                            router = self._router(
                                output_format=TopKOutputFormat.BYPASSED,
                                routed_scaling_factor=2.5,
                                apply_routed_scaling_factor_on_output=True,
                            )
                            router.topk_config.custom_routing_function = score_router
                            with get_exec().moe.override(
                                ep_dispatch_algorithm="static" if placement else None
                            ):
                                lazy = router.forward_cuda(
                                    hidden,
                                    logits,
                                    num_token_non_padded=torch.tensor(real),
                                )
                            with (
                                get_parallel().override(tp_group=None),
                                patch.dict(
                                    sys.modules, {"flashinfer.fused_moe": provider}
                                ),
                                patch.object(
                                    flashinfer_trtllm,
                                    "trtllm_fp8_block_scale_moe_out_wrapper",
                                    inspect.unwrap(
                                        flashinfer_trtllm.trtllm_fp8_block_scale_moe_out_wrapper
                                    ),
                                ),
                                patch.object(
                                    flashinfer_trtllm,
                                    "trtllm_fp8_per_tensor_scale_moe_wrapper",
                                    inspect.unwrap(
                                        flashinfer_trtllm.trtllm_fp8_per_tensor_scale_moe_wrapper
                                    ),
                                ),
                                patch.object(
                                    flashinfer_trtllm,
                                    "trtllm_fp8_block_scale_routed_moe_out_wrapper",
                                    inspect.unwrap(
                                        flashinfer_trtllm.trtllm_fp8_block_scale_routed_moe_out_wrapper
                                    ),
                                ),
                                patch.object(
                                    flashinfer_trtllm,
                                    "trtllm_fp8_per_tensor_scale_routed_moe_out_wrapper",
                                    inspect.unwrap(
                                        flashinfer_trtllm.trtllm_fp8_per_tensor_scale_routed_moe_out_wrapper
                                    ),
                                ),
                                patch.object(
                                    flashinfer_trtllm,
                                    "per_token_group_quant_fp8",
                                    return_value=(hidden, torch.ones((1, 1))),
                                ),
                                patch.object(
                                    flashinfer_trtllm,
                                    "scaled_fp8_quant",
                                    return_value=(hidden, torch.ones(1)),
                                ),
                            ):
                                result = flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp8(
                                    StandardDispatchOutput(hidden, None, lazy),
                                    quant_info,
                                    MoeRunnerConfig(
                                        activation="silu",
                                        routed_scaling_factor=2.5,
                                    ),
                                ).hidden_states
                            scores = self.logits[:, :2].softmax(-1).to(score_dtype)
                            scores *= 2.5
                            expected = (
                                (scores * (torch.tensor(expected_ids).float() + 1))
                                .sum(-1, keepdim=True)
                                .expand_as(result)
                                .clone()
                            )
                            expected[real:] = 0
                            torch.testing.assert_close(
                                result.float(), expected, rtol=0.005, atol=0.005
                            )
                            self.assertTrue(bool(torch.isfinite(result).all()))

    def test_fp8_route_scores_preserve_values_and_supported_storage(self):
        """FP16 external scores widen exactly; native BF16/FP32 scores keep storage."""
        from torch.utils._python_dispatch import TorchDispatchMode

        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        class ObserveScoreCopies(TorchDispatchMode):
            def __init__(self):
                self.copies = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in (
                    torch.ops.aten._to_copy.default,
                    torch.ops.aten.clone.default,
                ):
                    self.copies.append(func)
                return func(*args, **(kwargs or {}))

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    # Values expose FP16-to-BF16 double rounding and preserve the
                    # payload of dropped routes for the consumer to ignore.
                    storage = torch.tensor(
                        [
                            [0.333251953125, 9.0, 2.0**-20, 9.0],
                            [float("nan"), 9.0, float("inf"), 9.0],
                        ],
                        dtype=dtype,
                    )
                    weights = (
                        storage[:, ::2] if strided else storage[:, ::2].contiguous()
                    )
                    ids = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
                    original = weights.clone()
                    with ObserveScoreCopies() as observed:
                        routing = flashinfer_trtllm._get_routing_for_flashinfer_routed(
                            StandardTopKOutput(weights, ids, self.logits[:2]),
                            promote_half_scores=True,
                        )
                    if dtype == torch.float16:
                        self.assertEqual(
                            observed.copies, [torch.ops.aten._to_copy.default]
                        )
                    else:
                        self.assertEqual(len(observed.copies), int(strided))
                    actual_ids, actual_weights = routing
                    self.assertIs(actual_ids, ids)
                    expected_dtype = torch.float32 if dtype == torch.float16 else dtype
                    self.assertEqual(actual_weights.dtype, expected_dtype)
                    self.assertTrue(actual_weights.is_contiguous())
                    torch.testing.assert_close(
                        actual_weights.float(),
                        original.float(),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    torch.testing.assert_close(
                        weights, original, rtol=0, atol=0, equal_nan=True
                    )
                    if dtype != torch.float16 and not strided:
                        self.assertIs(actual_weights, weights)

    def test_fp8_deferred_finalize_keeps_materialized_routes(self):
        """Padding and placement retain the deferred GEMM2/score/permutation ABI."""
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.utils import RoutingMethodType

        provider = ModuleType("flashinfer.fused_moe")
        provider.Fp8QuantizationType = IntEnum(
            "Fp8QuantizationType", {"DeepSeekFp8": 0, "MxFp8": 1}
        )
        returned = []

        def deferred_result(ids, weights, kwargs):
            self.assertFalse(kwargs["do_finalize"])
            self.assertNotIn("output", kwargs)
            self.assertEqual(kwargs["top_k"], ids.shape[1])
            self.assertEqual(weights.dtype, torch.float32)
            indices = torch.arange(ids.numel(), dtype=torch.int32).reshape_as(ids)
            indices[ids < 0] = -1
            gemm2 = (ids.clamp_min(0).reshape(-1, 1) + 1).to(torch.bfloat16)
            gemm2 = gemm2.expand(-1, self.hidden.shape[1]).contiguous()
            returned[:] = [gemm2, weights, indices]
            return returned

        def routed_kernel(*, topk_ids, **kwargs):
            ids, weights = topk_ids
            expected_ids = [2, 0] if placement else [0, 1]
            self.assertEqual(ids[:real].tolist(), [expected_ids] * real)
            self.assertTrue(bool((ids[real:] == -1).all()))
            self.assertEqual(kwargs["routing_method_type"], RoutingMethodType.TopK)
            return deferred_result(ids, weights, kwargs)

        def logits_kernel(*, routing_logits, **kwargs):
            self.assertIs(routing_logits, logits)
            self.assertEqual(
                kwargs["routing_method_type"], RoutingMethodType.RenormalizeNaive
            )
            weights, ids = routing_logits.softmax(-1).topk(2)
            weights = weights / weights.sum(-1, keepdim=True) * 2.5
            return deferred_result(ids, weights, kwargs)

        provider.trtllm_fp8_block_scale_routed_moe = routed_kernel
        provider.trtllm_fp8_block_scale_moe = logits_kernel
        quant_info = flashinfer_trtllm.FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.empty((4, 4, 2)),
            w2_weight=torch.empty((4, 2, 2)),
            global_num_experts=4,
            local_expert_offset=0,
            local_num_experts=4,
            intermediate_size=2,
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            block_quant=True,
            weight_block_k=128,
            w13_weight_scale_inv=torch.ones((4, 1, 1)),
            w2_weight_scale_inv=torch.ones((4, 1, 1)),
        )
        for bound, placement in ((None, False), (2, False), (2, True), (0, True)):
            real = 4 if bound is None else bound
            with self.subTest(bound=bound, placement=placement):
                hidden, logits = self.hidden.clone(), self.logits.clone()
                hidden[real:] = logits[real:] = float("nan")
                router = self._router(
                    output_format=TopKOutputFormat.BYPASSED,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                )
                with get_exec().moe.override(
                    ep_dispatch_algorithm="static" if placement else None
                ):
                    lazy = router.forward_cuda(
                        hidden,
                        logits,
                        num_token_non_padded=(
                            None if bound is None else torch.tensor(bound)
                        ),
                    )
                with (
                    patch.dict(sys.modules, {"flashinfer.fused_moe": provider}),
                    patch.object(
                        flashinfer_trtllm,
                        "per_token_group_quant_fp8",
                        return_value=(hidden, torch.ones((1, 1))),
                    ),
                    patch.object(
                        flashinfer_trtllm, "use_symmetric_memory"
                    ) as allocation,
                    flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(),
                ):
                    result = (
                        flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp8(
                            StandardDispatchOutput(hidden, None, lazy),
                            quant_info,
                            MoeRunnerConfig(
                                activation="silu",
                                routed_scaling_factor=2.5,
                            ),
                        ).hidden_states
                    )
                    allocation.assert_not_called()
                self.assertIsInstance(
                    result, flashinfer_trtllm.FlashInferTrtllmDeferredFinalizeOutput
                )
                self.assertIs(result.gemm2_out, returned[0])
                self.assertIs(result.expert_weights, returned[1])
                self.assertIs(result.expanded_idx_to_permuted_idx, returned[2])
                self.assertEqual(result.top_k, 2)
                self.assertTrue(bool((returned[2][real:] == -1).all()))
                self.assertEqual(result.expert_weights.dtype, torch.float32)
                self.assertTrue(
                    bool(torch.isfinite(result.expert_weights[:real]).all())
                )
                # Dropped permutation entries exclude these still-poisoned scores.
                self.assertTrue(bool(torch.isnan(result.expert_weights[real:]).all()))
                torch.testing.assert_close(
                    result.expert_weights[:real].sum(-1), torch.full((real,), 2.5)
                )

    def test_in_range_draft_loads_experts_in_its_routing_order(self):
        """Draft expert weights stay in logical order despite a permuted target map."""
        outputs = []
        for is_draft in (False, True):
            method = UnquantizedFusedMoEMethod()
            scope = draft_model_build_scope() if is_draft else get_parallel().override()
            with (
                scope,
                patch.object(method, "create_weights"),
                get_parallel().override(
                    tp_size=1,
                    moe_ep_size=1,
                    moe_ep_rank=0,
                    moe_tp_size=1,
                    moe_tp_rank=0,
                ),
            ):
                layer = FusedMoE(
                    num_experts=4,
                    hidden_size=2,
                    intermediate_size=2,
                    layer_id=0,
                    top_k=2,
                    quant_method=method,
                )
            weights = torch.nn.Parameter(torch.zeros((4, 2, 2)), requires_grad=False)
            for expert_id in range(4):
                layer.weight_loader(
                    param=weights,
                    loaded_weight=torch.eye(2) * (expert_id + 1),
                    weight_name="experts.w2_weight",
                    shard_id="w2",
                    expert_id=expert_id,
                )
            self.assertEqual(
                weights[:, 0, 0].tolist(), [1, 2, 3, 4] if is_draft else [2, 4, 1, 3]
            )
            router = self._router(is_draft=is_draft)
            topk = router.forward_native(self.hidden, self.logits)
            expert_outputs = torch.einsum(
                "tkij,tj->tki", weights[topk.topk_ids], self.hidden
            )
            outputs.append((expert_outputs * topk.topk_weights[..., None]).sum(dim=1))
        torch.testing.assert_close(outputs[0], outputs[1])
        expected_scale = torch.softmax(self.logits[:, :2], dim=-1) @ torch.tensor(
            [1.0, 2.0]
        )
        torch.testing.assert_close(outputs[1], self.hidden * expected_scale[:, None])

    def test_identity_placement_keeps_in_kernel_routing_available(self):
        identity = torch.arange(4).view(1, 4)
        self.metadata.logical_to_rank_dispatch_physical_map.copy_(identity)
        self.metadata.logical_to_all_physical_map[:, :, 0].copy_(identity)
        self.metadata.physical_to_logical_map.copy_(identity)
        with get_exec().moe.override(
            enable_eplb=False,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            ep_dispatch_algorithm="dynamic",
        ):
            for output_format in (
                TopKOutputFormat.BYPASSED,
                TopKOutputFormat.TRITON_KERNEL,
            ):
                with self.subTest(output_format=output_format):
                    router = self._router(output_format=output_format)
                    with patch.object(
                        topk_module, "routing", return_value=(None, None, None, None, 2)
                    ):
                        output = router.forward_cuda(self.hidden, self.logits)
                    self.assertEqual(output.format, output_format)
                    if output_format == TopKOutputFormat.BYPASSED:
                        self.assertIs(output.router_logits, self.logits)
            for algorithm in ("fake", "lp"):
                with (
                    self.subTest(algorithm=algorithm),
                    get_exec().moe.override(ep_dispatch_algorithm=algorithm),
                ):
                    router = self._router(output_format=TopKOutputFormat.BYPASSED)
                    output = router.forward_cuda(self.hidden, self.logits)
                    with self.assertRaisesRegex(
                        NotImplementedError, "placement remapping requires"
                    ):
                        _ = output.router_logits

    def test_hash_npu_keeps_valid_ids_and_zero_padded_weights(self):
        router = HashTopK(2, 4, vocab_size=4, scoring_func="softmax", layer_id=0)
        with (
            patch.object(hash_topk_module, "_is_npu", True),
            envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False),
        ):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = router(self.hidden, self.logits, torch.arange(4))
        self.assertEqual(output.topk_ids.tolist(), [[2, 0], [0, 3], [3, 1], [1, 2]])
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))

    def test_hash_global_shared_column_stays_outside_expert_map(self):
        with get_model().override(
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1)
        ):
            router = HashTopK(
                2,
                4,
                vocab_size=4,
                scoring_func="sqrtsoftplus",
                routed_scaling_factor=2.0,
                layer_id=0,
            )
        with envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = router(self.hidden, self.logits, torch.arange(4))
        self.assertEqual(
            output.topk_ids.tolist(), [[2, 0, 4], [0, 3, 4], [-1, -1, -1], [-1, -1, -1]]
        )

    def test_deepseek_opaque_bypass_requires_identity_placement_and_native_policy(self):
        from sglang.srt.models import deepseek_v2

        moe = SimpleNamespace(
            alt_stream=object(),
            num_fused_shared_experts=0,
            shared_experts=object(),
            experts=SimpleNamespace(use_flashinfer_trtllm_moe=True),
            _enable_a2a_moe=False,
            _fuse_shared_experts_inside_sbo=False,
            is_hash=False,
            topk=self._router(output_format=TopKOutputFormat.BYPASSED),
        )
        with (
            patch.object(deepseek_v2, "_enable_pcg_dsv2_dual_stream", True),
            patch.object(
                deepseek_v2, "is_in_tc_piecewise_cuda_graph", return_value=True
            ),
            patch.object(
                deepseek_v2,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
            ),
            get_exec().moe.override(
                enable_eplb=False,
                init_expert_location="trivial",
                ep_num_redundant_experts=0,
                ep_dispatch_algorithm=None,
            ),
        ):
            self.assertTrue(
                deepseek_v2.DeepseekV2MoE._can_dual_stream_graph(moe, self.hidden)
            )
            with get_exec().moe.override(init_expert_location="reverse"):
                self.assertFalse(
                    deepseek_v2.DeepseekV2MoE._can_dual_stream_graph(moe, self.hidden)
                )
            moe.topk.topk_config.output_format = TopKOutputFormat.STANDARD
            self.assertFalse(
                deepseek_v2.DeepseekV2MoE._can_dual_stream_graph(moe, self.hidden)
            )

    def test_lp_keeps_global_shared_columns_outside_routed_domain(self):
        router = self._router(num_fused_shared_experts=1)
        ids = torch.tensor([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=torch.int32)
        weights = torch.tensor([[0.8, 0.2]]).repeat(4, 1)
        solved = []

        def solve(routed_ids):
            self.assertEqual(routed_ids.tolist(), [[0], [1], [2], [3]])
            solved.append(routed_ids.clone())
            return torch.ones(4, 1)

        def dispatch(routed_ids, info, probability=None):
            self.assertTrue(bool(((routed_ids >= 0) & (routed_ids < 4)).all()))
            return info.partial_logical_to_rank_dispatch_physical_map[routed_ids]

        get_resources().lplb_solvers[0] = SimpleNamespace(solve=solve)
        self.addCleanup(get_resources().lplb_solvers.pop, 0)
        with (
            get_exec().moe.override(ep_dispatch_algorithm="lp"),
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "topk_ids_logical_to_physical", dispatch),
        ):
            _, info = router.routing_inputs()
            output_ids, output_weights, _ = topk_module._post_process_topk_ids(
                ids,
                weights,
                topk_config=router.topk_config,
                router_logits=self.logits,
                layer_id=0,
                num_token_non_padded=torch.tensor(2),
                expert_location_dispatch_info=info,
            )
        self.assertEqual(len(solved), 1)
        self.assertEqual(output_ids.tolist(), [[2, 4], [0, 4], [-1, -1], [-1, -1]])
        torch.testing.assert_close(output_weights[:2], weights[:2])

    def test_lazy_lp_materialization_uses_its_owner_or_explicit_override(self):
        """Lazy dispatcher materialization must retain the layer's solver and capture slot."""
        get_resources().expert_location_metadata = _placement(num_layers=8)
        hidden = self.hidden.to(torch.bfloat16)
        group = SimpleNamespace(all_gatherv=lambda tensors, sizes: tensors)
        dispatcher = object.__new__(FlashinferDispatcher)
        dispatcher.ep_size = 1
        dispatcher.ep_rank = 0
        with get_parallel().override(moe_ep_size=1, moe_ep_rank=0):
            dispatcher.prefill_dispatcher = StandardDispatcher(
                SimpleNamespace(
                    runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                    num_experts=4,
                    num_local_experts=4,
                    num_fused_shared_experts=0,
                )
            )
        for owner, override, expected_owner in (
            (7, None, 7),
            (0, None, 0),
            (7, 0, 0),
            (0, 7, 7),
        ):
            with self.subTest(owner=owner, override=override):
                solved = []
                captured = {}

                def solver(layer_id):
                    def solve(ids):
                        self.assertEqual(ids.tolist(), [[0, 1]] * 4)
                        solved.append(layer_id)
                        return torch.ones((4, 1))

                    return SimpleNamespace(solve=solve)

                class Capture:
                    def capture(self, *, layer_id, topk_indices):
                        captured[layer_id] = topk_indices.clone()

                def dispatch_probability(ids, info, probability):
                    return info.partial_logical_to_rank_dispatch_physical_map[ids]

                with (
                    get_exec().moe.override(ep_dispatch_algorithm="lp"),
                    get_parallel().override(
                        tp_group=group, moe_ep_size=1, moe_ep_rank=0
                    ),
                    patch.dict(
                        get_resources().lplb_solvers,
                        {0: solver(0), 7: solver(7)},
                        clear=True,
                    ),
                    patch.object(get_resources(), "experts_capturer", Capture()),
                    patch.object(topk_module, "_is_cuda", True),
                    patch.object(
                        expert_location_dispatch,
                        "_topk_ids_logical_to_physical_probability",
                        dispatch_probability,
                    ),
                    patch(
                        "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
                        return_value=None,
                    ),
                ):
                    router = self._router(
                        layer_id=owner, output_format=TopKOutputFormat.BYPASSED
                    )
                    output = router.forward_cuda(
                        hidden, self.logits, num_token_non_padded=torch.tensor(2)
                    )
                    self.assertEqual(len(output), 5)
                    if override is not None:
                        output = output.to_standard(layer_id=override)
                    dispatched = dispatcher._dispatch_prefill_allgather(hidden, output)
                self.assertEqual(solved, [expected_owner])
                self.assertEqual(set(captured), {expected_owner})
                self.assertEqual(captured[expected_owner].tolist(), [[0, 1]] * 4)
                self.assertEqual(
                    dispatched.topk_output.topk_ids.tolist(),
                    [[2, 0], [2, 0], [-1, -1], [-1, -1]],
                )
                self.assertIs(dispatched.hidden_states, hidden)

    def test_empty_target_participates_in_lp_without_caller_layer_id(self):
        class Solver:
            def __init__(self):
                self.participations = []

            def solve(self, ids):
                self.participations.append(tuple(ids.shape))

        solver = Solver()
        get_resources().lplb_solvers[0] = solver
        self.addCleanup(get_resources().lplb_solvers.pop, 0)
        target = self._router()
        with draft_model_build_scope():
            draft = self._router()
        with get_parallel().override(tp_group=SimpleNamespace(world_size=1)):
            output = target.empty_topk_output(torch.device("cpu"))
            draft_output = draft.empty_topk_output(torch.device("cpu"))
        self.assertEqual(output.topk_ids.shape, (0, 2))
        self.assertEqual(draft_output.topk_ids.shape, (0, 2))
        self.assertEqual(solver.participations, [(0, 2)])

    def test_compile_reads_current_buffer(self):
        router = self._router(is_draft=True)
        batch = _batch(3)

        def mask_rows(x):
            bound, _ = router.routing_inputs()
            return x * (torch.arange(x.shape[0]) < bound)

        compiled = torch.compile(mask_rows, backend="eager", fullgraph=True)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=batch
        ):
            self.assertEqual(compiled(torch.ones(4)).tolist(), [1.0, 1.0, 1.0, 0.0])
            batch.num_token_non_padded.fill_(1)
            self.assertEqual(compiled(torch.ones(4)).tolist(), [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
