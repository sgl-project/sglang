"""GGUF expert kernels drop inactive routes during padded graph replay."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize, quantize

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import gguf as gguf_runtime
from sglang.srt.layers.quantization.gguf import (
    GGUFConfig,
    GGUFMoEMethod,
    fused_moe_gguf,
    ggml_moe_get_block_size,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestGgufInactiveRoutes(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cuda")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.AUTO, a2a_backend=MoeA2ABackend.NONE
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_cp_size=1,
                tp_group=None,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            )
        )

    @staticmethod
    def _weights(quant_type):
        rng = np.random.default_rng(491)
        shape13, shape2 = (4, 512, 256), (4, 256, 256)
        packed, dense = [], []
        for shape in (shape13, shape2):
            weight = rng.normal(0, 0.03, shape).astype(np.float32)
            qweight = quantize(weight, quant_type)
            packed.append(torch.from_numpy(qweight.copy()).cuda())
            dense.append(torch.from_numpy(dequantize(qweight, quant_type)).cuda())
        return packed, dense

    @staticmethod
    def _method(packed, quant_type, *, global_experts=4):
        method = GGUFMoEMethod(GGUFConfig())
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            num_experts=4,
            hidden_size=256,
            intermediate_size_per_partition=256,
            params_dtype=torch.float16,
        )
        for param, weight in zip((layer.w13_qweight, layer.w2_qweight), packed):
            param.materialize(weight.shape, device=weight.device, dtype=weight.dtype)
            param.data.copy_(weight)
        layer.w13_qweight_type.weight_type = int(quant_type)
        layer.w2_qweight_type.weight_type = int(quant_type)
        config = MoeRunnerConfig(
            num_experts=global_experts,
            num_local_experts=4,
            num_fused_shared_experts=0,
            hidden_size=256,
            intermediate_size_per_partition=256,
            top_k=2,
        )
        method.create_moe_runner(layer, config)
        config.runner_backend = method.get_moe_runner_backend(layer)
        return method, layer, StandardDispatcher(config)

    @staticmethod
    def _q8_1_matmul(inputs, weight, packed_weight, quant_type):
        # GGUF GEMMs quantize each 32 input values to Q8_1, including an FP16
        # scale and input sum. CUDA roundf rounds halfway values away from zero.
        blocks = inputs.float().unflatten(-1, (-1, 32))
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        # A scalar divisor lets PyTorch multiply by its rounded reciprocal,
        # changing Q8_1 codes at ties compared with the native FP32 division.
        scale = amax / torch.full_like(amax, 127.0)
        denominator = torch.where(scale == 0, 1.0, scale)
        codes = blocks.sign() * torch.floor(blocks.abs() / denominator + 0.5)
        quantized = codes * scale.half().float()
        result = torch.einsum(
            "...bi,...nbi->...n", quantized, weight.unflatten(-1, (-1, 32))
        )
        if quant_type == GGMLQuantizationType.Q4_0:
            # Q4_0 subtracts its zero point using the original FP16 input sum,
            # rather than summing the dequantized Q8_1 codes.
            weight_scale = (
                packed_weight.unflatten(-1, (-1, 18))[..., :2]
                .contiguous()
                .view(torch.float16)
                .squeeze(-1)
                .float()
            )
            sum_error = quantized.sum(dim=-1) - blocks.sum(dim=-1).half().float()
            result += 8 * torch.einsum("...b,...nb->...n", sum_error, weight_scale)
        return result.to(inputs.dtype)

    @classmethod
    def _reference(cls, hidden, logits, dense, packed, quant_type):
        weights, ids = logits.float().softmax(dim=-1).topk(2, dim=-1)
        projected = cls._q8_1_matmul(
            hidden.unsqueeze(1), dense[0][ids], packed[0][ids], quant_type
        )
        gate, up = projected.float().chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * up).to(hidden.dtype)
        down = cls._q8_1_matmul(activated, dense[1][ids], packed[1][ids], quant_type)
        weighted = (down.float() * weights.unsqueeze(-1)).to(hidden.dtype)
        return weighted.sum(dim=1).float()

    def test_block_size_metadata_without_tensor_arguments(self):
        # Runtime metadata must work without a Tensor-dispatched wheel operator.
        for quant_type in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0):
            with self.subTest(quant_type=quant_type):
                self.assertIn(ggml_moe_get_block_size(int(quant_type)), (4, 8))

    @torch.inference_mode()
    def test_padded_graph_replay_matches_unpadded_experts(self):
        """Changing the live count must zero dropped GGUF routes on every replay."""
        for quant_type in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0):
            packed, dense = self._weights(quant_type)
            for tokens, partial in ((64, 48), (65, 48), (96, 71)):
                with self.subTest(quant_type=quant_type, tokens=tokens):
                    self._check_graph(packed, dense, quant_type, tokens, partial)

    def _check_graph(self, packed, dense, quant_type, tokens, partial):
        method, layer, dispatcher = self._method(packed, quant_type)
        topk = TopK(
            top_k=2, layer_id=0, renormalize=False, allow_routed_experts_capture=False
        )
        topk.bind_runner_backend(method.get_moe_runner_backend(layer))
        generator = torch.Generator(device="cuda").manual_seed(tokens)
        base_hidden = torch.randn(
            tokens, 256, generator=generator, device="cuda"
        ).half()
        base_logits = torch.randn(tokens, 4, generator=generator, device="cuda")
        logits = base_logits.clone()
        hidden = base_hidden.clone()
        live_count = torch.tensor(tokens, device="cuda", dtype=torch.int32)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.zeros(tokens, device="cuda", dtype=torch.int64),
            req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
            seq_lens=torch.tensor([tokens], device="cuda", dtype=torch.int32),
            out_cache_loc=torch.arange(tokens, device="cuda"),
            seq_lens_sum=tokens,
            positions=torch.arange(tokens, device="cuda"),
            num_token_non_padded=live_count,
            global_num_token_non_padded=live_count,
        )

        def run():
            with forward_context(
                ForwardContext(attn_backend=None), forward_batch=batch
            ):
                routes = topk.forward_cuda(hidden, logits)
                dispatched = dispatcher.dispatch(hidden, routes)
                combined = method.apply(layer, dispatched)
                return dispatcher.combine(combined), routes.topk_ids

        native_outputs = []
        native_mmq = gguf_runtime.ggml_moe_a8

        def record_mmq_outputs(*args, **kwargs):
            if not torch.cuda.is_current_stream_capturing():
                # Fail before native dispatch if a slack block can index expert E.
                self.assertTrue(bool((args[3] < args[1].shape[0]).all()))
                used_blocks = args[4].item() // 4
                self.assertTrue(bool((args[3][used_blocks:] == -1).all()))
            result = native_mmq(*args, **kwargs)
            if torch.cuda.is_current_stream_capturing():
                native_outputs.append(result)
            return result

        # Observe the real provider's returned buffers without replacing its
        # computation. Poisoning them makes a skipped MMQ store deterministic.
        observe_mmq = (
            patch.object(gguf_runtime, "ggml_moe_a8", record_mmq_outputs)
            if tokens > 64
            else nullcontext()
        )
        with observe_mmq:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, ids = run()
        self.assertEqual(len(native_outputs), 2 if tokens > 64 else 0)
        # Dropped routes must not consume poisoned native intermediates.
        for live in (tokens, partial, 0, partial // 2, tokens):
            hidden.copy_(base_hidden)
            hidden[live:].fill_(float("nan"))
            logits.copy_(base_logits)
            logits[live:].fill_(float("nan"))
            live_count.fill_(live)
            output.fill_(float("nan"))
            for intermediate in native_outputs:
                intermediate.fill_(float("nan"))
            graph.replay()
            self.assertTrue(torch.isfinite(output).all().item())
            self.assertTrue((ids[live:] == -1).all().item())
            self.assertEqual(torch.count_nonzero(output[live:]).item(), 0)
            if live:
                expected = self._reference(
                    base_hidden[:live], base_logits[:live], dense, packed, quant_type
                )
                self.assertGreater(expected.abs().max().item(), 1e-3)
                torch.testing.assert_close(
                    output[:live].float(), expected, atol=2e-3, rtol=0.06
                )

    @torch.inference_mode()
    def test_mmq_prefix_preserves_activation_memory_budget(self):
        """Preparing MMQ routes must not duplicate the caller's activation matrix."""
        tokens, hidden, intermediate, experts = 8192, 4096, 128, 4
        inputs = torch.ones(tokens, hidden, device="cuda", dtype=torch.float16)
        ids = torch.zeros(tokens, 2, device="cuda", dtype=torch.int32)
        weights = torch.full((tokens, 2), 0.5, device="cuda")
        packed_up = torch.empty(
            experts,
            2 * intermediate,
            hidden // 32 * 18,
            device="cuda",
            dtype=torch.uint8,
        )
        packed_down = torch.empty(
            experts,
            hidden,
            intermediate // 32 * 18,
            device="cuda",
            dtype=torch.uint8,
        )

        class ProjectionReached(Exception):
            pass

        def stop_before_projection(*args, **kwargs):
            raise ProjectionReached

        def run_prefix():
            with self.assertRaises(ProjectionReached):
                fused_moe_gguf(
                    inputs,
                    packed_up,
                    packed_down,
                    weights,
                    ids,
                    GGMLQuantizationType.Q4_0,
                    GGMLQuantizationType.Q4_0,
                    "silu",
                )

        from torch.utils._python_dispatch import TorchDispatchMode

        allocations = []

        class ObserveAllocations(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                input_storage = {
                    arg.untyped_storage().data_ptr()
                    for arg in args
                    if isinstance(arg, torch.Tensor)
                }
                if (
                    isinstance(result, torch.Tensor)
                    and result.numel()
                    and result.untyped_storage().data_ptr() not in input_storage
                ):
                    allocations.append(result.untyped_storage().nbytes())
                return result

        with patch.object(gguf_runtime, "ggml_moe_a8", stop_before_projection):
            run_prefix()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with ObserveAllocations():
                run_prefix()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - allocated
        # One output and four alignment buffers; no intermediate masks or ranges.
        self.assertLessEqual(len(allocations), 5)
        budget = inputs.numel() * inputs.element_size() + 128 * 1024
        self.assertLessEqual(peak, budget)

    @torch.inference_mode()
    def test_nonlocal_routes_preserve_local_contributions(self):
        """A nonlocal slot must not erase the same token's valid expert result."""
        packed, dense = self._weights(GGMLQuantizationType.Q4_0)
        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            moe_ep_size=2,
            moe_ep_rank=1,
        ):
            method, layer, dispatcher = self._method(
                packed, GGMLQuantizationType.Q4_0, global_experts=8
            )
        hidden = torch.full((3, 256), 0.25, device="cuda", dtype=torch.float16)
        ids = torch.tensor([[4, 1], [-1, 5], [0, 2]], device="cuda", dtype=torch.int32)
        weights = torch.tensor(
            [[0.7, float("nan")], [float("nan"), 0.6], [float("nan"), float("nan")]],
            device="cuda",
        )
        hidden[2].fill_(float("nan"))
        routes = StandardTopKOutput(weights, ids, None)
        output = dispatcher.combine(
            method.apply(layer, dispatcher.dispatch(hidden, routes))
        )
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(torch.count_nonzero(output[2]).item(), 0)
        for token, expert, scale in ((0, 0, 0.7), (1, 1, 0.6)):
            projected = self._q8_1_matmul(
                hidden[token],
                dense[0][expert],
                packed[0][expert],
                GGMLQuantizationType.Q4_0,
            )
            gate, up = projected.float().chunk(2)
            activated = (torch.nn.functional.silu(gate) * up).half()
            expected = (
                (
                    self._q8_1_matmul(
                        activated,
                        dense[1][expert],
                        packed[1][expert],
                        GGMLQuantizationType.Q4_0,
                    ).float()
                    * scale
                )
                .half()
                .float()
            )
            torch.testing.assert_close(
                output[token].float(), expected, atol=2e-3, rtol=0.06
            )

    @torch.inference_mode()
    def test_cuda_vector_uses_original_ids_without_route_preparation(self):
        """Real CUDA projections consume negative routes without a clamp launch."""
        from sgl_kernel.quantization import ggml_moe_a8_vec as legacy_vector
        from torch.utils._python_dispatch import TorchDispatchMode

        ids = torch.tensor(
            [[-1, 1], [0, -3], [-1, -1]], device="cuda", dtype=torch.int32
        )
        scores = torch.tensor(
            [[float("nan"), 0.75], [0.5, float("nan")], [float("nan"), float("nan")]],
            device="cuda",
        )
        hidden = torch.full((3, 256), 0.25, device="cuda", dtype=torch.float16)
        hidden[-1].fill_(float("nan"))
        raw_ids = ids.clone()
        route_ops = []
        actual_vector = gguf_runtime.ggml_moe_a8_vec

        class ObserveRoutePreparation(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in (
                    torch.ops.aten.clamp_min.default,
                    torch.ops.aten.clamp_min.out,
                    torch.ops.aten._to_copy.default,
                ):
                    route_ops.append(str(func))
                return func(*args, **(kwargs or {}))

        def observe_vector(inputs, weights, routes, *args):
            self.assertEqual(routes.data_ptr(), ids.data_ptr())
            return actual_vector(inputs, weights, routes, *args)

        for quant_type in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0):
            packed, _ = self._weights(quant_type)
            with patch.object(gguf_runtime, "ggml_moe_a8_vec", legacy_vector):
                expected = fused_moe_gguf(
                    hidden.nan_to_num(),
                    *packed,
                    scores.nan_to_num(),
                    ids.clamp_min(0),
                    quant_type,
                    quant_type,
                    "silu",
                )
            route_ops.clear()
            with (
                ObserveRoutePreparation(),
                patch.object(gguf_runtime, "ggml_moe_a8_vec", observe_vector),
            ):
                actual = fused_moe_gguf(
                    hidden, *packed, scores, ids, quant_type, quant_type, "silu"
                )
            self.assertEqual(route_ops, [])
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(ids, raw_ids, rtol=0, atol=0)

    @torch.inference_mode()
    def test_musa_branch_preserves_input_and_route_budgets(self):
        """MUSA consumers exclude NaNs without input copies or route temporaries.

        CUDA execution covers Python dispatch; native MUSA runs in its own CI lane.
        """
        from sgl_kernel import gelu_and_mul as legacy_gelu_and_mul
        from sgl_kernel import silu_and_mul as legacy_silu_and_mul
        from torch.utils._python_dispatch import TorchDispatchMode

        from sglang.kernels.ops.moe.gguf_moe_weights_aot import resolve_gguf_moe_weights
        from sglang.srt.utils import is_musa

        finalizer = resolve_gguf_moe_weights("musa" if is_musa() else "cuda")
        finalizers = (None,) if finalizer is None else (None, finalizer)
        for finalizer in finalizers:
            with self.subTest(native_finalizer=finalizer is not None):
                for quant_type in (
                    GGMLQuantizationType.Q4_0,
                    GGMLQuantizationType.Q8_0,
                ):
                    packed, _ = self._weights(quant_type)
                    for tokens in (3, 96):
                        for activation in ("silu", "gelu"):
                            with self.subTest(
                                quant_type=quant_type,
                                tokens=tokens,
                                activation=activation,
                            ):
                                repeats = (tokens + 2) // 3
                                ids = torch.tensor(
                                    [[0, 1], [1, -1], [-1, -1]],
                                    device="cuda",
                                    dtype=torch.int32,
                                ).repeat(repeats, 1)[:tokens]
                                weights = torch.tensor(
                                    [
                                        [0.25, 0.75],
                                        [0.5, float("nan")],
                                        [float("nan"), float("nan")],
                                    ],
                                    device="cuda",
                                ).repeat(repeats, 1)[:tokens]
                                inactive = ids < 0
                                inputs = (
                                    torch.linspace(
                                        0.125,
                                        2.0,
                                        tokens,
                                        device="cuda",
                                        dtype=torch.float16,
                                    )
                                    .unsqueeze(1)
                                    .expand(tokens, 256)
                                    .contiguous()
                                )
                                inputs[inactive.all(dim=-1)] = float("nan")
                                provider_name = (
                                    "ggml_moe_a8" if tokens > 64 else "ggml_moe_a8_vec"
                                )
                                native = (
                                    gguf_runtime.ggml_moe_a8
                                    if tokens > 64
                                    else gguf_runtime.ggml_moe_a8_vec
                                )
                                calls = 0
                                route_costs = []

                                class ObserveRouteCosts(TorchDispatchMode):
                                    def __torch_dispatch__(
                                        self, func, types, args=(), kwargs=None
                                    ):
                                        if func in (
                                            torch.ops.aten.lt.Scalar,
                                            torch.ops.aten.clamp_min.default,
                                            torch.ops.aten.clamp_min.out,
                                            torch.ops.aten.masked_fill.Scalar,
                                            torch.ops.aten.masked_fill_.Scalar,
                                            torch.ops.aten.arange.default,
                                        ):
                                            route_costs.append(str(func))
                                        return func(*args, **(kwargs or {}))

                                def observe_projection(
                                    projection_input, *args, **kwargs
                                ):
                                    nonlocal calls
                                    if calls == 0:
                                        self.assertEqual(
                                            projection_input.data_ptr(),
                                            inputs.data_ptr(),
                                        )
                                    if finalizer is not None and tokens <= 64:
                                        self.assertIs(args[1], ids)
                                    calls += 1
                                    return native(projection_input, *args, **kwargs)

                                with (
                                    patch.object(gguf_runtime, "_is_cuda", False),
                                    patch.object(gguf_runtime, "_is_musa", True),
                                    patch.object(
                                        gguf_runtime,
                                        "_musa_apply_gguf_moe_weights",
                                        None,
                                    ),
                                    patch.object(
                                        gguf_runtime,
                                        "silu_and_mul",
                                        legacy_silu_and_mul,
                                    ),
                                    patch.object(
                                        gguf_runtime,
                                        "gelu_and_mul",
                                        legacy_gelu_and_mul,
                                    ),
                                ):
                                    expected = fused_moe_gguf(
                                        inputs.nan_to_num(),
                                        *packed,
                                        weights.masked_fill(inactive, 0),
                                        ids.clamp_min(0),
                                        quant_type,
                                        quant_type,
                                        activation,
                                    )
                                    with (
                                        ObserveRouteCosts(),
                                        patch.object(
                                            gguf_runtime,
                                            provider_name,
                                            observe_projection,
                                        ),
                                        patch.object(
                                            gguf_runtime,
                                            "_musa_apply_gguf_moe_weights",
                                            finalizer,
                                        ),
                                    ):
                                        actual = fused_moe_gguf(
                                            inputs,
                                            *packed,
                                            weights,
                                            ids,
                                            quant_type,
                                            quant_type,
                                            activation,
                                        )
                                self.assertEqual(calls, 2)
                                if finalizer is not None:
                                    self.assertEqual(route_costs, [])
                                self.assertGreater(expected.abs().max().item(), 1e-3)
                                self.assertTrue(torch.isfinite(actual).all().item())
                                self.assertEqual(
                                    torch.count_nonzero(
                                        actual[inactive.all(dim=-1)]
                                    ).item(),
                                    0,
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

    @torch.inference_mode()
    def test_unquantized_fallback_drops_negative_ids(self):
        """Dropped routes stay zero without adding a valid-route accumulator."""
        from torch.utils._python_dispatch import TorchDispatchMode

        generator = torch.Generator(device="cuda").manual_seed(17)
        w13 = torch.randn(2, 32, 16, generator=generator, device="cuda").half() / 16
        w2 = torch.randn(2, 16, 16, generator=generator, device="cuda").half() / 16
        hidden = torch.ones(4, 16, device="cuda", dtype=torch.float16)
        hidden[-1].fill_(float("nan"))
        ids = torch.tensor(
            [[0, -3, -1], [0, 1, 0], [-1, 1, 0], [-1, -3, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        weights = torch.tensor(
            [
                [1.0, float("nan"), float("nan")],
                [0.2, 0.3, 0.5],
                [float("nan"), 0.75, 0.25],
                [float("nan"), float("nan"), float("nan")],
            ],
            device="cuda",
        )
        original_hidden, original_ids, original_weights = (
            hidden.clone(),
            ids.clone(),
            weights.clone(),
        )
        observed = {
            op: []
            for op in (
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.add_.Tensor,
                torch.ops.aten.mul_.Tensor,
                torch.ops.aten.zero_.default,
            )
        }

        class ObserveTensorOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in observed:
                    observed[func].append(args[0].data_ptr())
                return func(*args, **(kwargs or {}))

        with ObserveTensorOps():
            output = fused_moe_gguf(
                hidden,
                w13,
                w2,
                weights,
                ids,
                GGMLQuantizationType.F16,
                GGMLQuantizationType.F16,
                "silu",
            )
        projected = torch.einsum("th,enh->ten", hidden.nan_to_num(), w13)
        gate, up = projected.float().chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * up).to(hidden.dtype)
        experts = torch.einsum("tek,eok->teo", activated, w2)
        selected = experts.gather(
            1, ids.clamp_min(0).long().unsqueeze(-1).expand(-1, -1, hidden.shape[1])
        )
        expected = (
            selected.float() * weights.masked_fill(ids < 0, 0).unsqueeze(-1)
        ).sum(dim=1)
        self.assertGreater(expected.abs().max().item(), 1e-3)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(torch.count_nonzero(output[-1]).item(), 0)
        torch.testing.assert_close(output.float(), expected, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(hidden, original_hidden, equal_nan=True)
        torch.testing.assert_close(ids, original_ids)
        torch.testing.assert_close(weights, original_weights, equal_nan=True)
        self.assertNotEqual(output.data_ptr(), hidden.data_ptr())
        self.assertEqual(observed[torch.ops.aten.zeros_like.default], [])
        self.assertEqual(
            observed[torch.ops.aten.zero_.default], [output[-1].data_ptr()]
        )
        weighted_outputs = observed[torch.ops.aten.mul_.Tensor]
        self.assertEqual(len(weighted_outputs), 6)
        self.assertEqual(
            observed[torch.ops.aten.add_.Tensor],
            [weighted_outputs[1], weighted_outputs[1], weighted_outputs[4]],
        )


if __name__ == "__main__":
    unittest.main()
