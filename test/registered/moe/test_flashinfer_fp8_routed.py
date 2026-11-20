"""Native FP8 routing keeps live-count bounds, score values, and deferred top-k."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.runtime_context import (
    get_context,
    get_flags,
    get_model,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# This uses one GPU; the pool supplies the required Blackwell FP8 kernels.
register_cuda_ci(est_time=20, stage="extra-b", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10,
    "Requires Blackwell FlashInfer TRTLLM FP8 kernels",
)
class TestFlashinferFp8Routed(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.layers.moe.utils import (
            MoeA2ABackend,
            MoeRunnerBackend,
            initialize_moe_config,
        )

        scopes = ExitStack()
        self.addCleanup(scopes.close)
        scopes.enter_context(
            get_context().override_server_args(
                model_path="dummy", device="cuda", enable_eplb=False
            )
        )
        initialize_moe_config()
        scopes.enter_context(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        scopes.enter_context(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                tp_group=None,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            )
        )
        scopes.enter_context(get_model().override(shared_experts=SharedExpertsConfig()))
        scopes.enter_context(torch.no_grad())

    @staticmethod
    def _group_roundtrip(value):
        grouped = value.float().reshape(value.shape[0], -1, 128)
        scales = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
        return ((grouped / scales).to(torch.float8_e4m3fn).float() * scales).reshape_as(
            value
        )

    def _check_live_routes(self, *, block_quant, dtype):
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as runner
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import TopK, TopKOutputChecker
        from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.llama4 import Llama4MoE

        m, h, inter, experts = 4, 256, 256, 4
        clean = (
            ((torch.arange(m * h, device="cuda").reshape(m, h) % 17) + 64) / 128
        ).to(dtype)
        hidden = clean.clone()
        clean_logits = torch.full((m, experts), -2.0, device="cuda")
        selected = torch.arange(m, device="cuda") % experts
        clean_logits[torch.arange(m, device="cuda"), selected] = (
            1.0 + torch.arange(m, device="cuda") / 8
        )
        logits = clean_logits.clone()
        live = torch.tensor(m, device="cuda", dtype=torch.int32)
        w13 = torch.zeros((experts, 2 * inter, h), device="cuda")
        w2 = torch.zeros((experts, h, inter), device="cuda")
        diagonal = torch.arange(h, device="cuda")
        for expert in range(experts):
            w13[expert, diagonal, diagonal] = 0.5 + expert / 8
            w13[expert, inter + diagonal, diagonal] = 0.25
            w2[expert, diagonal, diagonal] = 0.5
        w13q, w2q = w13.to(torch.float8_e4m3fn), w2.to(torch.float8_e4m3fn)
        a1, a2 = (
            torch.tensor([1 / 128], device="cuda"),
            torch.tensor([1 / 256], device="cuda"),
        )
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128] if block_quant else None,
        )
        quant_args = dict(
            w13_weight=w13q,
            w2_weight=w2q,
            global_num_experts=experts,
            local_expert_offset=0,
            local_num_experts=experts,
            intermediate_size=inter,
            routing_method_type=RoutingMethodType.Llama4,
            block_quant=block_quant,
        )
        if block_quant:
            # Fp8MoEMethod's TRTLLM loader stores the up half before the gate.
            quant_args.update(
                w13_weight=torch.cat((w13[:, inter:], w13[:, :inter]), dim=1).to(
                    torch.float8_e4m3fn
                ),
                weight_block_k=quant_config.weight_block_size[1],
                w13_weight_scale_inv=torch.ones(
                    (experts, 2 * inter // 128, h // 128), device="cuda"
                ),
                w2_weight_scale_inv=torch.ones(
                    (experts, h // 128, inter // 128), device="cuda"
                ),
            )
        else:
            layer = torch.nn.Module()
            layer.moe_runner_config = SimpleNamespace(is_gated=True)
            for name, value in {
                "w13_weight": w13q,
                "w2_weight": w2q,
                "w13_input_scale": a1,
                "w2_input_scale": a2,
                "w13_weight_scale": torch.ones(experts, device="cuda"),
                "w2_weight_scale": torch.ones(experts, device="cuda"),
            }.items():
                setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
            # Follow ModelOptFp8MoEMethod's preparation from logical gate/up.
            runner.align_fp8_moe_weights_for_flashinfer_trtllm(
                layer, swap_w13_halves=True
            )
            quant_args.update(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_input_scale=layer.w13_input_scale,
                output1_scales_scalar=layer.output1_scales_scalar,
                output1_scales_gate_scalar=layer.output1_scales_gate_scalar,
                output2_scales_scalar=layer.output2_scales_scalar,
            )
        quant = runner.FlashInferTrtllmFp8MoeQuantInfo(**quant_args)
        # Llama4's TopK owns k=1; its expert runner need not duplicate that setting.
        config = MoeRunnerConfig(activation="silu")
        router = TopK(
            top_k=1,
            layer_id=0,
            renormalize=False,
            custom_routing_function=Llama4MoE.custom_routing_function,
        )
        router.bind_runner_backend(
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            routing_method_type=RoutingMethodType.Llama4,
        )
        context = ForwardContext(
            attn_backend=object(), moe_num_token_non_padded=lambda: live
        )
        without_bound = ForwardContext(attn_backend=object())

        def entry(deferred, *, bounded=True):
            with forward_context(context if bounded else without_bound):
                lazy = router.forward_cuda(hidden, logits)
                self.assertTrue(TopKOutputChecker.format_is_bypassed(lazy))
                self.assertIs(lazy.num_token_non_padded, live if bounded else None)
                output = runner.fused_experts_none_to_flashinfer_trtllm_fp8(
                    StandardDispatchOutput(hidden, None, lazy), quant, config
                ).hidden_states
            if deferred:
                self.assertEqual(output.top_k, 1)
                return runner.finalize_flashinfer_trtllm_deferred_output(output, None)
            return output

        def reference(real):
            expected = torch.zeros((m, h), device="cuda", dtype=torch.bfloat16)
            if real == 0:
                return expected
            x = clean[:real].float()
            dequant = (
                self._group_roundtrip(x)
                if block_quant
                else (x / a1).to(torch.float8_e4m3fn).float() * a1
            )
            scores = clean_logits[:real].max(-1).values.sigmoid().to(dtype).float()
            if not block_quant:
                # Llama4 supplies BF16 token scales to the native FP8 GEMM1.
                dequant = dequant * scores.to(torch.bfloat16).float()[:, None]
            # Match this native runner's final-weight contract, independently
            # of its route materialization and grouped GEMM layout.
            for token in range(real):
                expert = int(selected[token])
                gate, up = torch.nn.functional.linear(
                    dequant[token], w13[expert]
                ).chunk(2)
                act = torch.nn.functional.silu(gate) * up
                aq = (
                    self._group_roundtrip(act[None])[0]
                    if block_quant
                    else (act / a2).to(torch.float8_e4m3fn).float() * a2
                )
                expected[token] = (
                    torch.nn.functional.linear(aq, w2[expert]) * scores[token]
                )
            return expected

        for deferred in (False, True) if block_quant else (False,):
            with (
                runner.flashinfer_trtllm_deferred_finalize_context()
                if deferred
                else nullcontext()
            ):
                valid_logits = entry(deferred, bounded=False).clone()
                valid_routed = entry(deferred).clone()
                torch.testing.assert_close(
                    valid_routed, valid_logits, rtol=0.01, atol=0.001
                )
                for _ in range(3):
                    entry(deferred)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = entry(deferred)
                for state, real in (
                    ("valid", m),
                    ("partial", 2),
                    ("inactive", 0),
                    ("recovery", m),
                ):
                    with self.subTest(deferred=deferred, state=state):
                        live.fill_(real)
                        hidden.copy_(clean)
                        logits.copy_(clean_logits)
                        hidden[real:] = logits[real:] = float("nan")
                        saved_hidden, saved_logits = hidden.clone(), logits.clone()
                        eager = entry(deferred).clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            eager.float(),
                            reference(real).float(),
                            rtol=0.01,
                            atol=0.001,
                        )
                        torch.testing.assert_close(
                            captured, eager, rtol=0.01, atol=0.001
                        )
                        self.assertTrue(bool(torch.isfinite(captured).all()))
                        self.assertTrue(bool((captured[real:] == 0).all()))
                        torch.testing.assert_close(
                            hidden, saved_hidden, rtol=0, atol=0, equal_nan=True
                        )
                        torch.testing.assert_close(
                            logits, saved_logits, rtol=0, atol=0, equal_nan=True
                        )
                del graph, captured
                live.fill_(m)
                hidden.copy_(clean)
                logits.copy_(clean_logits)

    def test_block_fp8_live_routes_and_deferred_finalize(self):
        self._check_live_routes(block_quant=True, dtype=torch.bfloat16)

    def test_per_tensor_fp8_half_routes(self):
        self._check_live_routes(block_quant=False, dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()
