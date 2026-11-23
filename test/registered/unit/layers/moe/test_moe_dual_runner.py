"""Unit tests for Routing-Aware Dual-Runner MoE dispatch."""

from __future__ import annotations

import unittest
from contextlib import contextmanager

from sglang.srt.environ import envs

if not envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.is_set():
    envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.dual_runner import (
    DualMoeRunner,
    get_moe_dual_runner_unsupported_reason,
    resolve_moe_dual_runner_threshold,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import (
    extract_fp32_block_scale_from_ue8m0,
    per_block_cast_to_fp8,
    transform_scale_ue8m0,
)
from sglang.srt.runtime_context import get_context, get_flags
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _make_routing(m: int, num_experts: int, top_k: int, *, skewed: bool = False):
    g = torch.Generator(device="cpu").manual_seed(1234 + m)
    if skewed:
        probs = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float32)
        ids = torch.stack(
            [torch.multinomial(probs, top_k, generator=g) for _ in range(m)]
        )
    else:
        ids = torch.stack(
            [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(m)]
        )
    weights = torch.softmax(torch.randn(m, top_k, generator=g), dim=-1)
    return weights.cuda(), ids.to(torch.int32).cuda()


class TestMoeDualRunner(CustomTestCase):
    _dist_initialized = False

    @classmethod
    def _ensure_cuda_dist(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if not cls._dist_initialized:
            init_single_process_dist(master_port=find_available_port(29640))
            cls._dist_initialized = True

    @contextmanager
    def _moe_runtime_env(
        self,
        *,
        threshold: int = 0,
        runner_backend: MoeRunnerBackend = MoeRunnerBackend.AUTO,
        a2a_backend: MoeA2ABackend = MoeA2ABackend.NONE,
        fuse_swiglu_interleaved: bool = False,
        precompile_stage: bool = False,
    ):
        self._ensure_cuda_dist()
        with (
            envs.SGLANG_MOE_DUAL_RUNNER_THRESHOLD.override(threshold),
            envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.override(fuse_swiglu_interleaved),
            envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.override(precompile_stage),
            get_context().override_server_args(
                model_path="dummy",
                moe_runner_backend=runner_backend.value,
                moe_a2a_backend=a2a_backend.value,
            ),
            get_flags().moe.override(
                runner_backend=runner_backend,
                a2a_backend=a2a_backend,
            ),
        ):
            yield

    def _make_fused_moe(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        apply_router_weight_on_input: bool = False,
        with_bias: bool = False,
        w13: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        process_weights: bool = False,
    ):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        layer = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=0,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            with_bias=with_bias,
            inplace=False,
        ).cuda()
        if w13 is not None:
            layer.w13_weight.data.copy_(w13)
        if w2 is not None:
            layer.w2_weight.data.copy_(w2)
        if w13_scale is not None:
            layer.w13_weight_scale_inv.data.copy_(w13_scale)
        if w2_scale is not None:
            layer.w2_weight_scale_inv.data.copy_(w2_scale)
        if process_weights:
            layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_resolve_threshold_formula(self):
        cfg = MoeRunnerConfig(num_experts=128, num_local_experts=128, top_k=8)
        self.assertEqual(resolve_moe_dual_runner_threshold(0, cfg, is_fp8=False), 0)
        self.assertEqual(resolve_moe_dual_runner_threshold(512, cfg, is_fp8=False), 512)
        self.assertEqual(resolve_moe_dual_runner_threshold(-1, cfg, is_fp8=False), 8192)
        self.assertEqual(
            resolve_moe_dual_runner_threshold(-1, cfg, is_fp8=True, scale_ue8m0=False),
            16384,
        )
        self.assertEqual(
            resolve_moe_dual_runner_threshold(-1, cfg, is_fp8=True, scale_ue8m0=True),
            4096,
        )

        cfg_ep4 = MoeRunnerConfig(num_experts=128, num_local_experts=32, top_k=8)
        self.assertEqual(
            resolve_moe_dual_runner_threshold(-1, cfg_ep4, is_fp8=False), 2048
        )

    def test_extract_fp32_block_scale_from_ue8m0(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            self.skipTest("JIT DeepGEMM is required for UE8M0 layout test")

        device = "cuda"
        for num_experts, mn, k in ((4, 1536, 2048), (4, 2048, 768), (4, 256, 128)):
            sn, sk = mn // 128, k // 128
            exps = torch.randint(
                115, 132, (num_experts, sn, sk), dtype=torch.int32, device=device
            )
            expected_fp32 = (exps << 23).view(torch.float32)
            for use_torch_impl in (True, False):
                packed = transform_scale_ue8m0(
                    expected_fp32, mn=mn, use_torch_impl=use_torch_impl
                )
                recovered = extract_fp32_block_scale_from_ue8m0(packed, mn=mn, k=k)
                self.assertEqual(recovered.dtype, torch.float32)
                self.assertEqual(recovered.shape, expected_fp32.shape)
                self.assertTrue(torch.equal(recovered, expected_fp32))

    def test_bf16_dual_runner_dispatch_numerics_and_cudagraph(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if get_device_sm() not in (90, 100):
            self.skipTest("Dual-Runner requires SM90 or SM100")
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            self.skipTest("JIT DeepGEMM is required for BF16 Dual-Runner test")

        for intermediate_size in (768, 384, 192):
            with self.subTest(intermediate_size=intermediate_size):
                self._check_bf16_dual_runner(intermediate_size)

        # Verify `sglang.compile_deep_gemm` forces threshold=1 so warmup requests
        # compile DeepGEMM contiguous kernels ahead of serving.
        with self._moe_runtime_env(
            threshold=-1,
            runner_backend=MoeRunnerBackend.AUTO,
            precompile_stage=True,
        ):
            precompile_layer = self._make_fused_moe(
                num_experts=32,
                top_k=8,
                hidden_size=2048,
                intermediate_size=768,
            )
            self.assertIsNotNone(precompile_layer.quant_method.dual_runner)
            self.assertEqual(precompile_layer.quant_method.dual_runner.threshold, 1)
            self.assertEqual(
                precompile_layer.moe_runner_config.dual_runner_threshold, 1
            )

    def _check_bf16_dual_runner(self, intermediate_size: int):
        num_experts, hidden_size, top_k = 32, 2048, 8
        torch.manual_seed(42)
        w13 = (
            torch.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        )
        w2 = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        )

        with self._moe_runtime_env(
            threshold=0, runner_backend=MoeRunnerBackend.TRITON
        ):
            ref_layer = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                w13=w13,
                w2=w2,
                process_weights=True,
            )

        # Verify SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED=1 cannot interleave w13 when Dual-Runner is active.
        with self._moe_runtime_env(
            threshold=128,
            runner_backend=MoeRunnerBackend.AUTO,
            fuse_swiglu_interleaved=True,
        ):
            dual_layer = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                w13=w13,
                w2=w2,
                process_weights=True,
            )

            qm = dual_layer.quant_method
            self.assertIsNotNone(qm.dual_runner)
            self.assertEqual(qm.dual_runner.threshold, 128)
            self.assertEqual(dual_layer.moe_runner_config.dual_runner_threshold, 128)
            self.assertFalse(qm.w13_swiglu_interleaved)

            calls = []
            orig_triton_run = qm.dual_runner.runner_triton.run
            orig_dg_run = qm.dual_runner.runner_deep_gemm.run

            def wrapped_triton_run(*args, **kwargs):
                calls.append(("triton", args[0].hidden_states.shape[0]))
                return orig_triton_run(*args, **kwargs)

            def wrapped_dg_run(*args, **kwargs):
                calls.append(("deep_gemm", args[0].hidden_states.shape[0]))
                return orig_dg_run(*args, **kwargs)

            qm.dual_runner.runner_triton.run = wrapped_triton_run
            qm.dual_runner.runner_deep_gemm.run = wrapped_dg_run

            for m, expected_backend, skewed in (
                (1, "triton", False),
                (16, "triton", True),
                (64, "triton", True),
                (128, "deep_gemm", False),
                (512, "deep_gemm", True),
            ):
                x = (
                    torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
                    * 0.5
                )
                tw, ti = _make_routing(m, num_experts, top_k, skewed=skewed)
                topk_out = StandardTopKOutput(tw, ti, None)
                calls.clear()
                got = dual_layer(x, topk_out)
                ref = ref_layer(x, topk_out)
                self.assertEqual(calls, [(expected_backend, m)])
                rel_err = float(
                    (got.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-6)
                )
                self.assertLess(rel_err, 1e-2)

            # Verify CUDA Graph capture & replay on both Triton (< threshold) and
            # DeepGEMM compact (>= threshold) paths with zero host-device sync.
            for m_cg, expected_backend in ((8, "triton"), (256, "deep_gemm")):
                x_cg = (
                    torch.randn(m_cg, hidden_size, dtype=torch.bfloat16, device="cuda")
                    * 0.5
                )
                tw_cg, ti_cg = _make_routing(m_cg, num_experts, top_k, skewed=True)
                topk_cg = StandardTopKOutput(tw_cg, ti_cg, None)
                ref_cg = ref_layer(x_cg, topk_cg)

                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(2):
                        _ = dual_layer(x_cg, topk_cg)
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()

                calls.clear()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out_cg = dual_layer(x_cg, topk_cg)
                self.assertEqual(calls, [(expected_backend, m_cg)])
                g.replay()
                torch.cuda.synchronize()
                cg_rel_err = float(
                    (out_cg.float() - ref_cg.float()).norm()
                    / ref_cg.float().norm().clamp_min(1e-6)
                )
                self.assertLess(cg_rel_err, 1e-2)
                del g

    def test_fp8_block_dual_runner_numerics_and_guardrails(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if get_device_sm() not in (90, 100):
            self.skipTest("Dual-Runner requires SM90 or SM100")
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            self.skipTest("JIT DeepGEMM is required for FP8 Dual-Runner test")

        num_experts, hidden_size, intermediate_size, top_k = 32, 2048, 768, 8
        torch.manual_seed(7)
        w13_bf16 = (
            torch.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        )
        w2_bf16 = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        )

        # Build FP8 weights with deliberately non-power-of-two per-block scales.
        w13_fp8 = torch.empty_like(w13_bf16, dtype=torch.float8_e4m3fn)
        w13_scale = torch.empty(
            num_experts,
            (2 * intermediate_size) // 128,
            hidden_size // 128,
            dtype=torch.float32,
            device="cuda",
        )
        w2_fp8 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
        w2_scale = torch.empty(
            num_experts,
            hidden_size // 128,
            intermediate_size // 128,
            dtype=torch.float32,
            device="cuda",
        )
        for e in range(num_experts):
            q1, s1 = per_block_cast_to_fp8(w13_bf16[e].float())
            q2, s2 = per_block_cast_to_fp8(w2_bf16[e].float())
            jitter1 = torch.empty_like(s1).uniform_(0.85, 1.15)
            jitter2 = torch.empty_like(s2).uniform_(0.85, 1.15)
            w13_fp8[e].copy_(q1)
            w13_scale[e].copy_(s1 * jitter1)
            w2_fp8[e].copy_(q2)
            w2_scale[e].copy_(s2 * jitter2)

        fp8_cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        with self._moe_runtime_env(
            threshold=0, runner_backend=MoeRunnerBackend.TRITON
        ):
            self.assertFalse(Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled())
            ref_layer = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=fp8_cfg,
                w13=w13_fp8,
                w2=w2_fp8,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                process_weights=True,
            )

        with self._moe_runtime_env(
            threshold=128, runner_backend=MoeRunnerBackend.AUTO
        ):
            dual_layer = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=fp8_cfg,
                w13=w13_fp8,
                w2=w2_fp8,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                process_weights=True,
            )

            qm = dual_layer.quant_method
            self.assertIsNotNone(qm.dual_runner)
            self.assertEqual(qm.dual_runner.threshold, 128)
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
                self.assertTrue(qm._has_fp32_ue8m0_scales)
                self.assertNotIn("w13_weight_scale_inv_fp32", dual_layer.state_dict())

            # On SM100 UE8M0, weights are requantized to power-of-two scales, adding ~3-4% vs SM90 FP32 scale ref.
            tol = 6e-2 if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 else 2e-2
            for m, skewed in ((8, True), (256, False)):
                x = (
                    torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
                    * 0.5
                )
                tw, ti = _make_routing(m, num_experts, top_k, skewed=skewed)
                topk_out = StandardTopKOutput(tw, ti, None)
                got = dual_layer(x, topk_out)
                ref = ref_layer(x, topk_out)
                rel_err = float(
                    (got.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-6)
                )
                self.assertLess(rel_err, tol)

        # Guardrail 1: Explicit `--moe-runner-backend triton` must NOT activate Dual-Runner.
        with self._moe_runtime_env(
            threshold=128, runner_backend=MoeRunnerBackend.TRITON
        ):
            triton_only = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=fp8_cfg,
            )
            self.assertIsNone(triton_only.quant_method.dual_runner)

        # Guardrail 2: Per-tensor FP8 & unsupported features fall back cleanly.
        fp8_pertensor_cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=None,
        )
        with self._moe_runtime_env(
            threshold=-1, runner_backend=MoeRunnerBackend.AUTO
        ):
            pertensor_fp8 = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=192,
                quant_config=fp8_pertensor_cfg,
            )
            self.assertIsNone(pertensor_fp8.quant_method.dual_runner)

            arwi_layer = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=fp8_cfg,
                apply_router_weight_on_input=True,
            )
            self.assertIsNone(arwi_layer.quant_method.dual_runner)

            bias_layer = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=fp8_cfg,
                with_bias=True,
            )
            self.assertIsNone(bias_layer.quant_method.dual_runner)

        # Guardrail 3: DeepEP A2A backend keeps Dual-Runner disabled.
        with self._moe_runtime_env(
            threshold=-1,
            runner_backend=MoeRunnerBackend.AUTO,
            a2a_backend=MoeA2ABackend.DEEPEP,
        ):
            reason = get_moe_dual_runner_unsupported_reason(
                arwi_layer.moe_runner_config, with_bias=False
            )
            self.assertIn("moe_a2a_backend", reason or "")
            self.assertIsNone(
                DualMoeRunner.try_create(
                    arwi_layer,
                    arwi_layer.moe_runner_config,
                    is_fp8=True,
                    with_bias=False,
                    quant_unsupported_reason_fn=lambda: None,
                    get_triton_quant_info=arwi_layer.quant_method.get_triton_quant_info,
                    get_deep_gemm_quant_info=arwi_layer.quant_method._get_deep_gemm_quant_info,
                    owner_name="Fp8MoEMethod",
                    label="FP8",
                )
            )

        # Guardrail 4: Auto `-1` threshold resolves to M_eff=1024 (SM90) / M_eff=256 (SM100 UE8M0) & enforces inter % 128 == 0.
        with self._moe_runtime_env(
            threshold=-1, runner_backend=MoeRunnerBackend.AUTO
        ):
            auto_fp8 = self._make_fused_moe(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=fp8_cfg,
            )
            self.assertIsNotNone(auto_fp8.quant_method.dual_runner)
            expected_auto = 1024 if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 else 4096
            self.assertEqual(auto_fp8.quant_method.dual_runner.threshold, expected_auto)
            auto_fp8.intermediate_size_per_partition = 192
            self.assertIn(
                "128",
                auto_fp8.quant_method._dual_runner_unsupported_reason(
                    auto_fp8, auto_fp8.moe_runner_config
                )
                or "",
            )


if __name__ == "__main__":
    unittest.main()
