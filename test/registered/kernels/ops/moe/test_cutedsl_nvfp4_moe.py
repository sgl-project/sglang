"""Correctness tests for Blackwell SM100 CuTe DSL NVFP4 masked and routed MoE."""

import unittest

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.moe import nvfp4_masked_grouped_moe, nvfp4_routed_grouped_moe
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _ref_flashinfer_moe_masked(
    x: torch.Tensor,
    in_gs: torch.Tensor,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    a2_gs: torch.Tensor,
    w2_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    masked_m: torch.Tensor,
) -> torch.Tensor:
    from flashinfer import (
        scaled_fp4_grouped_quantize,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )
    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

    G, m_max, k = x.shape
    n = w2.shape[-1] * 2
    a_q, a_q_sf = scaled_fp4_grouped_quantize(x, masked_m, in_gs)
    inter = torch.empty(
        (G, m_max, 2 * n), dtype=torch.bfloat16, device=x.device
    ).permute(1, 2, 0)
    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (w1.permute(1, 2, 0), w1_sf),
        inter,
        masked_m,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        alpha=w1_alpha.view(1, 1, G),
        alpha_dtype="float32",
    )
    inter = inter.permute(2, 0, 1)
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(inter, masked_m, a2_gs)
    out = torch.empty((G, m_max, k), dtype=torch.bfloat16, device=x.device).permute(
        1, 2, 0
    )
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_sf),
        out,
        masked_m,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        alpha=w2_alpha.view(1, 1, G),
        alpha_dtype="float32",
    )
    return out.permute(2, 0, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCuteDSLNvfp4MoE(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_hip_runtime() or get_jit_cuda_arch().major != 10:
            raise unittest.SkipTest("NVIDIA SM10x required")

    def test_nvfp4_masked_grouped_moe_silu_and_dual_gemm(self):
        from flashinfer import scaled_fp4_grouped_quantize

        for m_max, mm_vals in [(256, [80, 176, 64, 200]), (512, [160, 352, 0, 480])]:
            with self.subTest(m_max=m_max):
                torch.manual_seed(1234 + m_max)
                G, k, n = 4, 512, 256
                x = (
                    torch.randn((G, m_max, k), device="cuda", dtype=torch.bfloat16)
                    * 0.25
                )
                masked_m = torch.tensor(mm_vals, device="cuda", dtype=torch.int32)
                in_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
                a2_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
                w1_alpha = torch.ones((G,), device="cuda", dtype=torch.float32) * 0.125
                w2_alpha = torch.ones((G,), device="cuda", dtype=torch.float32) * 0.125

                w1_bf16 = (
                    torch.randn((G, 2 * n, k), device="cuda", dtype=torch.bfloat16)
                    * 0.25
                )
                w2_bf16 = (
                    torch.randn((G, k, n), device="cuda", dtype=torch.bfloat16) * 0.25
                )
                w1_q, w1_sf = scaled_fp4_grouped_quantize(
                    w1_bf16,
                    torch.full((G,), 2 * n, device="cuda", dtype=torch.int32),
                    torch.ones((G,), device="cuda", dtype=torch.float32),
                )
                w2_q, w2_sf = scaled_fp4_grouped_quantize(
                    w2_bf16,
                    torch.full((G,), k, device="cuda", dtype=torch.int32),
                    torch.ones((G,), device="cuda", dtype=torch.float32),
                )
                w1 = w1_q.permute(2, 0, 1).contiguous()
                w2 = w2_q.permute(2, 0, 1).contiguous()

                ref = _ref_flashinfer_moe_masked(
                    x,
                    in_gs,
                    w1,
                    w1_sf,
                    w1_alpha,
                    w2,
                    a2_gs,
                    w2_sf,
                    w2_alpha,
                    masked_m,
                )
                out_fast = nvfp4_masked_grouped_moe(
                    (x, None),
                    in_gs,
                    w1,
                    w1_sf,
                    w1_alpha,
                    w2,
                    a2_gs,
                    w2_sf,
                    w2_alpha,
                    masked_m,
                    activation="silu",
                    use_fused_dual_gemm=False,
                )
                out_dual = nvfp4_masked_grouped_moe(
                    (x, None),
                    in_gs,
                    w1,
                    w1_sf,
                    w1_alpha,
                    w2,
                    a2_gs,
                    w2_sf,
                    w2_alpha,
                    masked_m,
                    activation="silu",
                )

                for g in range(G):
                    mg = int(masked_m[g].item())
                    if mg > 0:
                        torch.testing.assert_close(
                            out_fast[g, :mg], ref[g, :mg], rtol=1e-3, atol=1e-3
                        )
                        torch.testing.assert_close(
                            out_dual[g, :mg], ref[g, :mg], rtol=1e-3, atol=1e-3
                        )

                # Verify CUDA Graph capture + replay and all-zero masked_m safety
                stream = torch.cuda.Stream()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    nvfp4_masked_grouped_moe(
                        (x, None),
                        in_gs,
                        w1,
                        w1_sf,
                        w1_alpha,
                        w2,
                        a2_gs,
                        w2_sf,
                        w2_alpha,
                        masked_m,
                    )
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    out_graph = nvfp4_masked_grouped_moe(
                        (x, None),
                        in_gs,
                        w1,
                        w1_sf,
                        w1_alpha,
                        w2,
                        a2_gs,
                        w2_sf,
                        w2_alpha,
                        masked_m,
                    )
                graph.replay()
                stream.synchronize()
                for g in range(G):
                    mg = int(masked_m[g].item())
                    if mg > 0:
                        torch.testing.assert_close(
                            out_graph[g, :mg], ref[g, :mg], rtol=1e-3, atol=1e-3
                        )

                zero_mm = torch.zeros((G,), device="cuda", dtype=torch.int32)
                out_zero_mm = nvfp4_masked_grouped_moe(
                    (x, None),
                    in_gs,
                    w1,
                    w1_sf,
                    w1_alpha,
                    w2,
                    a2_gs,
                    w2_sf,
                    w2_alpha,
                    zero_mm,
                )
                self.assertEqual(out_zero_mm.shape, (G, m_max, k))

    def test_nvfp4_routed_grouped_moe(self):
        from flashinfer import scaled_fp4_grouped_quantize

        torch.manual_seed(4321)
        M, K, N, G, topk = 64, 512, 256, 4, 2
        hidden_states = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        logits = torch.randn((M, G), device="cuda", dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(logits, dim=-1), k=topk, dim=-1
        )
        topk_ids = topk_ids.to(torch.int32)

        in_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
        w1_alpha = torch.ones((G,), device="cuda", dtype=torch.float32) * 0.125
        w2_alpha = torch.ones((G,), device="cuda", dtype=torch.float32) * 0.125

        w1_bf16 = torch.randn((G, 2 * N, K), device="cuda", dtype=torch.bfloat16)
        w2_bf16 = torch.randn((G, K, N), device="cuda", dtype=torch.bfloat16)
        w1_q, w1_sf = scaled_fp4_grouped_quantize(
            w1_bf16,
            torch.full((G,), 2 * N, device="cuda", dtype=torch.int32),
            torch.ones((G,), device="cuda", dtype=torch.float32),
        )
        w2_q, w2_sf = scaled_fp4_grouped_quantize(
            w2_bf16,
            torch.full((G,), K, device="cuda", dtype=torch.int32),
            torch.ones((G,), device="cuda", dtype=torch.float32),
        )
        w1 = w1_q.permute(2, 0, 1)
        w2 = w2_q.permute(2, 0, 1)

        out = nvfp4_routed_grouped_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            in_gs,
            w1,
            w1_sf,
            w1_alpha,
            w2,
            a2_gs,
            w2_sf,
            w2_alpha,
        )

        # Reference via per-expert full-batch evaluation
        expert_x = hidden_states.unsqueeze(0).expand(G, M, K).contiguous()
        full_m = torch.full((G,), M, device="cuda", dtype=torch.int32)
        expert_all = _ref_flashinfer_moe_masked(
            expert_x,
            in_gs,
            w1,
            w1_sf,
            w1_alpha,
            w2,
            a2_gs,
            w2_sf,
            w2_alpha,
            full_m,
        )
        ref_out = torch.zeros((M, K), device="cuda", dtype=torch.float32)
        for m in range(M):
            for j in range(topk):
                g_id = int(topk_ids[m, j].item())
                w_val = float(topk_weights[m, j].item())
                ref_out[m] += w_val * expert_all[g_id, m].float()
        ref_out = ref_out.to(torch.bfloat16)

        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)

        # Edge cases: empty batch (M=0) and EP out-of-range / negative topk_ids
        empty_out = nvfp4_routed_grouped_moe(
            hidden_states[:0],
            topk_ids[:0],
            topk_weights[:0],
            in_gs,
            w1,
            w1_sf,
            w1_alpha,
            w2,
            a2_gs,
            w2_sf,
            w2_alpha,
        )
        self.assertEqual(empty_out.shape, (0, K))

        ep_topk_ids = topk_ids.clone()
        ep_topk_ids[::2, 1] = -1
        ep_topk_ids[1::2, 1] = G + 2
        out_ep = nvfp4_routed_grouped_moe(
            hidden_states,
            ep_topk_ids,
            topk_weights,
            in_gs,
            w1,
            w1_sf,
            w1_alpha,
            w2,
            a2_gs,
            w2_sf,
            w2_alpha,
        )
        ref_ep = torch.zeros((M, K), device="cuda", dtype=torch.float32)
        for m in range(M):
            for j in range(topk):
                g_id = int(ep_topk_ids[m, j].item())
                if 0 <= g_id < G:
                    w_val = float(topk_weights[m, j].item())
                    ref_ep[m] += w_val * expert_all[g_id, m].float()
        torch.testing.assert_close(
            out_ep, ref_ep.to(torch.bfloat16), rtol=1e-2, atol=1e-2
        )

        # Verify consecutive calls do not alias/mutate previously returned output tensors
        out_second = nvfp4_routed_grouped_moe(
            hidden_states * 2.0,
            topk_ids,
            topk_weights,
            in_gs,
            w1,
            w1_sf,
            w1_alpha,
            w2,
            a2_gs,
            w2_sf,
            w2_alpha,
        )
        self.assertNotEqual(out.data_ptr(), out_second.data_ptr())
        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)

    def test_prequantized_fp4_non_aliasing_and_sglang_layer_wrapper(self):
        import importlib.util
        import os

        from flashinfer import scaled_fp4_grouped_quantize

        import sglang

        try:
            from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
                flashinfer_cutedsl_moe_masked,
            )
        except ImportError:
            mod_path = os.path.join(
                os.path.dirname(sglang.__file__),
                "srt",
                "layers",
                "moe",
                "flashinfer_cutedsl_moe.py",
            )
            if not os.path.exists(mod_path):
                self.skipTest("sglang.srt not installed in standalone kernels build")
            spec = importlib.util.spec_from_file_location(
                "_flashinfer_cutedsl_moe", mod_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            flashinfer_cutedsl_moe_masked = mod.flashinfer_cutedsl_moe_masked

        torch.manual_seed(9876)
        G, m_max, k, n = 8, 64, 512, 256
        x1 = torch.randn((G, m_max, k), device="cuda", dtype=torch.bfloat16) * 0.25
        x2 = torch.randn((G, m_max, k), device="cuda", dtype=torch.bfloat16) * 0.5
        mm = torch.tensor(
            [16, 64, 0, 33, 8, 49, 1, 64], device="cuda", dtype=torch.int32
        )
        in_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
        w1_alpha = torch.full((G,), 0.125, device="cuda", dtype=torch.float32)
        w2_alpha = torch.full((G,), 0.125, device="cuda", dtype=torch.float32)

        w1_bf16 = torch.randn((G, 2 * n, k), device="cuda", dtype=torch.bfloat16) * 0.25
        w2_bf16 = torch.randn((G, k, n), device="cuda", dtype=torch.bfloat16) * 0.25
        w1_q, w1_sf = scaled_fp4_grouped_quantize(
            w1_bf16,
            torch.full((G,), 2 * n, device="cuda", dtype=torch.int32),
            torch.ones((G,), device="cuda", dtype=torch.float32),
        )
        w2_q, w2_sf = scaled_fp4_grouped_quantize(
            w2_bf16,
            torch.full((G,), k, device="cuda", dtype=torch.int32),
            torch.ones((G,), device="cuda", dtype=torch.float32),
        )
        w1 = w1_q.permute(2, 0, 1).contiguous()
        w2 = w2_q.permute(2, 0, 1).contiguous()

        # 1. Back-to-back calls must return distinct tensors and zero-input must yield exact 0.0
        y1 = flashinfer_cutedsl_moe_masked(
            (x1, None), in_gs, w1, w1_sf, w1_alpha, w2, a2_gs, w2_sf, w2_alpha, mm
        )
        y1_copy = y1.clone()
        y2 = flashinfer_cutedsl_moe_masked(
            (x2, None), in_gs, w1, w1_sf, w1_alpha, w2, a2_gs, w2_sf, w2_alpha, mm
        )
        self.assertNotEqual(y1.data_ptr(), y2.data_ptr())
        self.assertTrue(torch.equal(y1, y1_copy))

        y_zero = nvfp4_masked_grouped_moe(
            (torch.zeros_like(x1), None),
            in_gs,
            w1,
            w1_sf,
            w1_alpha,
            w2,
            a2_gs,
            w2_sf,
            w2_alpha,
            mm,
        )
        for g in range(G):
            mg = int(mm[g].item())
            if mg > 0:
                self.assertEqual(y_zero[g, :mg].abs().max().item(), 0.0)

        # 2. Pre-quantized FP4 dispatch (a_q, a_q_sf) with input_global_scale=None vs SGLANG_USE_CUTEDSL_NVFP4_GROUPED_MOE=0 reference
        a_q, a_q_sf = scaled_fp4_grouped_quantize(x1, mm, in_gs)
        old_env = os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_GROUPED_MOE")
        try:
            os.environ["SGLANG_USE_CUTEDSL_NVFP4_GROUPED_MOE"] = "0"
            ref_fp4 = flashinfer_cutedsl_moe_masked(
                (a_q, a_q_sf), None, w1, w1_sf, w1_alpha, w2, a2_gs, w2_sf, w2_alpha, mm
            )
            os.environ["SGLANG_USE_CUTEDSL_NVFP4_GROUPED_MOE"] = "1"
            out_fp4 = flashinfer_cutedsl_moe_masked(
                (a_q, a_q_sf), None, w1, w1_sf, w1_alpha, w2, a2_gs, w2_sf, w2_alpha, mm
            )
        finally:
            if old_env is None:
                os.environ.pop("SGLANG_USE_CUTEDSL_NVFP4_GROUPED_MOE", None)
            else:
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_GROUPED_MOE"] = old_env

        for g in range(G):
            mg = int(mm[g].item())
            if mg > 0:
                torch.testing.assert_close(
                    out_fp4[g, :mg], ref_fp4[g, :mg], rtol=1e-3, atol=1e-3
                )


if __name__ == "__main__":
    unittest.main()
