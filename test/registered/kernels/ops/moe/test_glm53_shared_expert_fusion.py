"""gfx950 component checks for GLM-5.3 shared-expert AITER fusion."""

import unittest

import torch
import torch.nn.functional as F
from einops import rearrange

import aiter
from aiter import dtypes, pertoken_quant
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

# backend-specific: exercises AITER's gfx950 block-FP8 fused-MoE path.
register_amd_ci(est_time=100, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(),
    "GLM-5.3 shared-expert fusion is validated only on gfx950",
)
class TestGlm53SharedExpertFusion(CustomTestCase):
    NUM_ROUTED_EXPERTS = 288
    NUM_FUSED_EXPERTS = 289
    HIDDEN_SIZE = 4096
    BLOCK_SIZE = 128
    ROUTED_TOPK = 8
    TOKEN_BUCKETS = (1, 8, 128, 2048, 8192, 16384)

    def _quantize_weight(self, weight):
        experts, out_features, in_features = weight.shape
        blocks = rearrange(
            weight.view(
                experts,
                out_features // self.BLOCK_SIZE,
                self.BLOCK_SIZE,
                in_features // self.BLOCK_SIZE,
                self.BLOCK_SIZE,
            ),
            "e nb_n b_n nb_k b_k -> e nb_n nb_k (b_n b_k)",
        ).contiguous()
        quantized, scale = pertoken_quant(blocks, quant_dtype=dtypes.fp8)
        quantized = rearrange(
            quantized.view(
                experts,
                out_features // self.BLOCK_SIZE,
                in_features // self.BLOCK_SIZE,
                self.BLOCK_SIZE,
                self.BLOCK_SIZE,
            ),
            "e nb_n nb_k b_n b_k -> e (nb_n b_n) (nb_k b_k)",
        ).contiguous()
        return (
            shuffle_weight(quantized, (16, 16)),
            scale.view(
                experts,
                out_features // self.BLOCK_SIZE,
                in_features // self.BLOCK_SIZE,
            ).contiguous(),
        )

    def _make_weights(self, tp_size):
        intermediate_size = 2048 // tp_size
        torch.manual_seed(20260921 + tp_size)
        gate_up = (
            torch.randn(
                self.NUM_FUSED_EXPERTS,
                intermediate_size * 2,
                self.HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        down = (
            torch.randn(
                self.NUM_FUSED_EXPERTS,
                self.HIDDEN_SIZE,
                intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        gate_up_q, gate_up_scale = self._quantize_weight(gate_up)
        down_q, down_scale = self._quantize_weight(down)
        del gate_up, down
        return gate_up_q, down_q, gate_up_scale, down_scale

    def _fmoe(self, x_q, x_scale, ids, weights, w1, w2, w1_scale, w2_scale):
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid, output = moe_sorting(
            ids,
            weights,
            w1.shape[0],
            self.HIDDEN_SIZE,
            torch.bfloat16,
        )
        aiter.fmoe_fp8_blockscale_g1u1(
            output,
            x_q,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid,
            ids.shape[1],
            x_scale.t().contiguous(),
            w1_scale,
            w2_scale,
            "",
            self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            None,
        )
        return output

    def _inputs(self, tokens, tp_size):
        torch.manual_seed(1000 + tokens + tp_size)
        hidden = (
            torch.randn(
                tokens,
                self.HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        scores = torch.randn(
            tokens,
            self.NUM_ROUTED_EXPERTS,
            device="cuda",
            dtype=torch.bfloat16,
        )
        routed_weights, routed_ids = fused_topk(
            hidden, scores, self.ROUTED_TOPK, True
        )
        hidden_q, hidden_scale = pertoken_quant(
            hidden.view(tokens, self.HIDDEN_SIZE // self.BLOCK_SIZE, self.BLOCK_SIZE),
            quant_dtype=dtypes.fp8,
        )
        return (
            hidden_q.view(tokens, self.HIDDEN_SIZE),
            hidden_scale.squeeze(-1),
            routed_ids,
            routed_weights,
        )

    def _separate_and_fused(self, inputs, weights):
        hidden_q, hidden_scale, routed_ids, routed_weights = inputs
        w1, w2, w1_scale, w2_scale = weights
        routed = self._fmoe(
            hidden_q,
            hidden_scale,
            routed_ids,
            routed_weights,
            w1[:-1],
            w2[:-1],
            w1_scale[:-1],
            w2_scale[:-1],
        )
        gate_up = aiter.gemm_a8w8_blockscale_bpreshuffle(
            hidden_q,
            w1[-1],
            hidden_scale.t().contiguous(),
            w1_scale[-1],
            torch.bfloat16,
        )
        activated_q, activated_scale = fused_clamp_act_mul(
            gate_up,
            dtype_quant=dtypes.fp8,
            transpose_scale=True,
            quant_block_size=self.BLOCK_SIZE,
        )
        shared = aiter.gemm_a8w8_blockscale_bpreshuffle(
            activated_q,
            w2[-1],
            activated_scale,
            w2_scale[-1],
            torch.bfloat16,
        )
        separate = routed + shared
        fused_ids = torch.cat(
            (
                routed_ids,
                torch.full(
                    (routed_ids.shape[0], 1),
                    self.NUM_ROUTED_EXPERTS,
                    device="cuda",
                    dtype=routed_ids.dtype,
                ),
            ),
            dim=1,
        )
        fused_weights = torch.cat(
            (
                routed_weights,
                torch.ones(
                    (routed_weights.shape[0], 1),
                    device="cuda",
                    dtype=routed_weights.dtype,
                ),
            ),
            dim=1,
        )
        fused = self._fmoe(
            hidden_q,
            hidden_scale,
            fused_ids,
            fused_weights,
            w1,
            w2,
            w1_scale,
            w2_scale,
        )
        return separate, fused

    def _assert_matches_separate_path(self, separate, fused):
        self.assertTrue(torch.isfinite(fused).all())
        cosine = F.cosine_similarity(
            fused.float().flatten(),
            separate.float().flatten(),
            dim=0,
        ).item()
        self.assertGreaterEqual(cosine, 0.999)
        torch.testing.assert_close(fused, separate, rtol=0.05, atol=0.02)

    @torch.inference_mode()
    def test_tp4_tp8_retained_buckets_match_separate_reference(self):
        """A wrong appended ID, non-unit shared weight, or incompatible weight
        layout silently changes model output while every operator still runs."""
        for tp_size in (4, 8):
            weights = self._make_weights(tp_size)
            for tokens in self.TOKEN_BUCKETS:
                with self.subTest(tp_size=tp_size, tokens=tokens):
                    separate, fused = self._separate_and_fused(
                        self._inputs(tokens, tp_size), weights
                    )
                    self._assert_matches_separate_path(separate, fused)
            del weights
            empty_gpu_cache()

    @torch.inference_mode()
    def test_fused_operator_replays_under_cuda_graph(self):
        """The serving path captures decode; eager-only correctness would miss
        graph-unsafe allocations or stale shared routing tensors."""
        for tp_size in (4, 8):
            weights = self._make_weights(tp_size)
            inputs = self._inputs(8, tp_size)
            _, eager = self._separate_and_fused(inputs, weights)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _, captured = self._separate_and_fused(inputs, weights)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, eager, rtol=0.05, atol=0.02)
            del graph, captured, eager, inputs, weights
            empty_gpu_cache()


if __name__ == "__main__":
    unittest.main()
