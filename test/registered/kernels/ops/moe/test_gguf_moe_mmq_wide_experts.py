"""GGUF batched MMQ MoE must not drop experts whose index is above 255."""

import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.layers.quantization.gguf import fused_moe_gguf
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="nightly", runner_config="1-gpu-large")

Q4_K = 12
Q4_K_BLOCK = 256
Q4_K_BYTES = 144
# Wider than the 255 cap the MoE kernel inherited from vLLM, so a prefill-sized
# batch necessarily routes tokens to expert ids that the old guard rejected.
# 512 also matches Qwen3-Next, the model that first hit this.
NUM_EXPERTS = 512
HIDDEN = 256
FFN = 256
TOKENS = 128  # > 64 selects the batched MMQ path in fused_moe_gguf
TOP_K = 2
SEED = 0


def _packed_weight(out_features: int, in_features: int) -> torch.Tensor:
    """Packed Q4_K bytes: random payload with finite fp16 scales.

    Built directly rather than through ``gguf.quants`` so the test does not
    depend on the gguf-py quantizer. Both kernels consume the same bytes, which
    is all this comparison needs.
    """
    nb = in_features // Q4_K_BLOCK
    weight = torch.randint(
        0, 256, (NUM_EXPERTS, out_features, nb, Q4_K_BYTES), dtype=torch.uint8
    )
    for offset in (0, 2):
        scales = (torch.rand(NUM_EXPERTS, out_features, nb) * 1e-3 + 5e-4).half()
        weight[..., offset : offset + 2] = scales.view(torch.uint8).reshape(
            NUM_EXPERTS, out_features, nb, 2
        )
    return weight.reshape(NUM_EXPERTS, out_features, -1).to(torch.int8).cuda()


def _vec_reference(x, w1, w2, topk_ids, topk_weights):
    """Same routing, evaluated with the per-slot kernel that has no expert cap."""
    from sgl_kernel.quantization import ggml_moe_a8_vec

    hidden = ggml_moe_a8_vec(x, w1, topk_ids, TOP_K, Q4_K, 2 * FFN, TOKENS)
    hidden = torch.nn.functional.silu(hidden[:, :FFN]) * hidden[:, FFN:]
    out = ggml_moe_a8_vec(hidden, w2, topk_ids, 1, Q4_K, HIDDEN, TOKENS * TOP_K)
    out = out.view(TOKENS, TOP_K, HIDDEN) * topk_weights.unsqueeze(-1)
    return out.sum(dim=1)


class TestGgufMoeMmqWideExperts(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-only GGUF MoE kernels")
    @torch.inference_mode()
    def test_batched_mmq_matches_vec_for_experts_above_255(self):
        w1 = _packed_weight(2 * FFN, HIDDEN)
        w2 = _packed_weight(HIDDEN, FFN)

        generator = torch.Generator(device="cuda").manual_seed(SEED)
        x = torch.randn(TOKENS, HIDDEN, generator=generator, device="cuda")
        topk_ids = torch.randint(
            0, NUM_EXPERTS, (TOKENS, TOP_K), generator=generator, device="cuda"
        ).to(torch.int32)
        topk_weights = torch.rand(TOKENS, TOP_K, generator=generator, device="cuda")

        # Guard the premise: the routing table has to exercise the wide range,
        # otherwise this test would pass even with the expert cap in place.
        self.assertGreater(int((topk_ids >= 256).sum()), 0)
        self.assertGreater(int((topk_ids < 256).sum()), 0)

        actual = fused_moe_gguf(x, w1, w2, topk_weights, topk_ids, Q4_K, Q4_K, "silu")
        expected = _vec_reference(x, w1, w2, topk_ids, topk_weights)

        self.assertTrue(bool(torch.isfinite(actual).all()))
        # Both paths run the same quantised weights through the same q8_1 input
        # quantisation, so they only differ by accumulation order. Measure the
        # error relative to the output scale: the dropped experts show up as
        # ~1e-1 relative L2 error (or NaN), the fixed kernel stays below 1e-2.
        rel_l2 = ((actual - expected).norm() / expected.norm()).item()
        self.assertLess(rel_l2, 2e-2, f"relative L2 error {rel_l2:.4f}")


if __name__ == "__main__":
    unittest.main()
