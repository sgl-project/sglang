"""The gfx950 Kimi-K3 fused router top-k + MoE route sort against aiter's chain."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the route sort hands aiter's gfx950 A8W4 MoE its routes.
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8: 896 experts, top-16, latent 3584, hidden 7168, up to 8 bank rows.
E, K, LATENT, H, NB = 896, 16, 3584, 7168, 9
BLOCK_M = 32


def _routes(out, num_tokens):
    """{(token, slot): (row, weight)} over the valid rows; padding rows carry weight 0."""
    pad = (K << 24) | num_tokens
    ids, weights = out.sorted_ids.cpu(), out.sorted_weights.cpu()
    routes = {}
    for row in range(int(out.num_valid_ids[0])):
        v = int(ids[row])
        if v == pad:
            assert float(weights[row]) == 0.0
            continue
        routes[(v & 0xFFFFFF, v >> 24)] = (row, float(weights[row]))
    return routes


@unittest.skipUnless(is_gfx95_supported(), "gfx950 route sort")
class TestKimiK3RouteSort(CustomTestCase):
    @torch.inference_mode()
    def test_matches_aiter_topk_sort_and_quant(self):
        """On router logits strided inside the fused front's output, one launch gives
        aiter's biased top-k and moe_sorting (ids, sorted rows, expert blocks, valid
        counts), a zeroed moe_buf and fused_dynamic_mxfp8_quant_moe_sort's
        activations; a NaN or saturated logits row still yields 16 distinct experts."""
        import aiter
        from aiter import biased_grouped_topk
        from aiter.fused_moe import moe_sorting

        from sglang.kernels.ops.moe.kimi_k3_route_sort import route_sort

        g = torch.Generator(device="cuda").manual_seed(2)
        bias = (torch.randn(E, generator=g, device="cuda") * 0.05).bfloat16()
        for num_tokens in (1, 4, 16):
            front = torch.randn(
                num_tokens, 1536 + E + LATENT, generator=g, device="cuda"
            )
            front = (front * 2).bfloat16()
            logits, routed = front[:, 1536 : 1536 + E], front[:, 1536 + E :]
            want_w = torch.empty(num_tokens, K, dtype=torch.float32, device="cuda")
            want_ids = torch.empty(num_tokens, K, dtype=torch.int32, device="cuda")
            biased_grouped_topk(logits, bias, want_w, want_ids, 1, 1, True, 1.0)
            sorted_ids, _, sorted_experts, num_valid, _ = moe_sorting(
                want_ids, want_w, E, LATENT, torch.bfloat16, BLOCK_M
            )
            buf = torch.full((num_tokens, LATENT), 5.0, device="cuda").bfloat16()
            got = route_sort(
                logits, bias, topk=K, block_m=BLOCK_M, model_dim=LATENT, moe_buf=buf,
                routed_scaling_factor=1.0, renormalize=True, quant_input=routed,
            )  # fmt: skip
            want_a1, _ = aiter.fused_dynamic_mxfp8_quant_moe_sort(
                routed, sorted_ids, num_valid, num_tokens, K, BLOCK_M
            )
            rows = int(num_valid[0])
            with self.subTest(num_tokens=num_tokens):
                self.assertTrue(torch.equal(got.topk_ids, want_ids))
                torch.testing.assert_close(
                    got.topk_weights, want_w, rtol=1e-6, atol=1e-7
                )
                self.assertTrue(torch.equal(got.num_valid_ids, num_valid))
                self.assertTrue(torch.equal(got.sorted_ids[:rows], sorted_ids[:rows]))
                self.assertTrue(
                    torch.equal(
                        got.sorted_expert_ids[: rows // BLOCK_M],
                        sorted_experts[: rows // BLOCK_M],
                    )
                )
                self.assertTrue(bool((buf == 0).all()))
                self.assertTrue(
                    torch.equal(got.a1.view(torch.uint8), want_a1.view(torch.uint8))
                )
            bad = front.clone()
            bad[0, 1536 : 1536 + E] = float("nan")
            bad[-1, 1536 : 1536 + E] = -1e30
            ids = route_sort(
                bad[:, 1536 : 1536 + E], bias, topk=K, block_m=BLOCK_M,
                model_dim=LATENT, moe_buf=None, routed_scaling_factor=1.0,
                renormalize=True,
            ).topk_ids  # fmt: skip
            with self.subTest(num_tokens=num_tokens, degenerate=True):
                self.assertTrue(bool(((ids >= 0) & (ids < E)).all()))
                self.assertTrue(all(len(set(row)) == K for row in ids.tolist()))


def _row_scales(scales: torch.Tensor, row: int) -> torch.Tensor:
    """The e8m0 group scales of a sorted row in aiter's shuffled layout."""
    n_pad = scales.shape[1]
    flat = scales.flatten()
    idx = [
        (row // 32 * n_pad) * 32
        + (g // 8) * 256
        + (g % 4) * 64
        + (row % 16) * 4
        + (g % 8) // 4 * 2
        + (row % 32) // 16
        for g in range(LATENT // 32)
    ]
    return flat[idx]


if __name__ == "__main__":
    unittest.main()
