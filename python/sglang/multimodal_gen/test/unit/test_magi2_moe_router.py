# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.moe_multihead import Magi2MultiHeadRouter


def _make_router(
    *,
    num_heads,
    num_experts,
    head_dim,
    top_k,
    route_scale=1.0,
    route_norm=True,
    seed=0,
):
    torch.manual_seed(seed)
    router = Magi2MultiHeadRouter(
        num_heads=num_heads,
        num_experts=num_experts,
        head_dim=head_dim,
        top_k=top_k,
        route_scale=route_scale,
        route_norm=route_norm,
    )
    with torch.no_grad():
        router.gate.copy_(torch.randn(num_heads * num_experts, head_dim))
    return router.requires_grad_(False)


def _unbiased_scores(router, tokens):
    """Reference per-head sigmoid scores, ``[tokens, heads, experts]``."""
    gate = router.gate.detach().view(
        router.num_heads, router.num_experts, router.head_dim
    )
    return torch.einsum("thd,hed->the", tokens.float(), gate.float()).sigmoid()


class TestMagi2MoERouterSelection(unittest.TestCase):
    def test_each_head_selects_only_from_its_own_expert_bank(self):
        heads, experts, top_k = 4, 8, 2
        router = _make_router(
            num_heads=heads, num_experts=experts, head_dim=16, top_k=top_k
        )
        ids = router(torch.randn(6, heads, 16))[0].view(6, heads, top_k)

        for head in range(heads):
            low, high = head * experts, (head + 1) * experts
            self.assertTrue(((ids[:, head] >= low) & (ids[:, head] < high)).all())
        for row in ids.reshape(-1, top_k):
            self.assertEqual(len(set(row.tolist())), top_k)

    def test_flattened_rows_are_token_major(self):
        num_tokens, heads, experts, top_k = 5, 3, 8, 2
        router = _make_router(
            num_heads=heads, num_experts=experts, head_dim=16, top_k=top_k
        )
        tokens = torch.randn(num_tokens, heads, 16)
        ids, weights = router(tokens)

        scores = _unbiased_scores(router, tokens)
        local = scores.topk(k=top_k, dim=-1).indices
        head_offset = torch.arange(heads).view(1, heads, 1) * experts

        got = ids.view(num_tokens, heads, top_k).sort(dim=-1)
        want = (local + head_offset).sort(dim=-1)
        self.assertTrue(torch.equal(got.values, want.values))

        got_weights = weights.view(num_tokens, heads, top_k).gather(-1, got.indices)
        want_weights = F.normalize(scores.gather(-1, local), p=1, dim=-1).gather(
            -1, want.indices
        )
        self.assertTrue(torch.allclose(got_weights, want_weights, atol=1e-5))


class TestMagi2MoERouterWeights(unittest.TestCase):
    def test_expert_bias_steers_selection_but_never_reaches_the_weights(self):
        heads, experts, top_k = 2, 8, 2
        router = _make_router(
            num_heads=heads,
            num_experts=experts,
            head_dim=16,
            top_k=top_k,
            route_norm=False,
        )
        tokens = torch.randn(6, heads, 16)
        scores = _unbiased_scores(router, tokens)
        base_ids = router(tokens)[0]

        target = scores.sum(dim=0).argmin(dim=-1)
        promoted = target + torch.arange(heads) * experts
        with torch.no_grad():
            router.expert_bias[promoted] = 10.0
        ids, weights = router(tokens)

        self.assertFalse(torch.equal(ids, base_ids))
        hit = ids.view(-1, heads, top_k) == promoted.view(1, heads, 1)
        self.assertTrue(hit.any(dim=-1).all())

        # A leaked +10 bias would put these weights near 10.7; sigmoid caps at 1.
        self.assertLessEqual(float(weights.max()), 1.0)
        want = scores[:, torch.arange(heads), target].unsqueeze(-1).expand_as(hit)
        self.assertTrue(
            torch.allclose(weights.view(-1, heads, top_k)[hit], want[hit], atol=1e-5)
        )

    def test_route_scale_is_applied_exactly_once(self):
        scale = 2.5
        shape = dict(num_heads=2, num_experts=8, head_dim=16, top_k=4)
        scaled = _make_router(**shape, route_scale=scale)
        unit = _make_router(**shape, route_scale=1.0)
        tokens = torch.randn(6, 2, 16)

        scaled_ids, scaled_weights = scaled(tokens)
        unit_ids, unit_weights = unit(tokens)
        totals = scaled_weights.sum(-1)

        self.assertTrue(torch.equal(scaled_ids, unit_ids))
        self.assertTrue(
            torch.allclose(totals, torch.full_like(totals, scale), atol=1e-5)
        )
        self.assertTrue(torch.allclose(scaled_weights, unit_weights * scale, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
