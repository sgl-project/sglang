import unittest

import torch

from sglang.kernels.ops.moe import kimi_k3_fused_routing
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOPK = 16
HIDDEN = 3584
SCALE = 2.827


def _routing_inputs(m: int, seed: int, shared: bool = False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(m, NUM_EXPERTS, generator=g, device="cuda")
    if shared:
        scores = scores[:1].expand(m, -1) + 1e-3 * scores
    bias = 0.1 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
    x = torch.randn(m, HIDDEN, generator=g, device="cuda")
    return scores, bias, x


def _check_metadata(test, ids, meta, first, num_local, tile):
    flat = ids.flatten().long()
    local = flat - first
    is_local = (flat >= 0) & (local >= 0) & (local < num_local)
    counts = torch.bincount(local[is_local], minlength=num_local)
    tiles = (counts + tile - 1) // tile
    tile_offset = torch.cumsum(tiles, 0) - tiles
    num_ctas = int(tiles.sum())
    test.assertEqual(int(meta.num_non_exiting_ctas.item()), num_ctas)
    test.assertEqual(int(meta.total_num_padded_tokens.item()), num_ctas * tile)

    experts = torch.arange(num_local, device="cuda")
    cta_expert = torch.repeat_interleave(experts, tiles)
    cta = torch.arange(num_ctas, device="cuda")
    mn_limit = torch.minimum(
        (cta + 1) * tile, (tile_offset * tile + counts)[cta_expert]
    )
    test.assertTrue(
        torch.equal(meta.cta_idx_xy_to_batch_idx[:num_ctas].long(), cta_expert)
    )
    test.assertTrue(
        torch.equal(meta.cta_idx_xy_to_mn_limit[:num_ctas].long(), mn_limit)
    )

    per_expert = torch.zeros(NUM_EXPERTS, dtype=torch.long, device="cuda")
    per_expert[first : first + num_local] = counts
    test.assertTrue(torch.equal(meta.num_tokens_per_expert.long(), per_expert))

    e2p = meta.expanded_idx_to_permuted_idx.long()
    test.assertTrue(bool((e2p[~is_local] == -1).all()))
    p = e2p[is_local]
    seg_start = (tile_offset * tile)[local[is_local]]
    test.assertTrue(
        bool(((p >= seg_start) & (p < seg_start + counts[local[is_local]])).all())
    )
    test.assertEqual(int(torch.unique(p).numel()), int(p.numel()))
    p2t = meta.permuted_idx_to_token_idx.long()
    token = torch.arange(flat.numel(), device="cuda")[is_local] // TOPK
    test.assertTrue(torch.equal(p2t[p], token))
    rows = torch.arange(num_ctas * tile, device="cuda")
    padding = rows >= mn_limit.repeat_interleave(tile)
    test.assertTrue(bool((p2t[: num_ctas * tile][padding] == -1).all()))


class TestKimiK3FusedRouting(CustomTestCase):
    def setUp(self):
        if not kimi_k3_fused_routing.available():
            self.skipTest("Kimi-K3 fused routing kernel unavailable")

    def _run(self, scores, bias, x, *, num_valid, first, num_local, tile, wdtype):
        return kimi_k3_fused_routing.fused_routing(
            scores,
            bias,
            x,
            TOPK,
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_scale=True,
            num_token_non_padded=num_valid,
            local_expert_offset=first,
            num_local_experts=num_local,
            tile_tokens=tile,
            weights_dtype=wdtype,
        )

    def test_matches_route_quant_and_metadata(self):
        cases = [
            (1, 1, 0, NUM_EXPERTS, 8, False),
            (6, 6, 0, 448, 8, False),
            (12, 10, 224, 224, 32, False),
            (12, 9, 0, 448, 8, True),
            (16, 16, 448, 448, 8, True),
            (24, 19, 0, 448, 16, False),
            (96, 90, 0, 448, 64, False),
            (256, 250, 0, NUM_EXPERTS, 192, False),
            (64, 60, 448, 448, 256, False),
        ]
        for m, valid, first, num_local, tile, shared in cases:
            with self.subTest(m=m, valid=valid, first=first, tile=tile, shared=shared):
                scores, bias, x = _routing_inputs(m, seed=m + tile, shared=shared)
                wdtype = torch.bfloat16 if m <= 64 else torch.float32
                num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
                out = self._run(
                    scores,
                    bias,
                    x,
                    num_valid=num_valid,
                    first=first,
                    num_local=num_local,
                    tile=tile,
                    wdtype=wdtype,
                )
                ref_w, ref_i = route_radix(scores, bias, TOPK, True, SCALE, True)
                ref_i[valid:] = -1
                ref_q, ref_s = per_token_group_quant(x, group_size=32, scale_ue8m0=True)
                self.assertTrue(torch.equal(out.ids, ref_i))
                self.assertTrue(
                    torch.equal(out.weights.view(torch.int32), ref_w.view(torch.int32))
                )
                self.assertTrue(
                    torch.equal(out.x_q.view(torch.uint8), ref_q.view(torch.uint8))
                )
                self.assertTrue(torch.equal(out.x_s, ref_s))
                self.assertTrue(
                    torch.equal(out.metadata.expert_weights, ref_w.to(wdtype))
                )
                _check_metadata(self, out.ids, out.metadata, first, num_local, tile)


if __name__ == "__main__":
    unittest.main()
