import unittest

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.version.hip
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    "gfx950 (MI35x) only",
)
class TestSmallMRouterGfx950(CustomTestCase):
    def _ints(self, *shape, bound):
        return torch.randint(-bound, bound + 1, shape, device="cuda").bfloat16()

    def test_router(self):
        from sglang.kernels.ops.moe import smallm_router_gfx950 as R
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            fused_append_shared_experts_with_weights,
        )
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        torch.manual_seed(0)
        # multiples of 2^-6 times small ints: every dot product is exact in fp32, so the gate GEMM rounding
        # is order-independent and the bf16 logits carry many exact ties (lowest expert id must win)
        wg, ws = self._ints(512, 4096, bound=1) / 64, self._ints(1, 4096, bound=1) / 64
        for tok in (1, 3, 4, 8):
            x = self._ints(tok, 4096, bound=2)
            self.assertTrue(R.smallm_router_supported(x))
            w, ids = R.smallm_router(x, wg, ws, 0.5)
            logits = (x.double() @ wg.double().T).bfloat16()
            ref_w, ref_ids = moe_fused_gate(logits, None, 10, scoring_func="softmax")
            ref_ids, ref_w = fused_append_shared_experts_with_weights(
                ref_ids,
                ref_w,
                None,
                1,
                N=512,
                fuse_gate=True,
                hidden_states=x,
                gate_weight=ws,
                scale=0.5,
            )
            self.assertTrue(torch.equal(ids, ref_ids), f"tok={tok}")
            torch.testing.assert_close(w, ref_w, atol=1e-5, rtol=1e-5)

        for bad in (
            self._ints(9, 4096, bound=2),
            self._ints(41, 4096, bound=2),
            x.half(),
            x[:, :2048],
        ):
            self.assertFalse(R.smallm_router_supported(bad))
        with envs.SGLANG_ROCM_SMALLM_ROUTER.override(False):
            self.assertFalse(R.smallm_router_supported(x))

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gw, gids = R.smallm_router(x, wg, ws, 0.5)
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(gids, ids) and torch.equal(gw, w))

    def test_glm_router(self):
        from sglang.kernels.ops.moe import smallm_router_gfx950 as R

        torch.manual_seed(0)
        # the gate GEMM is exact in fp32, so only the bf16 round of the logits and the sigmoid are shared with the
        # reference; distinct fp32 biases in GLM-5.3-Flash's range keep the top-8 free of ties
        wg = self._ints(R.GLM_E, 4096, bound=1) / 64
        bias = 7.4 + 0.4 * torch.rand(R.GLM_E, device="cuda")
        rsf, shared_w = 2.5, 0.75
        for tok in (1, 2, 4, 6, 8, 12, 16):
            x = self._ints(tok, 4096, bound=2)
            self.assertTrue(R.smallm_router_supported(x, R.GLM_MAX_TOK))
            w, ids = R.smallm_router_glm(x, wg, bias, rsf, shared_w)
            scores = torch.sigmoid((x.double() @ wg.double().T).bfloat16().float())
            ref_ids = (scores + bias).topk(R.GLM_TOPK, dim=1).indices.sort(1).values
            ref_w = scores.gather(1, ref_ids)
            ref_w = ref_w / ref_w.sum(1, keepdim=True) * rsf

            routed = ids[:, : R.GLM_TOPK].long().sort(1)
            self.assertTrue(torch.equal(routed.values, ref_ids), f"tok={tok}")
            routed_w = w[:, : R.GLM_TOPK].gather(1, routed.indices)
            torch.testing.assert_close(routed_w, ref_w, atol=1e-5, rtol=1e-5)
            self.assertTrue(bool((ids[:, R.GLM_TOPK] == R.GLM_E).all()), f"tok={tok}")
            self.assertTrue(bool((w[:, R.GLM_TOPK] == shared_w).all()), f"tok={tok}")

        self.assertFalse(
            R.smallm_router_supported(self._ints(17, 4096, bound=2), R.GLM_MAX_TOK)
        )

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gw, gids = R.smallm_router_glm(x, wg, bias, rsf, shared_w)
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(gids, ids) and torch.equal(gw, w))

    def _glm_inputs(self, R, tok):
        torch.manual_seed(0)
        wg = self._ints(R.GLM_E, 4096, bound=1) / 64
        bias = 7.4 + 0.4 * torch.rand(R.GLM_E, device="cuda")
        return self._ints(tok, 4096, bound=2), wg, bias

    def _assert_padded_rows(self, R, w, ids, n_valid, shared_w):
        self.assertTrue(bool((ids[n_valid:, : R.GLM_TOPK] == 0).all()))
        self.assertTrue(bool((ids[n_valid:, R.GLM_TOPK] == R.GLM_E).all()))
        self.assertTrue(bool((w[n_valid:] == 0).all()))
        self.assertTrue(bool((w[:n_valid, R.GLM_TOPK] == shared_w).all()))

    def test_glm_router_padded_rows(self):
        from sglang.kernels.ops.moe import smallm_router_gfx950 as R

        rsf, shared_w = 2.5, 0.75
        for tok in (4, 8, 16):
            x, wg, bias = self._glm_inputs(R, tok)
            w, ids = R.smallm_router_glm(x, wg, bias, rsf, shared_w)
            for n_valid in (0, 1, tok - 1, tok):
                n = torch.tensor(n_valid, dtype=torch.int32, device="cuda")
                pw, pids = R.smallm_router_glm(
                    x, wg, bias, rsf, shared_w, num_token_non_padded=n
                )
                msg = f"tok={tok} n_valid={n_valid}"
                self.assertTrue(torch.equal(pids[:n_valid], ids[:n_valid]), msg)
                self.assertTrue(torch.equal(pw[:n_valid], w[:n_valid]), msg)
                self._assert_padded_rows(R, pw, pids, n_valid, shared_w)

        # the bound is read at replay, as cuda_graph_runner refills it in place for each batch
        n = torch.tensor(tok, dtype=torch.int32, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gw, gids = R.smallm_router_glm(
                x, wg, bias, rsf, shared_w, num_token_non_padded=n
            )
        for n_valid in (tok, 5, 1):
            n.fill_(n_valid)
            g.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(gids[:n_valid], ids[:n_valid]), n_valid)
            self.assertTrue(torch.equal(gw[:n_valid], w[:n_valid]), n_valid)
            self._assert_padded_rows(R, gw, gids, n_valid, shared_w)

    def test_glm_moe_topk_output_routing_inputs(self):
        from types import SimpleNamespace

        from sglang.kernels.ops.moe import smallm_router_gfx950 as R
        from sglang.srt.configs.shared_experts import SharedExpertsConfig
        from sglang.srt.layers.moe.topk import TopKConfig

        x, wg, bias = self._glm_inputs(R, 8)
        cfg = TopKConfig(
            top_k=R.GLM_TOPK + 1,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
            correction_bias=bias,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            fused_shared_experts_scaling_factor=0.75,
        )
        n = torch.tensor(5, dtype=torch.int32, device="cuda")
        routing = [(n, None)]
        moe = SimpleNamespace(
            _smallm_router_glm_ok=True,
            gate=SimpleNamespace(weight=wg),
            layer_id=0,
            topk=SimpleNamespace(topk_config=cfg, routing_inputs=lambda: routing[0]),
        )
        out = R.glm_moe_topk_output(moe, x)
        self.assertIsNotNone(out)
        self._assert_padded_rows(R, out.topk_weights, out.topk_ids, 5, 0.75)
        # expert-location dispatch (EPLB, non-trivial placement) needs the unfused remap
        routing[0] = (n, object())
        self.assertIsNone(R.glm_moe_topk_output(moe, x))

    def test_glm_router_matches_unfused_path(self):
        from sglang.kernels.ops.moe import smallm_router_gfx950 as R
        from sglang.srt.configs.shared_experts import SharedExpertsConfig
        from sglang.srt.layers.moe import topk as topk_mod

        if not topk_mod._use_aiter:
            self.skipTest("needs SGLANG_USE_AITER=1 (set by AMD CI)")
        # the token count crosses GLM_MAX_TOK here, so rows 0..15 must route the same on both sides of it: the
        # fused kernel at M = 16 and the aiter path that M = 17 falls back to, fp32 bias and padding included
        rsf, shared_w = 2.5, 0.75
        x, wg, bias = self._glm_inputs(R, R.GLM_MAX_TOK + 1)
        cfg = topk_mod.TopKConfig(
            top_k=R.GLM_TOPK + 1,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            renormalize=True,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
            correction_bias=bias,
            fp32_correction_bias=True,
            scoring_func="sigmoid",
            routed_scaling_factor=rsf,
            fused_shared_experts_scaling_factor=shared_w,
        )
        self.assertFalse(R.smallm_router_supported(x, R.GLM_MAX_TOK))
        logits = (x.double() @ wg.double().T).bfloat16()
        for n_valid in (None, R.GLM_MAX_TOK - 3):
            n = None
            if n_valid is not None:
                n = torch.tensor(n_valid, dtype=torch.int32, device="cuda")
            ref = topk_mod.select_experts(x, logits, cfg, num_token_non_padded=n)
            x16 = x[: R.GLM_MAX_TOK]
            self.assertTrue(R.smallm_router_supported(x16, R.GLM_MAX_TOK))
            w, ids = R.smallm_router_glm(
                x16, wg, bias, rsf, shared_w, num_token_non_padded=n
            )
            ref_ids = ref.topk_ids[: R.GLM_MAX_TOK].int()
            ref_w = ref.topk_weights[: R.GLM_MAX_TOK].float()
            # the routed slots are a set: order them by id on both sides
            got = ids[:, : R.GLM_TOPK].sort(1)
            exp = ref_ids[:, : R.GLM_TOPK].sort(1)
            msg = f"n_valid={n_valid}"
            self.assertTrue(torch.equal(got.values, exp.values), msg)
            self.assertTrue(
                torch.equal(ids[:, R.GLM_TOPK], ref_ids[:, R.GLM_TOPK]), msg
            )
            torch.testing.assert_close(
                w[:, : R.GLM_TOPK].gather(1, got.indices),
                ref_w[:, : R.GLM_TOPK].gather(1, exp.indices),
                atol=1e-5,
                rtol=1e-5,
                msg=msg,
            )
            torch.testing.assert_close(w[:, R.GLM_TOPK], ref_w[:, R.GLM_TOPK], msg=msg)

    def test_glm_router_ties(self):
        from sglang.kernels.ops.moe import smallm_router_gfx950 as R

        torch.manual_seed(0)
        # experts 3i..3i+2 share a gate row and a bias, so their s + bias tie exactly; groups of 3 put a tie
        # across the top-8 boundary, where the lowest expert id must win
        wg = (self._ints(R.GLM_E // 3, 4096, bound=1) / 64).repeat_interleave(3, dim=0)
        bias = 7.4 + 0.4 * torch.rand(R.GLM_E // 3, device="cuda")
        bias = bias.repeat_interleave(3)
        for tok in (1, 6, 16):
            x = self._ints(tok, 4096, bound=2)
            w, ids = R.smallm_router_glm(x, wg, bias, 2.5)
            scores = torch.sigmoid((x.double() @ wg.double().T).bfloat16().float())
            ref_ids = (scores + bias).sort(dim=1, descending=True, stable=True).indices
            self.assertTrue(
                torch.equal(ids[:, : R.GLM_TOPK].long(), ref_ids[:, : R.GLM_TOPK]),
                f"tok={tok}",
            )


if __name__ == "__main__":
    unittest.main()
