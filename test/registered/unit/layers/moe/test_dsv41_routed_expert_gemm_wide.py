"""DSV4.1 low-batch routed-expert GEMMs (--enable-dsv41-low-latency-decode).

CPU: the tile bound the route record is sized by holds for any routing, which calls
the pair admits, the deferred-finalize ABI it returns, and when the MoE method gets
it. GPU (SM100 family): at 1-48 rows, including experts chosen by more than 8 rows
(several tiles), the route + GEMM pair match FlashInfer's routed MoE, eager and
replayed from a CUDA graph.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models.dsv41_low_latency import routed_expert_gemm as reg
from sglang.srt.models.dsv41_low_latency.switch import ROUTED_EXPERT_MAX_ROWS
from sglang.test.test_utils import CustomTestCase

TARGET_EXPERTS, TARGET_TOP_K = 384, 6


def _routing(rows, distinct, generator, experts, top_k):
    distinct = min(max(distinct, top_k), experts)
    pool = torch.randperm(experts, generator=generator)[:distinct]
    ids = [
        pool[torch.randperm(distinct, generator=generator)[:top_k]] for _ in range(rows)
    ]
    return torch.stack(ids).to(torch.int32)


def _topk(ids, weights=None):
    if weights is None:
        weights = torch.full(ids.shape, 1.0 / ids.shape[1])
    return StandardTopKOutput(weights, ids, torch.zeros(ids.shape[0], 1))


class TestRouteRecordBound(CustomTestCase):
    def test_max_ctas_bounds_any_routing(self):
        for rows, top_k, experts in (
            (12, 6, 384),
            (48, 6, 384),
            (64, 8, 32),
            (40, 3, 128),
        ):
            generator = torch.Generator().manual_seed(rows * top_k)
            for distinct in (top_k, top_k + 1, 2 * top_k, 64, experts):
                with self.subTest(rows=rows, top_k=top_k, distinct=distinct):
                    ids = _routing(rows, distinct, generator, experts, top_k)
                    counts = torch.bincount(ids.flatten().long()).tolist()
                    tiles = sum(math.ceil(c / reg.TILE) for c in counts if c)
                    self.assertLessEqual(tiles, reg.max_ctas(rows, top_k, experts))


def _bare_gemm(max_rows=ROUTED_EXPERT_MAX_ROWS):
    gemm = object.__new__(reg.RoutedExpertGemm)
    gemm.max_rows, gemm.top_k = max_rows, TARGET_TOP_K
    return gemm


def _activation(rows):
    return (
        torch.zeros(rows, reg.HIDDEN, dtype=torch.uint8),
        torch.zeros(rows, reg.HIDDEN // 32, dtype=torch.uint8),
    )


class TestSupports(CustomTestCase):
    def test_admits_one_to_max_rows_of_this_layers_routing(self):
        gemm = _bare_gemm()
        for rows in (1, 8, 9, ROUTED_EXPERT_MAX_ROWS):
            with self.subTest(rows=rows):
                ids = torch.zeros(rows, TARGET_TOP_K, dtype=torch.int32)
                self.assertTrue(gemm.supports(*_activation(rows), _topk(ids)))

    def test_declines_everything_else(self):
        gemm = _bare_gemm()
        rows = 6
        x_q, x_s = _activation(rows)
        ids = torch.zeros(rows, TARGET_TOP_K, dtype=torch.int32)
        too_many = ROUTED_EXPERT_MAX_ROWS + 1
        cases = {
            "past the row bound": (
                *_activation(too_many),
                _topk(torch.zeros(too_many, TARGET_TOP_K, dtype=torch.int32)),
            ),
            "other top-k": (x_q, x_s, _topk(torch.zeros(rows, 3, dtype=torch.int32))),
            "no scales": (x_q, None, _topk(ids)),
            "BF16 activations": (x_q.bfloat16(), x_s, _topk(ids)),
            "BF16 weights": (x_q, x_s, _topk(ids, torch.zeros(ids.shape).bfloat16())),
            "non-standard top-k": (x_q, x_s, SimpleNamespace(topk_ids=ids)),
        }
        for name, args in cases.items():
            with self.subTest(name):
                self.assertFalse(gemm.supports(*args))


class TestApply(CustomTestCase):
    def test_returns_the_deferred_finalize_abi(self):
        """The fused finalize + all-reduce consumer reads a flat slot -> row map."""
        rows = 6
        ids = torch.zeros(rows, TARGET_TOP_K, dtype=torch.int32)
        triple = (
            torch.zeros(rows * TARGET_TOP_K * reg.TILE, reg.HIDDEN),
            torch.ones(rows, TARGET_TOP_K),
            torch.arange(rows * TARGET_TOP_K, dtype=torch.int32).view(rows, -1),
        )
        gemm = _bare_gemm()
        with patch.object(reg.RoutedExpertGemm, "__call__", return_value=triple):
            out = gemm.apply(*_activation(rows), _topk(ids)).hidden_states
        self.assertIsInstance(out, FlashInferTrtllmDeferredFinalizeOutput)
        self.assertIs(out.gemm2_out, triple[0])
        self.assertIs(out.expert_weights, triple[1])
        self.assertEqual(out.expanded_idx_to_permuted_idx.shape, (rows * TARGET_TOP_K,))
        self.assertEqual(out.top_k, TARGET_TOP_K)


class TestBuild(CustomTestCase):
    def _layer(self, local_experts=TARGET_EXPERTS):
        return SimpleNamespace(
            num_local_experts=local_experts,
            num_experts=TARGET_EXPERTS,
            top_k=TARGET_TOP_K,
            w13_weight="w13",
            w13_weight_scale_inv="w13_scale",
            w2_weight="w2",
            w2_weight_scale_inv="w2_scale",
        )

    def test_binds_the_layer_at_the_switch_row_bound(self):
        with patch.object(reg, "RoutedExpertGemm", return_value="pair") as pair:
            self.assertEqual(reg.build(self._layer(), "clamp"), "pair")
        self.assertEqual(
            pair.call_args.args, ("w13", "w13_scale", "w2", "w2_scale", "clamp")
        )
        self.assertEqual(
            pair.call_args.kwargs,
            {"top_k": TARGET_TOP_K, "max_rows": ROUTED_EXPERT_MAX_ROWS},
        )

    def test_keeps_flashinfer_when_the_kernels_cannot_serve(self):
        with patch.object(reg, "RoutedExpertGemm") as pair:
            self.assertIsNone(reg.build(self._layer(TARGET_EXPERTS // 4), "clamp"))
        pair.assert_not_called()
        for error in (ValueError, ImportError, OSError, RuntimeError):
            with self.subTest(error=error.__name__):
                with patch.object(reg, "RoutedExpertGemm", side_effect=error("x")):
                    self.assertIsNone(reg.build(self._layer(), "clamp"))


# ----------------------------------------------------------------------------- GPU


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


def _shuffled_layer(seed: int, experts: int):
    """Random MXFP4 expert weights, shuffled as Mxfp4FlashinferTrtllmMoEMethod does."""
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    e, h, p, s = experts, reg.HIDDEN, reg.PADDED, reg.SLICE

    def randint(low, high, shape):
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=generator
        )

    w13 = randint(0, 256, (e, 2 * p, h // 2))
    w13_s = randint(118, 126, (e, 2 * p, h // 32))
    w13.view(e, 2, p, -1)[:, :, s:] = 0
    w2 = randint(0, 256, (e, h, p // 2))
    w2_s = randint(118, 126, (e, h, p // 32))
    w2[:, :, s // 2 :] = 0
    cache: dict = {}
    g1, g1s, g2, g2s = [], [], [], []
    for i in range(e):
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13[i], 128)
        g1.append(w13[i][perm.cuda()].contiguous())
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_s[i], 128, num_elts_per_sf=16
        )
        g1s.append(block_scale_interleave(w13_s[i][perm.cuda()].contiguous()))
        perm = get_w2_permute_indices_with_cache(cache, w2[i], 128)
        g2.append(w2[i][perm.cuda()].contiguous())
        perm = get_w2_permute_indices_with_cache(
            cache, w2_s[i], 128, num_elts_per_sf=16
        )
        g2s.append(block_scale_interleave(w2_s[i][perm.cuda()].contiguous()))
    w13 = torch.stack(g1)
    w13_s = torch.stack(g1s).view(torch.float8_e4m3fn).reshape(e, 2 * p, -1)
    w2 = torch.stack(g2)
    w2_s = torch.stack(g2s).view(torch.float8_e4m3fn).reshape(e, h, -1)
    clamp = torch.full((e,), 10.0, dtype=torch.float32, device="cuda")
    return w13, w13_s, w2, w2_s, clamp


def _flashinfer(x_q, x_s, ids, weights, layer, output=None):
    """FlashInfer's routed MoE as SGLang serves it: GEMM2 rows and the slot -> row
    map, or (``output`` given) the finalized output."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

    from sglang.srt.layers.moe.utils import RoutingMethodType

    w13, s13, w2, s2, clamp = layer
    experts = w13.size(0)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, weights),
        routing_bias=None,
        routing_method_type=int(RoutingMethodType.TopK),
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w13,
        gemm1_weights_scale=s13,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=experts,
        top_k=ids.size(1),
        intermediate_size=reg.PADDED,
        local_expert_offset=0,
        local_num_experts=experts,
        do_finalize=output is not None,
        tune_max_num_tokens=64,
        output=output,
        enable_pdl=True,
    )
    if output is not None:
        return output
    return result[0], result[2].view(ids.shape)


def _gathered(rows_out, permuted):
    return rows_out[permuted.long()].float()


@unittest.skipUnless(
    _gpu_ready(), "needs an SM100-family GPU with FlashInfer and NVRTC"
)
class TestRoutedExpertGemmGpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.layer = _shuffled_layer(2, TARGET_EXPERTS)

    def _check(self, gemm, rows, distinct, graph):
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        generator = torch.Generator().manual_seed(900 + rows + distinct)
        x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16).cuda()
        x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
        x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
        ids = _routing(rows, distinct, generator, TARGET_EXPERTS, TARGET_TOP_K).cuda()
        weights = torch.softmax(
            torch.randn(rows, TARGET_TOP_K, generator=generator), -1
        ).cuda()
        topk = StandardTopKOutput(weights, ids, torch.zeros(rows, 1, device="cuda"))
        self.assertTrue(gemm.supports(x_q, x_s, topk))

        def run():
            return gemm.apply(x_q, x_s, topk).hidden_states

        if graph:
            run()
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                out = run()
            # Several replays: the route kernel flips its barrier word per launch.
            for _ in range(3):
                captured.replay()
        else:
            out = run()
        torch.cuda.synchronize()

        permuted = out.expanded_idx_to_permuted_idx.view(rows, TARGET_TOP_K)
        want_rows, want_permuted = _flashinfer(x_q, x_s, ids, weights, self.layer)
        want = _flashinfer(
            x_q,
            x_s,
            ids,
            weights,
            self.layer,
            output=torch.empty(rows, reg.HIDDEN, dtype=torch.bfloat16, device="cuda"),
        )
        torch.cuda.synchronize()
        # Same MMA order, SwiGLU and quantization as the TRT-LLM-gen tactics; one BF16 ulp.
        got_rows = _gathered(out.gemm2_out, permuted)
        torch.testing.assert_close(
            got_rows, _gathered(want_rows, want_permuted), rtol=2**-7, atol=1e-6
        )
        # What the consumer computes from the triple: the routing-weighted sum per row.
        finalized = (out.expert_weights.view(rows, -1, 1).float() * got_rows).sum(1)
        torch.testing.assert_close(finalized, want.float(), rtol=2**-7, atol=1e-3)

    def test_routing_matches_flashinfer(self):
        gemm = reg.RoutedExpertGemm(
            *self.layer, top_k=TARGET_TOP_K, max_rows=ROUTED_EXPERT_MAX_ROWS
        )
        self.assertIsNotNone(gemm.route_wide)
        for rows in (1, 6, 8, 9, 12, 24, 48):
            # 6 distinct experts: every expert takes every row (several tiles past 8).
            for distinct in (6, 20, TARGET_EXPERTS):
                for graph in (False, True):
                    with self.subTest(rows=rows, distinct=distinct, graph=graph):
                        self._check(gemm, rows, distinct, graph)

    def test_eight_row_bound_keeps_the_original_route(self):
        gemm = reg.RoutedExpertGemm(*self.layer, top_k=TARGET_TOP_K, max_rows=8)
        self.assertIsNone(gemm.route_wide)
        rows = 9
        x_q = torch.zeros(rows, reg.HIDDEN, dtype=torch.uint8, device="cuda")
        x_s = torch.zeros(rows, reg.HIDDEN // 32, dtype=torch.uint8, device="cuda")
        ids = torch.zeros(rows, TARGET_TOP_K, dtype=torch.int32, device="cuda")
        weights = torch.zeros(rows, TARGET_TOP_K, dtype=torch.float32, device="cuda")
        topk = StandardTopKOutput(weights, ids, torch.zeros(rows, 1, device="cuda"))
        self.assertFalse(gemm.supports(x_q, x_s, topk))


if __name__ == "__main__":
    unittest.main()
