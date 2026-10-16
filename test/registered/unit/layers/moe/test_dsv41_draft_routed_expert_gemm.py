"""DSpark draft MoE on the DSV4.1 routed-expert GEMM pair (--enable-dsv41-low-latency-decode).

CPU: the draft install binds the pair with the local finalize to the draft's MXFP4
MoE and the low-latency MXFP8 GEMM to its attention projections; a pair without the
finalize kernel declines non-deferred calls (the target's); the graph-rows warning.
GPU (SM100 family): for 128 experts / top-3 the route + GEMM pair + finalize match
FlashInfer's routed MoE at 1-40 rows, eager and replayed from a CUDA graph.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.dsv41_low_latency import routed_expert_gemm as reg
from sglang.srt.models.dsv41_low_latency.dspark import install as draft_install
from sglang.srt.models.dsv41_low_latency.dspark.worker import _check_draft_graph_rows
from sglang.srt.models.dsv41_low_latency.switch import ROUTED_EXPERT_MAX_ROWS
from sglang.test.test_utils import CustomTestCase

DRAFT_EXPERTS, DRAFT_TOP_K = 128, 3


def _stage(moe_method):
    mlp = MagicMock(spec=DeepseekV2MoE)
    mlp.experts = SimpleNamespace(quant_method=moe_method)
    attn = SimpleNamespace(wq_b=object(), wo_b=object())
    return SimpleNamespace(self_attn=attn, mlp=mlp)


def _mxfp4_method():
    method = object.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    method.low_batch_gemm_factory = None
    return method


class TestInstallDraftModel(CustomTestCase):
    def test_binds_the_pair_with_finalize_and_the_mxfp8_projections(self):
        method = _mxfp4_method()
        other = SimpleNamespace(low_batch_gemm_factory=None)
        model = SimpleNamespace(stages=[_stage(method), _stage(other)])
        with patch.object(draft_install.mxfp8_linear, "install") as mxfp8:
            draft_install.install_draft_model(model)
        factory = method.low_batch_gemm_factory
        self.assertIsInstance(factory, partial)
        self.assertIs(factory.func, reg.build)
        self.assertEqual(factory.keywords, {"finalize": True})
        # A non-MXFP4 MoE keeps its own kernels.
        self.assertIsNone(other.low_batch_gemm_factory)
        self.assertEqual(
            [call.args[0] for call in mxfp8.call_args_list],
            [(s.self_attn.wq_b, s.self_attn.wo_b) for s in model.stages],
        )

    def test_build_passes_the_finalize(self):
        layer = SimpleNamespace(
            num_local_experts=DRAFT_EXPERTS,
            num_experts=DRAFT_EXPERTS,
            top_k=DRAFT_TOP_K,
            w13_weight="w13",
            w13_weight_scale_inv="w13_scale",
            w2_weight="w2",
            w2_weight_scale_inv="w2_scale",
        )
        with patch.object(reg, "RoutedExpertGemm", return_value="pair") as pair:
            self.assertEqual(reg.build(layer, "clamp", finalize=True), "pair")
        self.assertTrue(pair.call_args.kwargs["finalize"])


def _bare_gemm(finalize_kernel):
    gemm = object.__new__(reg.RoutedExpertGemm)
    gemm.max_rows, gemm.top_k = ROUTED_EXPERT_MAX_ROWS, DRAFT_TOP_K
    gemm.finalize_kernel = finalize_kernel
    return gemm


def _inputs(rows):
    x_q = torch.zeros(rows, reg.HIDDEN, dtype=torch.uint8)
    x_s = torch.zeros(rows, reg.HIDDEN // 32, dtype=torch.uint8)
    ids = torch.zeros(rows, DRAFT_TOP_K, dtype=torch.int32)
    topk = StandardTopKOutput(
        torch.full(ids.shape, 1.0 / DRAFT_TOP_K), ids, torch.zeros(rows, 1)
    )
    return x_q, x_s, topk


class TestLocalFinalize(CustomTestCase):
    def test_non_deferred_calls_need_the_finalize_kernel(self):
        x_q, x_s, topk = _inputs(5)
        target = _bare_gemm(finalize_kernel=None)
        self.assertTrue(target.supports(x_q, x_s, topk, finalize=False))
        self.assertFalse(target.supports(x_q, x_s, topk, finalize=True))
        draft = _bare_gemm(finalize_kernel=object())
        self.assertTrue(draft.supports(x_q, x_s, topk, finalize=True))

    def test_apply_finalizes_into_the_output(self):
        rows = 5
        x_q, x_s, topk = _inputs(rows)
        triple = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        output = torch.empty(rows, reg.HIDDEN, dtype=torch.bfloat16)
        gemm = _bare_gemm(finalize_kernel=object())
        with (
            patch.object(reg.RoutedExpertGemm, "__call__", return_value=triple),
            patch.object(reg.RoutedExpertGemm, "finalize") as finalize,
        ):
            out = gemm.apply(x_q, x_s, topk, output).hidden_states
        self.assertIs(out, output)
        self.assertEqual(finalize.call_args.args, (*triple, output))


class TestDraftGraphRows(CustomTestCase):
    def test_warns_when_no_draft_graph_fits(self):
        for capture_bs, rows_per_request, ok in (
            ([1, 2, 4, 8], 5, True),  # one request: a 5-row graph
            ([8, 16], 6, True),  # 48 rows, the bound
            ([16, 32], 5, False),  # smallest graph has 80 rows
            ([], 5, True),  # no CUDA graphs
        ):
            with self.subTest(capture_bs=capture_bs):
                self.assertIs(
                    _check_draft_graph_rows(
                        capture_bs=capture_bs, rows_per_request=rows_per_request
                    ),
                    ok,
                )


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


def _flashinfer_finalized(x_q, x_s, ids, weights, layer):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

    from sglang.srt.layers.moe.utils import RoutingMethodType

    w13, s13, w2, s2, clamp = layer
    experts = w13.size(0)
    output = torch.empty(ids.size(0), reg.HIDDEN, dtype=torch.bfloat16, device="cuda")
    trtllm_fp4_block_scale_routed_moe(
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
        do_finalize=True,
        tune_max_num_tokens=64,
        output=output,
        enable_pdl=True,
    )
    return output


@unittest.skipUnless(
    _gpu_ready(), "needs an SM100-family GPU with FlashInfer and NVRTC"
)
class TestDraftRoutedExpertGemmGpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.layer = _shuffled_layer(1, DRAFT_EXPERTS)
        cls.gemm = reg.RoutedExpertGemm(
            *cls.layer,
            top_k=DRAFT_TOP_K,
            max_rows=ROUTED_EXPERT_MAX_ROWS,
            finalize=True,
        )

    def _check(self, rows, graph):
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        generator = torch.Generator().manual_seed(500 + rows)
        x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16).cuda()
        x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
        x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
        pool = torch.randperm(DRAFT_EXPERTS, generator=generator)[:12]
        ids = torch.stack(
            [
                pool[torch.randperm(12, generator=generator)[:DRAFT_TOP_K]]
                for _ in range(rows)
            ]
        ).to(device="cuda", dtype=torch.int32)
        weights = torch.softmax(
            torch.randn(rows, DRAFT_TOP_K, generator=generator), -1
        ).cuda()
        topk = StandardTopKOutput(weights, ids, torch.zeros(rows, 1, device="cuda"))
        output = torch.empty(rows, reg.HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(self.gemm.supports(x_q, x_s, topk, finalize=True))

        def run():
            return self.gemm.apply(x_q, x_s, topk, output).hidden_states

        if graph:
            run()  # warm-up outside capture
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                run()
            output.zero_()
            for _ in range(3):
                captured.replay()
        else:
            run()
        torch.cuda.synchronize()
        want = _flashinfer_finalized(x_q, x_s, ids, weights, self.layer)
        torch.cuda.synchronize()
        # Same GEMM2 rows within one BF16 ulp, then FlashInfer's finalize arithmetic.
        torch.testing.assert_close(output.float(), want.float(), rtol=2**-7, atol=1e-3)

    def test_finalized_output_matches_flashinfer(self):
        # 5 and 40 rows: DSpark draft steps at batch 1 and 8.
        for rows in (1, 5, 8, 40):
            for graph in (False, True):
                with self.subTest(rows=rows, graph=graph):
                    self._check(rows, graph)


if __name__ == "__main__":
    unittest.main()
