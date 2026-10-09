# SPDX-License-Identifier: Apache-2.0
"""Standalone GPU qualification; use the same upstream kernels as serving."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp_hip import (
    silu_and_mul_clamp_triton,
)
from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k
from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_gate
from sglang.kernels.ops.moe.shared_router_gfx950 import (
    finish,
    project,
    shared_router,
    unpack_shared_down,
)
from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
    mxfp8_gemv,
    prepare_mxfp8_native_weight,
)
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.amd.shared_router import try_shared_router
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


def operands(m, seed=0):
    torch.manual_seed(seed)
    x = torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((1152, 5120), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full((36, 160), 1 / 64, device="cuda")
    sw, ss = prepare_mxfp8_native_weight(w, scale, [32, 32])
    router = (torch.randn((384, 5120), device="cuda") / 64).to(torch.bfloat16)
    down = (
        torch.randn((5120, 576), device="cuda").to(torch.float8_e4m3fn).float() / 32
    ).to(torch.bfloat16)
    bias = (torch.randn(384, device="cuda") / 8).to(torch.bfloat16)
    return x, sw, ss, router, down, bias


def pack_reference_down(down):
    w = (down.float() * 32).to(torch.float8_e4m3fn)
    return prepare_mxfp8_native_weight(
        w, torch.full((160, 18), 1 / 32, device=down.device), [32, 32]
    )


def native(args, packed_down=None):
    x, sw, ss, router, down, bias = args
    gate = mxfp8_gemv(x, sw, ss)
    partials = rocm_router_gemv_split_k(x, router)
    logits = torch.empty((x.shape[0], 384), device=x.device, dtype=torch.float32)
    weights, ids = rocm_router_gate(logits, bias, 6, True, 1.5, partials=partials)
    if packed_down is None:
        packed_down = pack_reference_down(down)
    middle = silu_and_mul_clamp_triton(gate, 10.0, emit_fp8=True)
    output = mxfp8_gemv(middle.q, *packed_down, x_scale=middle.scale)
    return output, weights, ids, logits


def compare(actual, expected):
    out, weights, ids, logits = actual
    ref, ref_weights, ref_ids, ref_logits = expected
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-4)
    # Down uses a different FP32 reduction order, followed by native BF16 output.
    error = (out.float() - ref.float()).square().mean().sqrt()
    relative = error / ref.float().square().mean().sqrt().clamp_min(1e-6)
    assert torch.isfinite(out).all() and relative.item() < 0.004, relative.item()
    torch.testing.assert_close(out, ref, rtol=0.03, atol=0.03)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 ROCm GPU required")
class TestSharedRouter(CustomTestCase):
    def test_model_forward_preserves_collective_boundary(self):
        # Kernel correctness alone did not catch the removed _all_reduce_output
        # helper after the upstream port. Exercise the actual model forward hook
        # with only the expensive projection/expert computation replaced.
        from sglang.srt.models import deepseek_v2 as model

        x = torch.zeros((6, 5120), device="cuda", dtype=torch.bfloat16)
        topk = object()
        batch = NS(moe_num_token_non_padded=lambda: None)
        moe = NS(tp_size=4, is_deepseek_v4=True, routed_scaling_factor=1.5)
        for fused, skip, has_mhc in (
            (True, False, True),
            (False, False, True),
            (False, False, False),
            (False, True, True),
        ):
            events = []
            mhc = (
                NS(start_stats_before_all_reduce=lambda: events.append("stats"))
                if has_mhc
                else None
            )
            moe.experts = Mock(return_value=x)

            def fused_reduce(*args):
                events.append("fused")
                return fused

            def ordinary_reduce(value):
                events.append("ordinary")
                return value

            with (
                patch(
                    "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
                    return_value=False,
                ),
                patch.object(model, "_is_hip", True),
                patch.object(model, "_use_aiter", True),
                patch.object(
                    envs.SGLANG_DSV41_SHARED_ROUTER_FUSION, "get", return_value=True
                ),
                patch(
                    "sglang.srt.models.deepseek_common.amd.shared_router.try_shared_router",
                    return_value=(x, topk),
                ),
                patch.object(
                    model, "maybe_fuse_routed_scale_and_shared_add", return_value=x
                ),
                patch.object(
                    model, "should_skip_post_experts_all_reduce", return_value=skip
                ),
                patch(
                    "sglang.srt.layers.moe.mhc_post_fusion.current_mhc_post_fusion",
                    return_value=mhc,
                ),
                patch.object(
                    model._hip_moe, "fused_all_reduce_mhc", side_effect=fused_reduce
                ),
                patch.object(
                    model, "post_experts_all_reduce", side_effect=ordinary_reduce
                ),
            ):
                self.assertIs(model.DeepseekV2MoE.forward(moe, x, batch), x)
            moe.experts.assert_called_once_with(x, topk)
            expected = (
                ["ordinary"]
                if skip
                else ["fused"]
                if fused
                else ["fused", "stats", "ordinary"]
                if has_mhc
                else ["fused", "ordinary"]
            )
            self.assertEqual(events, expected)

    def test_graph_dynamic_padding_postprocess(self):
        # Exercise the actual HIP masks used by the model integration, not a
        # mocked postprocessor. Graph replay must read the current valid count.
        from sglang.srt.layers.moe.topk import TopKConfig, _post_process_topk_ids

        config = TopKConfig(top_k=6, allow_routed_experts_capture=False)
        for m in (6, 12):
            ids = torch.arange(m * 6, device="cuda", dtype=torch.int32).reshape(m, 6)
            weights = torch.ones((m, 6), device="cuda", dtype=torch.float32)
            logits = torch.zeros((m, 384), device="cuda", dtype=torch.float32)
            count = torch.tensor(m, device="cuda", dtype=torch.int32)

            def call():
                return _post_process_topk_ids(
                    ids.clone(),
                    weights.clone(),
                    config,
                    logits,
                    None,
                    num_token_non_padded=count,
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual_ids, actual_weights, _ = call()
            for n in (m, m - 2, 0, m):
                count.fill_(n)
                graph.replay()
                torch.testing.assert_close(actual_ids[:n], ids[:n], rtol=0, atol=0)
                torch.testing.assert_close(
                    actual_weights[:n], weights[:n], rtol=0, atol=0
                )
                self.assertTrue(bool((actual_ids[n:] == 0).all()))
                self.assertTrue(bool((actual_weights[n:] == 0).all()))

    def test_model_dispatch_and_weight_reload(self):
        batch = NS(
            forward_mode=NS(is_target_verify=lambda: True),
            moe_num_token_non_padded=lambda: None,
        )
        with envs.SGLANG_DSV41_SHARED_ROUTER_FUSION.override(False):
            self.assertIsNone(try_shared_router(None, None, None, False))
        with (
            envs.SGLANG_DSV41_SHARED_ROUTER_FUSION.override(True),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=NS(moe=NS(enable_eplb=False)),
            ),
        ):
            with patch("sglang.srt.utils.is_gfx95_supported", return_value=False):
                self.assertIsNone(try_shared_router(None, None, None, False))
            for m in (6, 12):
                args = operands(m)
                x, sw, ss, rw, down, bias = args
                dw, ds = pack_reference_down(down)
                moe = NS(
                    is_hash=False,
                    is_nextn=False,
                    tp_size=4,
                    moe_ep_size=1,
                    _shared_expert_tp1=False,
                    num_fused_shared_experts=0,
                    _enable_a2a_moe=False,
                    _fuse_shared_experts_inside_sbo=False,
                    routed_scaling_factor=1.5,
                    layer_id=10,
                    _shared_router_logged=True,
                    config=NS(
                        model_type="deepseek_v41",
                        n_routed_experts=384,
                        num_experts_per_tok=6,
                        scoring_func="sqrtsoftplus",
                        norm_topk_prob=True,
                    ),
                    gate=NS(
                        weight=rw,
                        e_score_correction_bias=bias,
                        e_score_correction_bias_vl=None,
                    ),
                    shared_experts=NS(
                        swiglu_limit=10.0,
                        gate_up_proj=NS(
                            mxfp8_native_ready=True,
                            weight=sw.view(torch.float8_e4m3fn),
                            weight_scale_mx_e8m0=ss,
                        ),
                        down_proj=NS(
                            mxfp8_native_ready=True,
                            weight=dw.view(torch.float8_e4m3fn),
                            weight_scale_mx_e8m0=ds,
                        ),
                    ),
                    topk=NS(topk_config=object()),
                )
                for field, value in (
                    ("is_hash", True),
                    ("is_nextn", True),
                    ("tp_size", 8),
                    ("moe_ep_size", 4),
                    ("_shared_expert_tp1", True),
                    ("num_fused_shared_experts", 1),
                    ("_enable_a2a_moe", True),
                ):
                    old = getattr(moe, field)
                    setattr(moe, field, value)
                    self.assertIsNone(try_shared_router(moe, x, batch, False), field)
                    setattr(moe, field, old)
                self.assertIsNone(try_shared_router(moe, x, batch, True))
                self.assertIsNone(try_shared_router(moe, x, None, False))
                for other_m in (0, 1, 4, 7, 24, 48, 96, 192):
                    other = torch.empty((other_m, 5120), device=x.device, dtype=x.dtype)
                    self.assertIsNone(try_shared_router(moe, other, batch, False))
                self.assertIsNone(try_shared_router(moe, x.float(), batch, False))
                with patch(
                    "sglang.srt.layers.moe.topk._post_process_topk_ids",
                    side_effect=lambda ids, weights, *a, **kw: (ids, weights, None),
                ) as post:
                    shared, topk = try_shared_router(moe, x, batch, False)
                    compare(
                        (shared, topk.topk_weights, topk.topk_ids, topk.router_logits),
                        native(args),
                    )
                    self.assertIn("num_token_non_padded", post.call_args.kwargs)
                    previous = moe._shared_router_down_bf16
                    try_shared_router(moe, x, batch, False)
                    self.assertIs(previous, moe._shared_router_down_bf16)
                    ds.add_(1)
                    shared2, _ = try_shared_router(moe, x, batch, False)
                    self.assertIsNot(previous, moe._shared_router_down_bf16)
                    torch.testing.assert_close(shared2, shared * 2, rtol=0, atol=0)

    def test_nonuniform_block_scales(self):
        # Exercise real MXFP8 layout conversion with different powers of two
        # per block, rather than testing only a tensor-wide scale.
        for m in (6, 12):
            for seed in (11, 23):
                x, _, _, router, _, bias = operands(m, seed)
                gate_codes = torch.randn((1152, 5120), device=x.device).to(
                    torch.float8_e4m3fn
                )
                gate_scales = torch.pow(
                    2.0, torch.randint(-8, -3, (36, 160), device=x.device).float()
                )
                sw, ss = prepare_mxfp8_native_weight(gate_codes, gate_scales, [32, 32])
                down_codes = torch.randn((5120, 576), device=x.device).to(
                    torch.float8_e4m3fn
                )
                down_scales = torch.pow(
                    2.0, torch.randint(-8, -3, (160, 18), device=x.device).float()
                )
                packed = prepare_mxfp8_native_weight(down_codes, down_scales, [32, 32])
                down = unpack_shared_down(
                    packed[0].view(torch.float8_e4m3fn), packed[1]
                )
                reference = (
                    down_codes.float()
                    * down_scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
                ).to(torch.bfloat16)
                torch.testing.assert_close(down, reference, rtol=0, atol=0)
                args = x, sw, ss, router, down, bias
                compare(shared_router(*args), native(args, packed))

    def test_alternating_graph_shapes_and_padding(self):
        # Keep both graphs alive: alternating replay must not reuse buffers
        # owned by a different shape. Zero rows model scheduler padding.
        states = []
        for m in (6, 12):
            args = operands(m, 41)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                shared_router(*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = shared_router(*args)
            states.append((args, graph, actual))
        for repeat in range(4):
            for args, graph, actual in states:
                m = args[0].shape[0]
                args[0].copy_(operands(m, 50 + repeat)[0])
                args[0][-2:].zero_()
                graph.replay()
                compare(actual, native(args))

    def test_unpack_down_exact(self):
        args = operands(6)
        w, scale = pack_reference_down(args[4])
        torch.testing.assert_close(
            unpack_shared_down(w.view(torch.float8_e4m3fn), scale),
            args[4],
            rtol=0,
            atol=0,
        )

    def test_projection_and_complete_front(self):
        for m in (6, 12):
            for seed in range(8):
                with self.subTest(m=m, seed=seed):
                    args = operands(m, seed)
                    gate, parts = project(*args[:4])
                    torch.testing.assert_close(
                        gate, mxfp8_gemv(*args[:3]), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        parts,
                        rocm_router_gemv_split_k(args[0], args[3]),
                        rtol=1e-4,
                        atol=1e-4,
                    )
                    compare(finish(gate, parts, args[4], args[5]), native(args))

    def test_graph_replay_changed_inputs(self):
        for m in (6, 12):
            args = operands(m)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    shared_router(*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = shared_router(*args)
            for seed in range(3):
                args[0].copy_(operands(m, seed + 100)[0])
                graph.replay()
                compare(actual, native(args))

    def test_ties_and_nonfinite_routing(self):
        for m in (6, 12):
            args = operands(m)
            gate, parts = project(*args[:4])
            # Test the top-k branch independently, including native tie policy.
            for special in (0.0, float("nan"), float("inf"), -float("inf")):
                parts.zero_()
                parts[0, :, :16] = special
                args[5].zero_()
                result = finish(gate, parts, args[4], args[5])
                logits = torch.empty_like(result[3])
                rw, ri = rocm_router_gate(logits, args[5], 6, True, 1.5, partials=parts)
                torch.testing.assert_close(result[2], ri, rtol=0, atol=0)
                torch.testing.assert_close(
                    result[1], rw, rtol=0, atol=0, equal_nan=True
                )

    def test_reject_unsupported_rows(self):
        for m in (1, 2, 24, 48, 96, 192):
            with self.subTest(m=m), self.assertRaises(AssertionError):
                project(*operands(m)[:4])


if __name__ == "__main__":
    unittest.main()
