# SPDX-License-Identifier: Apache-2.0
"""LL backend selection and opt-in old/new two-layer CUDA Graph benchmark.

Numerical block-scale and two-layer independent-reference tests live in
 test_cutedsl_w4afp8_gemm_per_token_block.py. This file does not swallow errors.
"""

import json
import os
import statistics
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod, interleave_scales


class TestLLSelection(unittest.TestCase):
    def test_explicit_backend_selection(self):
        method = W4AFp8MoEMethod(SimpleNamespace())
        for name in (
            "a_strides1",
            "b_strides1",
            "c_strides1",
            "a_strides2",
            "b_strides2",
            "c_strides2",
            "s_strides13",
            "s_strides2",
            "expert_offsets",
            "problem_sizes1",
            "problem_sizes2",
        ):
            setattr(method, name, None)
        layer = SimpleNamespace(
            quant_method=method,
            w13_weight=None,
            w2_weight=None,
            w13_weight_scale_inv=None,
            w2_weight_scale_inv=None,
            w13_input_scale=None,
            w2_input_scale=None,
        )
        dispatch = (torch.empty(1, 0, 128), torch.empty(1, 0, 1), None, None, None, 0)
        for enabled in (False, True):
            with (
                patch.object(
                    envs.SGLANG_W4AFP8_CUTEDSL_LL, "get", return_value=enabled
                ),
                patch(
                    "sglang.srt.layers.moe.cutedsl_w4afp8_moe.cutedsl_w4afp8_moe_deepep_ll"
                ) as new,
                patch(
                    "sglang.srt.layers.moe.cutlass_w4a8_moe.cutlass_w4a8_moe_deepep_ll"
                ) as old,
            ):
                method.apply_deepep_ll(layer, dispatch)
                self.assertEqual(new.call_count, int(enabled))
                self.assertEqual(old.call_count, int(not enabled))

    def test_normal_rejects_block_fp8_even_when_empty(self):
        method = W4AFp8MoEMethod(SimpleNamespace())
        dispatch = SimpleNamespace(
            hidden_states=torch.empty(0, 128, dtype=torch.float8_e4m3fn),
            topk_ids=torch.empty(0, 1, dtype=torch.int64),
            topk_weights=torch.empty(0, 1),
        )
        with self.assertRaisesRegex(RuntimeError, "requires BF16"):
            method.apply_deepep_normal(SimpleNamespace(), dispatch)
        dispatch.hidden_states = torch.empty(0, 128, dtype=torch.bfloat16)
        self.assertIs(
            method.apply_deepep_normal(SimpleNamespace(), dispatch),
            dispatch.hidden_states,
        )


@unittest.skipUnless(os.getenv("SGLANG_BENCH_W4AFP8_AB") == "1", "opt-in GPU benchmark")
class TestLLBenchmark(unittest.TestCase):
    def test_two_layer_ab(self):
        from sglang.srt.layers.moe.cutedsl_w4afp8_moe import (
            cutedsl_w4afp8_moe_deepep_ll,
        )
        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe_deepep_ll

        torch.manual_seed(42)
        results = []
        for e, m, k, n in [
            (64, 128, 4096, 1024),
            (64, 256, 4096, 1024),
            (8, 128, 7168, 2048),
        ]:
            for active in (8, m):
                a = (torch.randn(e, m, k, device="cuda") * 8).to(torch.float8_e4m3fn)
                sa = (
                    torch.rand(e, k // 128, m, device="cuda").transpose(1, 2) * 0.01
                    + 0.005
                )
                w1 = torch.randint(
                    -128, 128, (e, 2 * n, k // 2), device="cuda", dtype=torch.int8
                )
                w2 = torch.randint(
                    -128, 128, (e, k, n // 2), device="cuda", dtype=torch.int8
                )
                s1 = interleave_scales(
                    torch.full(
                        (e, 2 * n, k // 128), 0.01, device="cuda", dtype=torch.bfloat16
                    )
                )
                s2 = interleave_scales(
                    torch.full(
                        (e, k, n // 128), 0.01, device="cuda", dtype=torch.bfloat16
                    )
                )
                mask = torch.full((e,), active, dtype=torch.int32, device="cuda")
                mask[0] = 0
                ids = torch.zeros(8, 10, dtype=torch.int64, device="cuda")
                strides = [
                    torch.full((e, 3), v, device="cuda", dtype=torch.int64)
                    for v in (k, k, 2 * n, n, n, k, 2 * n, k)
                ]
                offsets = torch.zeros(e + 1, dtype=torch.int32, device="cuda")
                sizes = [
                    torch.empty(e, 3, dtype=torch.int32, device="cuda")
                    for _ in range(2)
                ]
                scales = [torch.full((1,), v, device="cuda") for v in (0.03, 0.03)]
                args = (
                    a,
                    sa,
                    w1,
                    w2,
                    s1,
                    s2,
                    ids,
                    mask,
                    *strides,
                    offsets,
                    *sizes,
                    *scales,
                )
                graphs = {}
                for label, fn in (
                    ("cutlass", cutlass_w4a8_moe_deepep_ll),
                    ("cute", cutedsl_w4afp8_moe_deepep_ll),
                ):
                    for _ in range(3):
                        output = fn(*args, expected_m=active)
                    self.assertTrue(torch.isfinite(output[1:, :active]).all().item())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(10):
                            output = fn(*args, expected_m=active)
                    graphs[label] = graph
                times = {key: [] for key in graphs}
                for repeat in range(7):
                    for label in list(graphs) if repeat % 2 else list(reversed(graphs)):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        graphs[label].replay()
                        end.record()
                        end.synchronize()
                        times[label].append(start.elapsed_time(end) * 100)
                result = dict(e=e, capacity=m, active=active, k=k, n=n, us=times)
                result["speedup"] = statistics.median(
                    times["cutlass"]
                ) / statistics.median(times["cute"])
                results.append(result)
                print(json.dumps(result), flush=True)
        output_path = os.getenv("SGLANG_W4AFP8_AB_RESULT")
        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    unittest.main()
