import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMoeFinalizeGateAdd(unittest.TestCase):
    def inputs(self, rows, weight_dtype, padded=False):
        torch.manual_seed(rows)
        hidden, top_k = 2560, 10
        gemm2 = torch.randn(
            rows * top_k + 32, hidden + 128, device="cuda", dtype=torch.bfloat16
        )[:, :hidden]
        assert gemm2.stride(0) > gemm2.shape[1]
        perm = torch.randperm(gemm2.shape[0], device="cuda")[: rows * top_k].int()
        if padded:
            perm[::7] = -1
        weights = torch.softmax(torch.randn(rows, top_k, device="cuda"), -1).to(
            weight_dtype
        )
        hidden_states = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.02
        shared = torch.randn_like(hidden_states) * 0.05
        return gemm2, perm, weights, hidden_states, gate, shared, top_k

    def reference(self, args):
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add,
        )
        from sglang.kernels.ops.moe.moe_finalize_fuse_shared import (
            moe_finalize_fuse_shared,
        )

        gemm2, perm, weights, hidden, gate, shared, top_k = args
        # The CUDA reference uses size(1) as its row stride.
        out = moe_finalize_fuse_shared(gemm2.contiguous(), perm, weights, None, top_k)
        fused_gate_sigmoid_mul_add(hidden, gate, shared, out)
        return out

    def test_bf16_bitwise_and_graph_replay(self):
        from sglang.kernels.ops.moe.moe_finalize_gate_add import moe_finalize_gate_add

        for rows in (1, 64, 256, 384, 1024):
            for padded in (False, True):
                with self.subTest(rows=rows, padded=padded):
                    args = self.inputs(rows, torch.bfloat16, padded)
                    reference = self.reference(args)
                    actual = moe_finalize_gate_add(*args)
                    self.assertTrue(
                        torch.equal(
                            reference.view(torch.int16), actual.view(torch.int16)
                        )
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        moe_finalize_gate_add(*args)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        replayed = moe_finalize_gate_add(*args)
                    for _ in range(2):
                        graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(
                            reference.view(torch.int16), replayed.view(torch.int16)
                        )
                    )

    def test_fp32_weights_use_reference_rounding(self):
        from sglang.kernels.ops.moe.moe_finalize_gate_add import moe_finalize_gate_add

        args = self.inputs(384, torch.float32, padded=True)
        reference = self.reference(args)
        actual = moe_finalize_gate_add(*args)
        self.assertTrue(
            torch.equal(reference.view(torch.int16), actual.view(torch.int16))
        )


if __name__ == "__main__":
    unittest.main()
