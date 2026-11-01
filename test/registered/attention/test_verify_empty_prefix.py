"""An empty committed prefix must reduce to causal draft-only attention."""

import unittest

import torch

from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestVerifyEmptyPrefix(CustomTestCase):
    def test_empty_prefix_graph_replay(self):
        torch.manual_seed(42)
        for fn in (verify_splitkv_fwd, verify_shared_kv_fwd):
            with self.subTest(kernel=fn.__name__):
                # Request zero is empty; another request keeps the batch's
                # index buffer nonempty so the shared-KV wrapper handles it.
                q = torch.randn(6, 4, 64, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(6, 1, 64, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                kb = torch.randn(32, 1, 64, device="cuda", dtype=torch.bfloat16)
                vb = torch.randn_like(kb)
                qo = torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)
                kvp = torch.tensor([0, 0, 32], device="cuda", dtype=torch.int32)
                ids = torch.arange(32, device="cuda", dtype=torch.int64)
                out = torch.empty_like(q)

                def run():
                    self.assertTrue(
                        fn(
                            q,
                            k,
                            v,
                            out,
                            kb,
                            vb,
                            qo,
                            kvp,
                            ids,
                            None,
                            True,
                            None,
                            3,
                            1.0,
                            1.0,
                            sm_scale=0.125,
                        )
                    )

                def check():
                    self.assertTrue(torch.isfinite(out).all().item())
                    # Compute the three-token causal reference in FP64, using
                    # only the current draft keys/values of the empty request.
                    qr = q[:3].double().transpose(0, 1)
                    scores = qr @ k[:3, 0].double().T * 0.125
                    mask = torch.ones(3, 3, device="cuda", dtype=torch.bool).tril()
                    scores.masked_fill_(~mask, -float("inf"))
                    ref = (scores.softmax(-1) @ v[:3, 0].double()).transpose(0, 1)
                    torch.testing.assert_close(
                        out[:3].double(), ref, atol=0.02, rtol=0.01
                    )

                run()
                check()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                # Exercise empty -> nonempty -> empty with unchanged pointers.
                kvp.copy_(torch.tensor([0, 1, 32], device="cuda", dtype=torch.int32))
                graph.replay()
                self.assertTrue(torch.isfinite(out).all().item())
                kvp.copy_(torch.tensor([0, 0, 32], device="cuda", dtype=torch.int32))
                q.mul_(0.5)
                out.fill_(float("nan"))
                graph.replay()
                check()


if __name__ == "__main__":
    unittest.main()
