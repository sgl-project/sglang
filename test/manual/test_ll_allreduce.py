"""Two-GPU test for the LL all-reduce through host memory (SGLANG_LL_ALLREDUCE).

    torchrun --nproc_per_node=2 test/manual/test_ll_allreduce.py

Checks bit-exactness against torch.distributed's NCCL all-reduce for decode
sizes, that only small bf16 messages are routed to it, that it replays
correctly inside a CUDA graph (one graph holding many calls, as a decode step
does), and reports the per-call time against NCCL.
"""
import os
import sys
import time
import unittest

import torch
import torch.distributed as dist


class TestLLAllReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        cls.rank = dist.get_rank()
        cls.world = dist.get_world_size()
        torch.cuda.set_device(cls.rank)
        cls.device = torch.device(f"cuda:{cls.rank}")
        cls.cpu_group = dist.new_group(backend="gloo")
        from sglang.srt.distributed.device_communicators.ll_allreduce import (
            LLAllReduce,
        )

        cls.comm = LLAllReduce(cls.cpu_group, cls.rank, cls.world, cls.device)

    def _check(self, numel, iters=5):
        for _ in range(iters):
            x = torch.randn(numel, dtype=torch.bfloat16, device=self.device) * (self.rank + 1)
            ref = x.clone()
            dist.all_reduce(ref)
            y = x.clone()
            self.assertTrue(self.comm.should_use(y))
            out = self.comm.all_reduce(y)
            torch.cuda.synchronize()
            self.assertIs(out, y)
            self.assertTrue(torch.equal(ref, y), f"mismatch at numel={numel}")

    def test_decode_sizes_bit_exact(self):
        # hidden 2048 / 4096 / 7168 at batch 1, and hidden 4096 at batch 8
        for numel in (2048, 4096, 7168, 8 * 4096):
            self._check(numel)

    def test_routing(self):
        big = torch.randn(1 << 20, dtype=torch.bfloat16, device=self.device)
        self.assertFalse(self.comm.should_use(big))
        f32 = torch.randn(4096, dtype=torch.float32, device=self.device)
        self.assertFalse(self.comm.should_use(f32))
        odd = torch.randn(4095, dtype=torch.bfloat16, device=self.device)
        self.assertFalse(self.comm.should_use(odd))

    def test_cuda_graph_replay(self):
        calls = 88
        xs = [torch.zeros(4096, dtype=torch.bfloat16, device=self.device) for _ in range(calls)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for x in xs:
                self.comm.all_reduce(x)  # warm-up outside the capture
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for x in xs:
                self.comm.all_reduce(x)
        for _ in range(3):
            inputs = [torch.randn(4096, dtype=torch.bfloat16, device=self.device) * (self.rank + 1) for _ in xs]
            refs = [i.clone() for i in inputs]
            for r in refs:
                dist.all_reduce(r)
            for x, i in zip(xs, inputs):
                x.copy_(i)
            torch.cuda.synchronize()
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            for x, r in zip(xs, refs):
                self.assertTrue(torch.equal(x, r))

    def test_speed_report(self):
        x = torch.randn(4096, dtype=torch.bfloat16, device=self.device)
        n = 2000

        def timed(fn):
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            t = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / n

        ll = timed(lambda: self.comm.all_reduce(x))
        nccl = timed(lambda: dist.all_reduce(x))
        if self.rank == 0:
            print(f"4096 x bf16 all-reduce: LL {ll*1e6:.1f} us, NCCL {nccl*1e6:.1f} us")


if __name__ == "__main__":
    if os.environ.get("WORLD_SIZE") != "2":
        print("run with: torchrun --nproc_per_node=2 " + __file__)
        sys.exit(0)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
