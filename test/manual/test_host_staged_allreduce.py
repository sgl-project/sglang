"""Two-GPU test for the host-staged all-reduce (SGLANG_HOST_STAGED_ALLREDUCE).

Runs the communicator directly under torchrun (no ray, no engine):

    torchrun --nproc_per_node=2 test/manual/test_host_staged_allreduce.py

Checks bit-exactness against torch.distributed's NCCL all-reduce for the
message sizes a TP=2 prefill produces, that the size threshold routes small
messages elsewhere, and that a CUDA-graph capture never sees the host path.
"""
import os
import sys
import time
import unittest

import torch
import torch.distributed as dist


class TestHostStagedAllReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        cls.rank = dist.get_rank()
        cls.world = dist.get_world_size()
        torch.cuda.set_device(cls.rank)
        cls.device = torch.device(f"cuda:{cls.rank}")
        cls.cpu_group = dist.new_group(backend="gloo")
        from sglang.srt.distributed.device_communicators.host_staged_allreduce import (
            HostStagedAllReduce,
        )

        cls.comm = HostStagedAllReduce(cls.cpu_group, cls.rank, cls.world, cls.device)

    def _check(self, numel, dtype=torch.bfloat16, iters=3):
        for _ in range(iters):
            x = torch.randn(numel, dtype=dtype, device=self.device) * (self.rank + 1)
            ref = x.clone()
            dist.all_reduce(ref)
            y = x.clone()
            self.assertTrue(self.comm.should_use(y))
            out = self.comm.all_reduce(y)
            torch.cuda.synchronize()
            self.assertIs(out, y)
            self.assertTrue(torch.equal(ref, y), f"mismatch at numel={numel}")

    def test_prefill_sizes_bit_exact(self):
        # [tokens, hidden] bf16 for a 2048-token chunk at hidden 2560 / 5120,
        # a 1260-token tail, and a message larger than one 32 MB region.
        for numel in (2048 * 2560, 2048 * 5120, 1260 * 5120, 3 * 2048 * 5120):
            self._check(numel)

    def test_fp32_and_odd_shapes(self):
        self._check(1000 * 4096, dtype=torch.float32)
        self._check(1 << 20)  # exactly the default threshold in bytes / 2

    def test_threshold_and_capture_guard(self):
        small = torch.randn(4096, dtype=torch.bfloat16, device=self.device)
        self.assertFalse(self.comm.should_use(small))
        big = torch.randn(2048 * 5120, dtype=torch.bfloat16, device=self.device)
        self.assertTrue(self.comm.should_use(big))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            big.mul_(1)  # keep the capture non-empty
            self.assertFalse(self.comm.should_use(big))

    def test_speed_report(self):
        x = torch.randn(2048 * 5120, dtype=torch.bfloat16, device=self.device)
        for _ in range(5):
            self.comm.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(20):
            self.comm.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 20
        if self.rank == 0:
            print(f"host-staged all-reduce 20 MB: {dt*1e6:.0f} us ({x.numel()*2/dt/1e9:.1f} GB/s)")


if __name__ == "__main__":
    if os.environ.get("WORLD_SIZE") != "2":
        print("run with: torchrun --nproc_per_node=2 " + __file__)
        sys.exit(0)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
