"""Real two-GPU PP result transport followed by scheduler commit processing."""

import multiprocessing as mp
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dspark_pp_result_utils import relay_pp_result
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
    "two CUDA devices and NCCL required",
)
class TestDSparkPPResultNCCL(CustomTestCase):
    def test_prefill_and_accepted_block_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=relay_pp_result, args=(rank, root, 2, True))
                for rank in range(2)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 120
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0, 0])
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
