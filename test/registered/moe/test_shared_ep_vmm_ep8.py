import os

import pytest
import torch
import torch.distributed as dist

from sglang.srt.layers.moe.shared_ep.epoch import create_gpu_epoch
from sglang.srt.layers.moe.shared_ep.vmm import allocate_rank_major_vmm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=40, stage="extra-b", runner_config="8-gpu-h200")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and "RANK" in os.environ),
    reason="run with torchrun on a CUDA node",
)
class TestSharedEpVmmEp8:
    @classmethod
    def setup_class(cls):
        cls.rank = int(os.environ["RANK"])
        cls.world_size = int(os.environ["WORLD_SIZE"])
        cls.local_rank = int(os.environ["LOCAL_RANK"])
        if cls.world_size != 8:
            raise AssertionError(f"expected EP8, got world size {cls.world_size}")
        torch.cuda.set_device(cls.local_rank)
        dist.init_process_group("gloo")

    @classmethod
    def teardown_class(cls):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def test_every_rank_reads_every_owner_segment(self):
        allocation = None
        try:
            logical_rank_bytes = 4096
            allocation = allocate_rank_major_vmm(
                cpu_group=dist.group.WORLD,
                device=torch.device("cuda", self.local_rank),
                logical_rank_bytes=logical_rank_bytes,
            )
            allocation.local_storage.fill_(self.rank + 1)
            torch.cuda.synchronize()
            dist.barrier()

            for owner in range(self.world_size):
                segment = allocation.global_storage.narrow(
                    0,
                    allocation.rank_offset(owner),
                    logical_rank_bytes,
                )
                assert torch.all(
                    segment == owner + 1
                ).item(), f"rank {self.rank} observed wrong data for owner {owner}"
            dist.barrier()
        finally:
            if allocation is not None:
                allocation.close()

    def test_gpu_epoch_publishes_two_generations_without_host_barrier(self):
        """Release/acquire epochs must order peer payloads and safe reuse."""
        allocation = None
        input_epoch = None
        completion_epoch = None
        try:
            logical_rank_bytes = 4096
            allocation = allocate_rank_major_vmm(
                cpu_group=dist.group.WORLD,
                device=torch.device("cuda", self.local_rank),
                logical_rank_bytes=logical_rank_bytes,
            )
            input_epoch = create_gpu_epoch(
                cpu_group=dist.group.WORLD,
                device=torch.device("cuda", self.local_rank),
                rank=self.rank,
                world_size=self.world_size,
            )
            completion_epoch = create_gpu_epoch(
                cpu_group=dist.group.WORLD,
                device=torch.device("cuda", self.local_rank),
                rank=self.rank,
                world_size=self.world_size,
            )

            for generation in (1, 2):
                allocation.local_storage.fill_(generation * 16 + self.rank)
                input_epoch.publish()
                input_epoch.wait_all()
                for owner in range(self.world_size):
                    segment = allocation.global_storage.narrow(
                        0,
                        allocation.rank_offset(owner),
                        logical_rank_bytes,
                    )
                    assert torch.all(segment == generation * 16 + owner).item(), (
                        f"rank {self.rank} expected {generation * 16 + owner} for "
                        f"owner {owner}, observed {torch.unique(segment).tolist()}"
                    )
                completion_epoch.publish()
                completion_epoch.wait_all()
            dist.barrier()
        finally:
            if completion_epoch is not None:
                completion_epoch.close()
            if input_epoch is not None:
                input_epoch.close()
            if allocation is not None:
                allocation.close()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
