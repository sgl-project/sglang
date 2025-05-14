import traceback
import unittest

import torch
import torch.multiprocessing as mp
from sglang.srt.model_executor import expert_location_updater
from sglang.test.test_utils import CustomTestCase
from sglang.utils import is_in_ci
from torch.multiprocessing import Process
from tqdm import tqdm


class TestExpertLocationUpdater(CustomTestCase):
    def test_cpu(self):
        self._test_core(num_gpus=32, nnodes=4, num_logical_experts=256, num_physical_experts=288, device="cpu")
        self._test_core(num_gpus=144, nnodes=18, num_logical_experts=256, num_physical_experts=288, device="cpu")
        self._test_common(device="cpu")

    def test_gpu(self):
        if is_in_ci():
            return
        self._test_common(device="cuda")

    def _test_common(self, device):
        for nnodes in [1, 2, 4]:
            for num_logical_experts in [2, 5, 20, 200]:
                for num_physical_experts in [4, 16, 220]:
                    if num_logical_experts > num_physical_experts: continue
                    self._test_core(num_gpus=8, nnodes=nnodes, num_logical_experts=num_logical_experts,
                                    num_physical_experts=num_physical_experts, device=device)

    def _test_core(
        self,
        num_gpus: int,
        **kwargs,
    ):
        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for rank in range(num_gpus):
            p = Process(
                target=_run_subprocess,
                kwargs=dict(
                    rank=rank,
                    num_gpus=num_gpus,
                    output_writer=output_writer,
                    **kwargs,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(num_gpus):
            self.assertTrue(output_reader.recv(), f"Subprocess has error, please see logs above.")

        for p in processes:
            p.join()


def _run_subprocess(
    rank: int,
    num_gpus: int,
    nnodes: int,
    num_logical_experts: int,
    num_physical_experts: int,
    output_writer,
    device: str,
):
    try:
        torch.random.manual_seed(42)
        num_local_physical_experts = num_physical_experts // num_gpus
        num_gpu_per_node = TODO

        def _create_routed_experts_weights(physical_to_logical_map):
            local_logical_expert_ids = physical_to_logical_map[
                                       rank * num_local_physical_experts: (rank + 1) * num_local_physical_experts].cpu()
            return [
                local_logical_expert_ids.to(device),
                torch.tensor([
                    [local_logical_expert_id * 10, local_logical_expert_id * 100]
                    for local_logical_expert_id in local_logical_expert_ids.tolist()
                ], device=device),
            ]

        def _create_physical_to_logical_map():
            ans = torch.concat([
                torch.arange(0, num_logical_experts),
                torch.randint(0, num_logical_experts, (num_physical_experts - num_logical_experts,)),
            ])
            ans = ans[torch.randperm(ans.shape[0])]
            return ans

        physical_to_logical_map = _create_physical_to_logical_map()
        routed_experts_weights = _create_routed_experts_weights(physical_to_logical_map)

        for _ in tqdm(range(5000)):
            new_physical_to_logical_map = _create_physical_to_logical_map()
            expect_new_weights = _create_routed_experts_weights(new_physical_to_logical_map)

            expert_location_updater.update_expert_weights_single_layer(
                routed_experts_weights=routed_experts_weights,
                temp_buffers=expert_location_updater.create_temp_buffers(routed_experts_weights),
                old_physical_to_logical_map=physical_to_logical_map,
                new_physical_to_logical_map=new_physical_to_logical_map,
                num_local_physical_experts=num_local_physical_experts,
                num_gpu_per_node=num_gpu_per_node,
                rank=rank,
            )
            assert all(torch.all(x == y) for x, y in zip(routed_experts_weights, expect_new_weights, strict=True))

            physical_to_logical_map = new_physical_to_logical_map

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


if __name__ == "__main__":
    unittest.main()
