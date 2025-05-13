import traceback
import unittest

import torch
import torch.multiprocessing as mp
from sglang.srt.model_executor import expert_location_updater
from sglang.test.test_utils import CustomTestCase
from sglang.utils import is_in_ci
from torch.multiprocessing import Process


class TestExpertLocationUpdater(CustomTestCase):
    def test_cpu(self):
        self._test_core(num_gpus=32, nnodes=4, num_logical_experts=256, num_physical_experts=288)
        self._test_core(num_gpus=144, nnodes=18, num_logical_experts=256, num_physical_experts=288)
        self._test_common()

    def test_gpu(self):
        if is_in_ci():
            return
        self._test_common(device="cuda")

    def _test_common(self, device="cpu"):
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
        def _create_routed_experts_weights(physical_to_logical_map):
            return TODO

        routed_experts_weights = _create_routed_experts_weights(TODO)
        for _ in range(5000):
            TODO_prepare
            expert_location_updater.update_expert_weights_single_layer(
                routed_experts_weights=routed_experts_weights,
                temp_buffers=expert_location_updater.create_temp_buffers(routed_experts_weights),
                old_physical_to_logical_map=old_physical_to_logical_map,
                new_physical_to_logical_map=new_physical_to_logical_map,
            )
            expect_new_weights = _create_routed_experts_weights(new_physical_to_logical_map)
            assert all(torch.all(x == y) for x, y in zip(routed_experts_weights, expect_new_weights, strict=True))

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


if __name__ == "__main__":
    unittest.main()
