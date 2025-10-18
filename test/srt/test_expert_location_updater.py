import os
import traceback
import unittest
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from sglang.srt.eplb import expert_location_updater
from sglang.test.test_utils import CustomTestCase, find_available_port
from sglang.utils import is_in_ci

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


@dataclass
class _TestInfo:
    nnodes: int
    num_logical_experts: int
    num_physical_experts: int
    num_repeat: int = 5000


class TestExpertLocationUpdater(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_cpu(self):
        self._test_common(device="cpu")
        self._test_core(
            num_gpus=32,
            device="cpu",
            infos=[
                _TestInfo(
                    nnodes=4,
                    num_logical_experts=256,
                    num_physical_experts=288,
                    num_repeat=10000,
                )
            ],
        )

    def test_cpu_slow(self):
        if is_in_ci():
            return
        self._test_core(
            num_gpus=144,
            device="cpu",
            infos=[
                _TestInfo(
                    nnodes=18,
                    num_logical_experts=256,
                    num_physical_experts=288,
                    num_repeat=10000,
                )
            ],
        )

    def test_gpu(self):
        if is_in_ci():
            return
        self._test_common(device=device_type)

    def _test_common(self, device):
        infos = []

        for nnodes in [1, 2, 4]:
            for num_logical_experts in [2, 5, 20, 256]:
                for num_physical_experts in [8, 16, 256, 288]:
                    if num_logical_experts > num_physical_experts:
                        continue
                    infos.append(
                        _TestInfo(
                            nnodes=nnodes,
                            num_logical_experts=num_logical_experts,
                            num_physical_experts=num_physical_experts,
                        )
                    )

        self._test_core(num_gpus=8, device=device, infos=infos)

    def _test_core(
        self,
        num_gpus: int,
        device: str,
        infos: List[_TestInfo],
    ):
        master_port = find_available_port(23456)

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for rank in range(num_gpus):
            p = Process(
                target=_run_subprocess,
                kwargs=dict(
                    rank=rank,
                    num_gpus=num_gpus,
                    output_writer=output_writer,
                    master_port=master_port,
                    device=device,
                    infos=infos,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(num_gpus):
            self.assertTrue(
                output_reader.recv(), f"Subprocess has error, please see logs above."
            )

        for p in processes:
            p.join()


def _run_subprocess(
    rank: int,
    num_gpus: int,
    master_port: int,
    device: str,
    infos: List[_TestInfo],
    output_writer,
):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        torch.random.manual_seed(42)
        torch.distributed.init_process_group(
            rank=rank,
            world_size=num_gpus,
            backend={"cpu": "gloo", "cuda": None}[device],
        )
        if device == "cuda":
            torch.cuda.set_device(f"cuda:{rank}")
        if device == "xpu":
            torch.xpu.set_device(f"xpu:{rank}")

        for info in infos:
            _execute_test(info, rank=rank, num_gpus=num_gpus, device=device)

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


def _execute_test(info: _TestInfo, rank: int, num_gpus: int, device: str):
    if rank == 0:
        print(f"Test: {num_gpus=} {info=}", flush=True)

    assert info.num_physical_experts % num_gpus == 0
    num_local_physical_experts = info.num_physical_experts // num_gpus
    assert num_gpus % info.nnodes == 0
    num_gpu_per_node = num_gpus // info.nnodes

    def _create_routed_experts_weights(physical_to_logical_map):
        local_logical_expert_ids = physical_to_logical_map[
            rank * num_local_physical_experts : (rank + 1) * num_local_physical_experts
        ].cpu()
        return [
            local_logical_expert_ids.to(device).clone(),
            torch.tensor(
                [
                    [local_logical_expert_id * 10, local_logical_expert_id * 100]
                    for local_logical_expert_id in local_logical_expert_ids.tolist()
                ],
                device=device,
            ),
        ]

    def _create_physical_to_logical_map():
        if rank == 0:
            ans = torch.concat(
                [
                    torch.arange(0, info.num_logical_experts),
                    torch.randint(
                        0,
                        info.num_logical_experts,
                        (info.num_physical_experts - info.num_logical_experts,),
                    ),
                ]
            )
            ans = ans[torch.randperm(ans.shape[0])]
        else:
            ans = torch.empty((info.num_physical_experts,), dtype=torch.int64)

        assert ans.dtype == torch.int64 and ans.shape == (info.num_physical_experts,)
        ans = ans.to(device)
        torch.distributed.broadcast(ans, src=0)

        return ans.cpu()

    physical_to_logical_map = _create_physical_to_logical_map()
    routed_experts_weights = _create_routed_experts_weights(physical_to_logical_map)

    for i in range(info.num_repeat):
        if rank == 0 and ((i % 500 == 0) or (i == info.num_repeat - 1)):
            print(f"Step {i}/{info.num_repeat}", flush=True)

        new_physical_to_logical_map = _create_physical_to_logical_map()
        expect_new_weights = _create_routed_experts_weights(new_physical_to_logical_map)

        output_logs = expert_location_updater.update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights,
            temp_buffers=expert_location_updater.create_temp_buffers(
                routed_experts_weights
            ),
            old_physical_to_logical_map=physical_to_logical_map.tolist(),
            new_physical_to_logical_map=new_physical_to_logical_map.tolist(),
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            debug=True,
        )

        local_has_error = not all(
            torch.all(x == y)
            for x, y in zip(routed_experts_weights, expect_new_weights, strict=True)
        )
        global_has_error = torch.tensor(local_has_error, device=device)
        torch.distributed.all_reduce(
            global_has_error, op=torch.distributed.ReduceOp.MAX
        )

        if global_has_error.cpu().item():
            output_logs_str = "\n".join(output_logs)
            local_message = (
                f"===================== rank {rank} ============================\n"
                f"{num_gpus=} {info=}\n"
                f"{routed_experts_weights[0].tolist()=}\n"
                f"{expect_new_weights[0].tolist()=}\n"
                f"{physical_to_logical_map.tolist()=}\n"
                f"{new_physical_to_logical_map.tolist()=}\n"
                f"===logs===\n"
                f"{output_logs_str}\n"
                f"==============================================================\n"
            )

            global_messages = ([None] * num_gpus) if rank == 0 else None
            torch.distributed.gather_object(local_message, global_messages, dst=0)

            if rank == 0:
                print("\n\n".join(global_messages), flush=True)
            raise AssertionError(f"Error happens, see logs above")

        physical_to_logical_map = new_physical_to_logical_map


if __name__ == "__main__":
    unittest.main()
