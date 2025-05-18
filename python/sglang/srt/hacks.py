import os
import time
from pathlib import Path

import torch
import zmq
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.utils import get_zmq_socket


class OtherProcessKiller:
    def __init__(self):
        base_port = 50000
        num_tasks = 8

        hack_ipc_prefix = os.environ["SGLANG_HACK_IPC_PREFIX"]

        context = zmq.Context(2)
        self.senders = []
        for index in range(num_tasks):
            path = f"/tmp/demo_another_task_{hack_ipc_prefix}_{index}"
            Path(path).write_text("")
            endpoint = f"ipc://{path}"
            print(f"{endpoint=}")
            self.senders.append(get_zmq_socket(context, zmq.PUSH, endpoint, False))

    def kill(self):
        try:
            tp_rank = get_tensor_model_parallel_rank()
        except Exception as e:
            print(f"error getting tp_rank {e=}")
            tp_rank = -1

        # cmd = "pkill -f demo_another_task"
        print(
            f"[Hacks, TP{tp_rank}, {time.time()}] kill_other_memory_occupying_processes start"
        )

        for sender in self.senders:
            sender.send_pyobj("stop")
        # subprocess.run(cmd, shell=True)

        print(
            f"[Hacks, TP{tp_rank}, {time.time()}] kill_other_memory_occupying_processes subprocess end"
        )


def busy_wait_until_enough_memory():
    # TODO overlap
    while True:
        free_memory, _ = torch.cuda.mem_get_info()
        if free_memory > 70_000_000_000:
            break
        time.sleep(0.001)

    print(
        f"[Hacks, TP{get_tensor_model_parallel_rank()}, {time.time()}] busy_wait_until_enough_memory see free memory {free_memory=}"
    )


def export_model_params(model):
    params = []
    for name, param in model.named_parameters():
        param_pin_memory = torch.empty_like(param.data, device="cpu", pin_memory=True)
        param_pin_memory.copy_(param.data)
        params.append((name, param_pin_memory))
    return dict(params=params)


def import_model_param(model, data):
    self_named_params = dict(model.named_parameters())
    for name, tensor in data["params"]:
        self_named_params[name].data.copy_(tensor, non_blocking=True)


if __name__ == "__main__":
    other_process_killer = OtherProcessKiller()

    print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
    print(f"{time.time()=} kill start")
    other_process_killer.kill()
    print(f"{time.time()=} kill end")

    for i in range(5000):
        print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
        time.sleep(0.001)
