import subprocess
import time

import torch
from sglang.srt.distributed import get_tensor_model_parallel_rank


def kill_other_memory_occupying_processes():
    cmd = "pkill -f demo_another_task"
    print(
        f"[Hacks, TP{get_tensor_model_parallel_rank()}, {time.time()}] kill_other_memory_occupying_processes start {cmd=}")
    subprocess.run(cmd, shell=True)
    print(
        f"[Hacks, TP{get_tensor_model_parallel_rank()}, {time.time()}] kill_other_memory_occupying_processes subprocess end")


def busy_wait_until_enough_memory():
    # TODO overlap
    while True:
        free_memory, _ = torch.cuda.mem_get_info()
        if free_memory > 70_000_000_000:
            break
        time.sleep(0.001)

    print(
        f"[Hacks, TP{get_tensor_model_parallel_rank()}, {time.time()}] busy_wait_until_enough_memory see free memory {free_memory=}")


def export_model_params(model):
    return dict(params=[(name, param.data.detach().cpu()) for name, param in model.named_parameters()])


def import_model_param(model, data):
    self_named_params = dict(model.named_parameters())
    for name, tensor in data["params"]:
        self_named_params[name].data[...] = tensor.to("cuda")


if __name__ == '__main__':
    print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
    print(f"{time.time()=} kill start")
    kill_other_memory_occupying_processes()
    print(f"{time.time()=} kill end")

    for i in range(5000):
        print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
        time.sleep(0.001)
