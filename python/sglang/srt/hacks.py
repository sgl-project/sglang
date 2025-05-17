import subprocess
import time

import torch


def kill_other_memory_occupying_processes():
    cmd = "pkill -f demo_another_task"
    print(f"[{time.time()=:.3f}] kill_other_memory_occupying_processes start {cmd=}")
    subprocess.run(cmd, shell=True)
    print(f"[{time.time()=:.3f}] kill_other_memory_occupying_processes subprocess end")


def busy_wait_until_enough_memory():
    while True:
        free_memory, _ = torch.cuda.mem_get_info()
        if free_memory > 70_000_000_000:
            break
        time.sleep(0.001)

    print(f"[{time.time()=:.3f}] busy_wait_until_enough_memory see free memory {free_memory=}")


if __name__ == '__main__':
    print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
    print(f"{time.time()=} kill start")
    kill_other_memory_occupying_processes()
    print(f"{time.time()=} kill end")

    for i in range(5000):
        print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
        time.sleep(0.001)
