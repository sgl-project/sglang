import subprocess
import time


def kill_other_memory_occupying_processes():
    cmd = "pkill -f demo_another_task"
    print(f"kill_other_memory_occupying_processes {cmd=}")
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    import torch

    print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
    print(f"{time.time()=} kill start")
    kill_other_memory_occupying_processes()
    print(f"{time.time()=} kill end")

    for i in range(5000):
        print(f"{time.time()=} {torch.cuda.mem_get_info()=}")
        time.sleep(0.001)
