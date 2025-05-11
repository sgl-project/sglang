import threading

import torch


def run_memory_transfer_experiment():
    thread = threading.Thread(target=_thread_entrypoint)
    thread.start()


def _thread_entrypoint():
    tensor_size = 1024 ** 3
    tensor_cpu_pinned = torch.rand((tensor_size,), dtype=torch.uint8, device="cpu", pin_memory=True)
    print(f"{tensor_cpu_pinned.nbytes=}")
    TODO
