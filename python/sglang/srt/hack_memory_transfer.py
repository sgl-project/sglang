import threading

import torch


def run_memory_transfer_experiment():
    thread = threading.Thread(target=_thread_entrypoint)
    thread.start()


def _thread_entrypoint():
    tensor_size = 1_000_000_000
    tensor_cpu_pinned = torch.rand((tensor_size,), dtype=torch.uint8, device="cpu", pin_memory=True)
    TODO
