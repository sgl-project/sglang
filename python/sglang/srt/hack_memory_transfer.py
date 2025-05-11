import threading
from datetime import datetime

import torch


def run_memory_transfer_experiment():
    thread = threading.Thread(target=_thread_entrypoint)
    thread.start()


def _thread_entrypoint():
    alt_stream = torch.cuda.Stream()
    with torch.cuda.stream(alt_stream):
        tensor_size = 1024 ** 3
        tensor_cpu_pinned = torch.rand((tensor_size,), dtype=torch.uint8, device="cpu", pin_memory=True)
        tensor_output = torch.empty((tensor_size,), dtype=torch.uint8, device="cuda")
        _log(f"{tensor_cpu_pinned.nbytes=} {tensor_output.nbytes=} {tensor_output.device=}")

        while True:
            output_tensor.copy_(input_tensor, non_blocking=True)
            TODO


def _log(msg):
    print(
        f"[memory_transfer_experiment, time={datetime.now().isoformat()}, device={torch.cuda.current_device()}] {msg}")
