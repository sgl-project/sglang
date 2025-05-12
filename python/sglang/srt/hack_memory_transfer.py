import threading
import time
from datetime import datetime

import torch


def run_memory_transfer_experiment():
    thread = threading.Thread(target=_thread_entrypoint)
    thread.start()


def _thread_entrypoint():
    num_repeat = 100

    alt_stream = torch.cuda.Stream()
    with torch.cuda.stream(alt_stream):
        tensor_size = 1024**3
        tensor_cpu_pinned = torch.rand(
            (tensor_size,), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        tensor_output = torch.empty((tensor_size,), dtype=torch.uint8, device="cuda")
        _log(
            f"startup {tensor_cpu_pinned.nbytes=} {tensor_output.nbytes=} {tensor_output.device=}"
        )

        while True:
            torch.cuda.synchronize()
            t_start = time.time()

            for _ in range(num_repeat):
                tensor_output.copy_(tensor_cpu_pinned, non_blocking=True)

            torch.cuda.synchronize()
            t_end = time.time()
            t_delta = t_end - t_start
            bandwidth = tensor_size * num_repeat / t_delta / 1e9
            _log(f"time={t_delta:.3f} bandwidth={bandwidth}GB/s")


def _log(msg):
    print(
        f"[memory_transfer_experiment, time={datetime.now().isoformat()}, device={torch.cuda.current_device()}] {msg}"
    )
