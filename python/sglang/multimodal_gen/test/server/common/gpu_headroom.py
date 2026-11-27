"""Sample free device memory outside the server process on NVIDIA CI runners."""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid

import pynvml
from cuda.bindings import driver


def visible_gpu_uuids() -> list[str]:
    # resolve the same logical ordinals as the server, including its device mask
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    args, _ = parser.parse_known_args()
    (status,) = driver.cuInit(0)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA initialization failed: {status}")
    uuids = []
    for ordinal in range(args.base_gpu_id, args.base_gpu_id + args.num_gpus):
        status, device = driver.cuDeviceGet(ordinal)
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cannot resolve CUDA device {ordinal}: {status}")
        status, device_uuid = driver.cuDeviceGetUuid(device)
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cannot read GPU UUID: {status}")
        uuids.append(f"GPU-{uuid.UUID(bytes=device_uuid.bytes)}")
    return uuids


class GpuHeadroomSampler:
    def __init__(self, extra_args: str, env_vars: dict[str, str]):
        # a fresh CUDA driver observes child-only environment overrides without
        # creating a device context in pytest or charging it to the benchmark
        uuids = subprocess.check_output(
            [sys.executable, __file__, *shlex.split(extra_args)],
            env=os.environ | env_vars,
            text=True,
        ).splitlines()
        self.samples: list[tuple[int, list[float]]] = []
        self.error: Exception | None = None
        self.stop_event = threading.Event()
        pynvml.nvmlInit()
        try:
            self.handles = [pynvml.nvmlDeviceGetHandleByUUID(u) for u in uuids]
            self._sample()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def _sample(self):
        free_mb = [
            pynvml.nvmlDeviceGetMemoryInfo(h).free / (1 << 20) for h in self.handles
        ]
        self.samples.append((time.monotonic_ns(), free_mb))

    def _run(self):
        try:
            while not self.stop_event.wait(0.2):
                self._sample()
        except Exception as exc:
            self.error = exc

    def close(self):
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()

    def minima(
        self, load_finished_ns: int, server_ready_ns: int
    ) -> dict[str, list[float]]:
        if self.error is not None:
            raise RuntimeError("GPU headroom sampling failed") from self.error
        self._sample()
        minima = {}
        for timestamp, free_mb in self.samples.copy():
            if timestamp <= load_finished_ns:
                phase = "load"
            elif timestamp <= server_ready_ns:
                phase = "warmup"
            else:
                phase = "serving"
            if phase in minima:
                minima[phase] = list(map(min, minima[phase], free_mb))
            else:
                minima[phase] = free_mb
        return minima


if __name__ == "__main__":
    print(*visible_gpu_uuids(), sep="\n")
