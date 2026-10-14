from __future__ import annotations

import os


def require_single_visible_gpu(physical_gpu: int | None = None) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly one free physical GPU before "
            "running this benchmark."
        )
    if physical_gpu is not None and devices[0].isdigit():
        selected = int(devices[0])
        if selected != physical_gpu:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES selects GPU {selected}, but "
                f"--physical-gpu is {physical_gpu}."
            )
    return devices[0]
