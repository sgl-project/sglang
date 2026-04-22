import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps"]


class DeviceConfig:
    device: Optional[torch.device]
    gpu_id: Optional[int]

    def __init__(self, device: str = "cuda", gpu_id: int = -1) -> None:
        if device in SUPPORTED_DEVICES:
            self.device_type = device
        else:
            raise RuntimeError(f"Not supported device type: {device}")
        self.device = torch.device(self.device_type)
        self.gpu_id = gpu_id
