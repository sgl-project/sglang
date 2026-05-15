import logging
from typing import Optional

import torch

from sglang.srt.platforms import current_platform

logger = logging.getLogger(__name__)

SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps"]


class DeviceConfig:
    device: Optional[torch.device]
    gpu_id: Optional[int]

    def __init__(self, device: str = "cuda", gpu_id: int = -1) -> None:
        if device in SUPPORTED_DEVICES:
            self.device_type = device
        else:
            # TODO: Fold OOT device validation into the platform abstraction.
            if (
                current_platform.is_out_of_tree()
                and device == current_platform.device_name
            ):
                self.device_type = current_platform.device_type
            else:
                raise RuntimeError(f"Not supported device type: {device}")
        self.device = torch.device(self.device_type)
        self.gpu_id = gpu_id
