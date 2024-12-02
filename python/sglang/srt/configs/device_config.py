import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device: str = "cuda") -> None:
        if device in ["cuda", "xpu", "hpu"]:
            self.device_type = device
        else:
            raise RuntimeError(f"Not supported device type: {device}")
        self.device = torch.device(self.device_type)
