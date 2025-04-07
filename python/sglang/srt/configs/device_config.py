import logging
from typing import Optional

import torch

from sglang.srt.platforms import current_platform

logger = logging.getLogger(__name__)


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device_type: str = current_platform.device_type) -> None:
        self.device_type = device_type
        self.device = torch.device(self.device_type)
