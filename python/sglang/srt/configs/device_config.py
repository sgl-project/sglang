import logging
from typing import Optional

import torch

from sglang.srt.platforms import current_platform, resolve_available_platforms

logger = logging.getLogger(__name__)


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device: str = current_platform.device_type) -> None:
        if device in resolve_available_platforms():
            self.device_type = device
        else:
            raise RuntimeError(f"Not supported device type: {device}")
        self.device = torch.device(self.device_type)
