from typing import Optional

import torch

from dataclasses import dataclass


@dataclass
class CombineOverlapArgs:
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: int
    signal: Optional[torch.Tensor] = None
    block_m: int = -1
    threshold: int = -1

@dataclass
class DownGemmOverlapArgs:
    TODO
