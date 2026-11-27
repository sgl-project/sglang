"""The GPU architecture a TOP process runs; the trainer and every engine must share it.

FA4, DeepGEMM and the JIT build kernels per architecture, so one contract on two
architectures runs two programs. Each scheduler under a contract reports its
device in ``get_internal_state`` (``/server_info`` per DP rank) under
``DEVICE_STATE_KEY``; the trainer compares it with its own report.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

DEVICE_STATE_KEY = "true_on_policy_device"


def device_report(device: Optional[int] = None) -> Dict[str, Any]:
    major, minor = torch.cuda.get_device_capability(device)
    return {"capability": [major, minor], "name": torch.cuda.get_device_name(device)}


def sm_name(capability: Sequence[int]) -> str:
    major, minor = capability
    return f"SM{major}{minor}"
