from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator_offline import (
    OfflineBlockAcceptEstimateRecorder,
)
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator_online import (
    OnlineBlockAcceptEstimateRecorder,
)

logger = logging.getLogger(__name__)

BlockAcceptEstimateRecorder = Union[
    OnlineBlockAcceptEstimateRecorder, OfflineBlockAcceptEstimateRecorder
]


def create_block_accept_estimate_recorder(
    *, gamma: int, device: Union[str, torch.device], tp_rank: int
) -> Optional[BlockAcceptEstimateRecorder]:
    if tp_rank != 0:
        return None

    mode = envs.SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_MODE.get()
    path = envs.SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH.get()
    online_log_interval = envs.SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL.get()

    if not mode:
        if path or online_log_interval > 0:
            mode = "online"
        else:
            return None

    if mode == "online":
        return OnlineBlockAcceptEstimateRecorder(
            path=path,
            gamma=gamma,
            device=device,
            online_log_interval=online_log_interval,
        )
    if mode == "offline":
        if not path:
            raise ValueError(
                "offline block accept estimator requires "
                "SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH to be set"
            )
        return OfflineBlockAcceptEstimateRecorder(path=path, gamma=gamma)
    raise ValueError(
        f"unknown SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_MODE={mode!r} "
        "(expected '', 'online', or 'offline')"
    )
