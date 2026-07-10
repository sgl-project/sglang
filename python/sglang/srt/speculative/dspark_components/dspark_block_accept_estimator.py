from __future__ import annotations

from typing import Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator_online import (
    OnlineBlockAcceptEstimateRecorder,
)

BlockAcceptEstimateRecorder = OnlineBlockAcceptEstimateRecorder


def create_block_accept_estimate_recorder(
    *, gamma: int, device: Union[str, torch.device], tp_rank: int
) -> Optional[BlockAcceptEstimateRecorder]:
    if tp_rank != 0:
        return None

    path = envs.SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH.get()
    online_log_interval = envs.SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL.get()
    if not path and online_log_interval <= 0:
        return None

    return OnlineBlockAcceptEstimateRecorder(
        path=path,
        gamma=gamma,
        device=device,
        online_log_interval=online_log_interval,
    )
