import numpy as np
import torch

from sglang.srt.sampling.penalties import StepPenalties


def build_step_penalties(reqs, pool):
    """The step's penalties from each request's index, as the scheduler's
    `SamplingParamsPool` gather gives them."""
    host = np.array(
        [-1 if r.penalty_index is None else r.penalty_index for r in reqs],
        dtype=np.int64,
    )
    indices = torch.tensor(host, dtype=torch.int32, device=pool.device)
    return StepPenalties.build(pool, indices, host)
