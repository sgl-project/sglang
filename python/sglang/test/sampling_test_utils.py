import numpy as np
import torch

from sglang.srt.sampling.logit_bias import LogitBiasRows
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


def dense_logit_bias_rows(bias: torch.Tensor) -> LogitBiasRows:
    """Every element of a dense [bs, vocab] fp32 bias as an entry, so applying it
    adds the whole tensor."""
    rows, token_ids = torch.meshgrid(
        torch.arange(bias.shape[0], device=bias.device),
        torch.arange(bias.shape[1], device=bias.device),
        indexing="ij",
    )
    return LogitBiasRows(
        rows=rows.flatten(), token_ids=token_ids.flatten(), values=bias.flatten()
    )
