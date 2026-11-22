"""The logit_bias sampling plugin: a dense [bs, vocab] additive bias, built at
each launch from the step's requests."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

import numpy as np
import torch

from sglang.srt.utils.device import async_h2d_tensor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class LogitBiasEntries(NamedTuple):
    """A request's logit_bias, parsed once. Raw bytes, so a launch joins every
    request's entries in one call."""

    token_ids: bytes  # int64
    values: bytes  # fp32

    @classmethod
    def parse(cls, logit_bias: Dict[str, float]) -> LogitBiasEntries:
        # Dedup on int(key) first ("1" and "01" collide): duplicate indices make
        # the index_put in build_logit_bias nondeterministic on CUDA.
        row_bias = {int(key): value for key, value in logit_bias.items()}
        return cls(
            token_ids=array("q", row_bias.keys()).tobytes(),
            values=array("f", row_bias.values()).tobytes(),
        )


def build_logit_bias(
    reqs: List[Req], *, vocab_size: int, device: str
) -> Optional[torch.Tensor]:
    """The step's bias, one row per request; None when no request carries one."""
    if all(r.sampling_params.logit_bias is None for r in reqs):
        return None
    entries = [r.logit_bias_entries for r in reqs]
    rows = [i for i, e in enumerate(entries) if e is not None]
    token_ids, values = zip(*[entries[i] for i in rows])
    logit_bias = torch.zeros(len(reqs), vocab_size, dtype=torch.float32, device=device)
    # Each value is 4 bytes of fp32.
    counts = np.fromiter(map(len, values), dtype=np.int64, count=len(rows)) // 4
    row_ids = np.repeat(np.array(rows, dtype=np.int64), counts)
    n = len(row_ids)
    if n:
        # One copy of 32-bit words: row ids and token ids (int64), then values (fp32).
        block = bytearray().join([row_ids.tobytes(), *token_ids, *values])
        staged = async_h2d_tensor(torch.frombuffer(block, dtype=torch.int32), device)
        row_words, token_id_words, value_words = staged.split([2 * n, 2 * n, n])
        logit_bias.index_put_(
            (row_words.view(torch.int64), token_id_words.view(torch.int64)),
            value_words.view(torch.float32),
        )
    return logit_bias


def apply_logit_bias(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    rows_per_request: int = 1,
) -> None:
    """Add each request's bias to its `rows_per_request` consecutive logits rows."""
    if sampling_info.logit_bias is None:
        return
    # In the bias's own dtype, so the sum rounds once to the logits dtype.
    logits.unflatten(0, (-1, rows_per_request)).add_(
        sampling_info.logit_bias[:, None, :]
    )
