from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Final, Optional, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.auxiliary_output import CommittedTokens
    from sglang.srt.managers.schedule_batch import ScheduleBatch

CANARY_INFOS_KEY: Final[str] = "canary_infos"


@dataclass(frozen=True, slots=True)
class CanaryInfoDeviceOutput:
    values: torch.Tensor

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> CanaryInfoHostOutput:
        return CanaryInfoHostOutput(values=copy_tensor(self.values))


@dataclass(frozen=True, slots=True)
class CanaryInfoHostOutput:
    values: torch.Tensor

    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None:
        rows = self.values.tolist()
        for req, row, commit in zip(batch.reqs, rows, commits, strict=True):
            if commit is None:
                continue
            num_tokens = len(commit.token_ids)
            if req.customized_info is None:
                req.customized_info = {}
            recorded = req.customized_info.setdefault(CANARY_INFOS_KEY, [])
            assert len(recorded) == commit.output_index
            assert len(row) >= num_tokens
            recorded.extend(row[:num_tokens])
