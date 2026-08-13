from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclass(slots=True, kw_only=True)
class NewTokenRatioTracker:
    init: float
    min: float
    decay: float
    current: float

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> NewTokenRatioTracker:
        init = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        min_ratio = min(
            init * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        decay = (init - min_ratio) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        return cls(init=init, min=min_ratio, decay=decay, current=init)

    def decay_step(self) -> None:
        self.current = max(self.current - self.decay, self.min)

    def reset(self) -> None:
        self.current = self.init

    @staticmethod
    def estimate_new_token_ratio_after_retract(reqs: Sequence[Req]) -> float:
        total_decoded_tokens = sum(len(r.output_ids) for r in reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in reqs)

        new_estimate_ratio = (
            total_decoded_tokens + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(reqs)
        ) / (
            total_max_new_tokens + 1
        )  # avoid zero division
        new_estimate_ratio = min(1.0, new_estimate_ratio)
        return new_estimate_ratio
