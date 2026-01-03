from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class SimRequest:
    request_id: str
    input_len: int
    output_len: int
    stage: RequestStage = RequestStage.PREFILL
    decoded_tokens: int = 0

    def seq_len(self) -> int:
        if self.stage == RequestStage.PREFILL:
            return self.input_len
        else:
            return self.input_len + self.decoded_tokens

    def is_finished(self) -> bool:
        return self.decoded_tokens >= self.output_len

    def copy(self) -> "SimRequest":
        return SimRequest(
            request_id=self.request_id,
            input_len=self.input_len,
            output_len=self.output_len,
            stage=self.stage,
            decoded_tokens=self.decoded_tokens,
        )

