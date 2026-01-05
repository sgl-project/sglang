from dataclasses import dataclass
from typing import Optional


@dataclass
class SimRequest:
    request_id: str
    input_len: int
    output_len: int
    decoded_tokens: int = 0
    group_id: Optional[str] = None
    prefix_len: int = 0

    def seq_len(self) -> int:
        return self.input_len + self.decoded_tokens

    def is_finished(self) -> bool:
        return self.decoded_tokens >= self.output_len
