from __future__ import annotations

from dataclasses import dataclass

@dataclass
class PeoOverlapArgs:
    use_expert_overlap: bool
    num_rounds: int
    round_id: int
    send_num_sms: int
    recv_num_sms: int
    hook_use_comm_stream: bool = False
    is_x_in_round: bool = False
