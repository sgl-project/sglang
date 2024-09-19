from dataclasses import dataclass


@dataclass
class ConfigStats:
    max_total_num_tokens: int
    max_prefill_tokens: int
    max_running_requests: int
    context_len: int