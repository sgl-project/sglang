from dataclasses import dataclass
from typing import List


@dataclass
class ConfigStats:
    # Model config
    max_total_num_tokens: int
    max_prefill_tokens: int
    max_running_requests: int
    context_len: int

@dataclass
class PrefillStats:
    # Request stats
    #   TODO Latency
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    # best_of_requests: List[int]
    # n_requests: List[int]
    finished_reason_requests: List[str]

@dataclass
class DecodeStats:
    # System stats (should have _sys suffix)
    #   Scheduler State
    num_running_sys: int
    num_waiting_sys: int
    gen_throughput: int
    token_usage: int

@dataclass
class SystemStats:
    #   KV Cache Usage in %
    # gpu_cache_usage_sys: float
    #   Prefix caching block hit rate
    new_seq:int
    new_token:int
    cached_token:int
    cache_hit_rate:float
    running_req:int
    queue_req:int
# TODO Iteration stats