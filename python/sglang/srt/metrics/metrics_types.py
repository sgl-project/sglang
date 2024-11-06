"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Metrics Types"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Stats:
    # config
    max_total_num_tokens: int = 0
    max_prefill_tokens: int = 0
    max_running_requests: int = 0
    context_len: int = 0
    # request stats
    num_prompt_tokens_requests: List[int] = field(default_factory=list)
    num_generation_tokens_requests: List[int] = field(default_factory=list)
    finished_reason_requests: List[str] = field(default_factory=list)
    # decode stats
    num_running_req: int = 0
    num_waiting_req: int = 0
    gen_throughput: float = 0.0
    num_token: int = 0
    token_usage: float = 0.0
    waiting_queue: int = 0
    time_e2e_requests: List[float] = field(default_factory=list)
    time_waiting_requests: List[float] = field(default_factory=list)
    time_decode_requests: List[float] = field(default_factory=list)
    # system stats
    token_usage: float = 0.0
    is_mixed_chunk: bool = False
    new_seq: int = 0
    new_token: int = 0
    cached_token: int = 0
    cache_hit_rate: float = 0.0
    running_req: int = 0
    queue_req: int = 0

    # Iteration stats (should have _iter suffix)
    num_prompt_tokens_iter: int = 0
    num_generation_tokens_iter: int = 0
    time_to_first_tokens_iter: List[float] = field(default_factory=list)
    time_per_output_tokens_iter: List[float] = field(default_factory=list)
