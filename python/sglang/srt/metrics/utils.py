# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Prometheus Metrics."""
import math
from typing import List


def two_sides_exponential_buckets(
    middle: float, width: float, count: int
) -> List[float]:
    buckets = []
    half_count = math.ceil(count / 2)
    distance = 1
    buckets.append(middle)
    for i in range(half_count):
        distance *= width
        buckets.append(middle + distance)
        buckets.append(max(0, middle - distance))
    return sorted(set(buckets))


def generate_buckets(
    prompt_tokens_buckets: List[str], default_buckets: List[float]
) -> List[float]:
    if not prompt_tokens_buckets:
        prompt_tokens_buckets = ["default"]

    assert len(prompt_tokens_buckets) > 0
    rule = prompt_tokens_buckets[0]
    if rule == "tse":
        middle, width, count = prompt_tokens_buckets[1:]
        return two_sides_exponential_buckets(float(middle), float(width), int(count))
    if rule == "default":
        return sorted(set(default_buckets))
    if rule == "customer":
        return [float(x) for x in prompt_tokens_buckets[1:]]
