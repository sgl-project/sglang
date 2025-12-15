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
    middle: float, base: float, count: int
) -> List[float]:
    buckets = []
    half_count = math.ceil(count / 2)
    distance = 1
    buckets.append(middle)
    for i in range(half_count):
        distance *= base
        buckets.append(middle + distance)
        buckets.append(max(0, middle - distance))
    return sorted(set(buckets))


def generate_buckets(
    buckets_rule: List[str], default_buckets: List[float]
) -> List[float]:
    if not buckets_rule:
        buckets_rule = ["default"]

    assert len(buckets_rule) > 0
    rule = buckets_rule[0]
    if rule == "tse":
        middle, base, count = buckets_rule[1:]
        assert float(base) > 1.0, "Base must be greater than 1.0"
        return two_sides_exponential_buckets(float(middle), float(base), int(count))
    if rule == "default":
        return sorted(set(default_buckets))
    assert rule == "custom"
    return sorted(set([float(x) for x in buckets_rule[1:]]))


def exponential_buckets(start: float, width: float, length: int) -> List[float]:
    buckets = []
    for i in range(length):
        buckets.append(start * (width**i))
    return buckets
