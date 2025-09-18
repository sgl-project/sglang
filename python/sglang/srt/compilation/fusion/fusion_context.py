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

import logging
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FusionPassStats:
    total_count: int
    total_time: int


_fusion_context = None


class FusionContext:
    def __init__(self):
        self.stats: dict[str, FusionPassStats] = {}

    def record_stats(self, pass_name: str, count: int, time: int):
        if pass_name in self.stats:
            self.stats[pass_name].total_count += count
            self.stats[pass_name].total_time += time
        else:
            self.stats[pass_name] = FusionPassStats(count, time)

    def log_stats(self):
        for pass_name, stats in self.stats.items():
            duration_ms = float(stats.total_time) / 1.0e6
            logger.debug(
                "%s completed in %.1f ms, matched %s times",
                pass_name,
                duration_ms,
                stats.total_count,
            )


def get_fusion_context() -> FusionContext:
    assert _fusion_context is not None
    return _fusion_context


@contextmanager
def fusion_context():
    global _fusion_context
    prev_context = _fusion_context
    _fusion_context = FusionContext()
    try:
        yield
    finally:
        _fusion_context = prev_context
