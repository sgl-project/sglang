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

"""Base cache class for constrained decoding tools."""

import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Dict, Tuple


@dataclass
class MapEntry:
    event: Event
    value: Any

    def __iter__(self):
        return iter((self.event, self.value))


class BaseToolCache:

    def __init__(self, enable=True):
        self.enable: bool = enable
        self.cache: Dict[str, MapEntry] = {}
        self.metrics: Dict[str, Any] = {}
        self.lock_cache: Lock = Lock()
        self.lock_metrics: Lock = Lock()
        self.reset()

    def reset(self):
        with self.lock_cache:
            self.cache = {}
        with self.lock_metrics:
            self.metrics = {"total": 0, "hit": 0, "avg_init_time": 0}

    def _init_with_timer(self, key) -> Tuple[Any, float]:
        start = time.monotonic()
        val = self.init_value(key)
        init_time = time.monotonic() - start
        return val, init_time

    def update_time(self, init_time):
        with self.lock_metrics:
            curr_total = self.metrics["total"]
            new_total = curr_total + 1

            # Update average init time without old_avg * old_total to avoid overflow.
            self.metrics["avg_init_time"] = (init_time / new_total) + (
                curr_total / new_total
            ) * self.metrics["avg_init_time"]

    def query(self, key):
        if not self.enable:
            value, init_time = self._init_with_timer(key)
            self.update_time(init_time)
            return value

        with self.lock_cache:
            if key in self.cache:
                entry = self.cache[key]
                cache_hit = True
            else:
                entry = MapEntry(Event(), None)
                self.cache[key] = entry
                cache_hit = False

        with self.lock_metrics:
            self.metrics["total"] += 1
            if cache_hit:
                self.metrics["hit"] += 1

        if cache_hit:
            entry.event.wait()
        else:
            entry.value, init_time = self._init_with_timer(key)
            self.update_time(init_time)
            entry.event.set()
        return entry.value

    def init_value(self, key):
        raise NotImplementedError()

    def get_cache_hit_rate(self):
        with self.lock_metrics:
            return self.metrics["hit"] / max(self.metrics["total"], 1)

    def get_avg_init_time(self):
        with self.lock_metrics:
            return self.metrics["avg_init_time"]
