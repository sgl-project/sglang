# Copyright 2023-2024 SGLang Team
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
"""The baseclass of a backend for grammar-guided constrained decoding."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Optional, Tuple


@dataclass
class CacheEntry:
    value: Any
    event: Event


class BaseGrammarObject:
    pass


class BaseGrammarBackend:
    def __init__(self):
        self.executor = ThreadPoolExecutor()
        self.cache = {}
        self.cache_lock = Lock()

    def init_value(self, key: Tuple[str, str]) -> BaseGrammarObject:
        with self.cache_lock:
            if key in self.cache:
                cache_hit = True
                entry = self.cache[key]
            else:
                cache_hit = False
                entry = CacheEntry(None, Event())
                self.cache[key] = entry

        if cache_hit:
            entry.event.wait()
        else:
            entry.value = self.init_value_impl(key)
            entry.event.set()
        return entry.value.copy() if entry.value else None

    def init_value_impl(self, key: Tuple[str, str]) -> BaseGrammarObject:
        raise NotImplementedError()

    def get_cached_value(self, key: Tuple[str, str]) -> Optional[BaseGrammarObject]:
        with self.cache_lock:
            entry = self.cache.get(key)
            if not entry or not entry.event.is_set():
                return None
            val = self.cache[key].value
            return val.copy() if val else None

    def get_future_value(self, key: Tuple[str, str]) -> Future:
        return self.executor.submit(self.init_value, key)

    def reset(self):
        with self.cache_lock:
            self.cache.clear()
