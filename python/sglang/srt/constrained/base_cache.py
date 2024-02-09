"""Base cache class."""

import time


class BaseCache:
    def __init__(self, enable=True):
        self.enable = enable
        self.reset()

    def reset(self):
        self.cache = {}
        self.metrics = {"total": 0, "hit": 0, "avg_init_time": 0}

    def query(self, key):
        def _init_with_timer(key):
            start = time.monotonic()
            val = self.init_value(key)
            init_time = time.monotonic() - start
            curr_total = self.metrics["total"]
            new_total = curr_total + 1

            # Update average init time without old_avg * old_total to avoid overflow.
            self.metrics["avg_init_time"] = (init_time / new_total) + (
                curr_total / new_total
            ) * self.metrics["avg_init_time"]
            return val

        if key in self.cache:
            self.metrics["hit"] += 1
            val = self.cache[key]
        else:
            # Cache miss or disabled.
            val = _init_with_timer(key)

        if self.enable:
            self.metrics["total"] += 1
            self.cache[key] = val
        return val

    def init_value(self, key):
        raise NotImplementedError

    def get_cache_hit_rate(self):
        if self.metrics["total"] == 0:
            return 0
        return self.metrics["hit"] / self.metrics["total"]

    def get_avg_init_time(self):
        return self.metrics["avg_init_time"]
