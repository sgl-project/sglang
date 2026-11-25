"""Bounded Mooncake receive-slot credits."""

import threading
import time


class ReceiveCredits:
    def __init__(self, count):
        self.available = list(range(count))
        self.condition = threading.Condition()

    def acquire(self, count=1):
        deadline = time.monotonic() + 60
        with self.condition:
            while len(self.available) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("KVServe receive credit timed out")
                self.condition.wait(remaining)
            result = self.available[:count]
            del self.available[:count]
            return result

    def release(self, slot):
        with self.condition:
            if slot in self.available:
                raise RuntimeError("Duplicate KVServe receive credit")
            self.available.append(slot)
            self.condition.notify_all()
