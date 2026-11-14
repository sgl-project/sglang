"""Compare long CPU token prefixes without allocating repeated array slices."""

import inspect
from array import array
from functools import wraps

import numpy as np

from sglang_simulator.hook import BaseHook


class C_RadixKeyHook(BaseHook):
    HOOK_CLASS_NAME = "RadixKey"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.radix_cache"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original = getattr(target, "match_at", None)
        if not callable(original) or tuple(inspect.signature(original).parameters) != (
            "self",
            "other",
            "offset",
            "page_size",
        ):
            return

        @wraps(original)
        def match_at(self, other, offset, page_size=1):
            if (
                type(self) is not target
                or type(other) is not target
                or type(offset) is not int
                or type(page_size) is not int
                or type(self.is_bigram) is not bool
                or type(other.is_bigram) is not bool
                or type(self.token_ids) is not array
                or type(other.token_ids) is not array
                or self.token_ids.typecode != "q"
                or other.token_ids.typecode != "q"
                or self.token_ids.itemsize != 8
                or other.token_ids.itemsize != 8
                or offset < 0
                or offset > len(other)
            ):
                return original(self, other, offset, page_size)
            size = min(self._raw_len(), other._raw_len() - offset)
            # Preserve the cheap divergent-prefix path. This threshold selects
            # an algorithm by input size, never a workload, model or topology.
            if (
                size < 4096
                or self.token_ids[:64] != other.token_ids[offset : offset + 64]
            ):
                return original(self, other, offset, page_size)
            self._check_compatible(other)
            if self.is_bigram != other.is_bigram:
                return original(self, other, offset, page_size)
            # frombuffer borrows both arrays only for this synchronous call.
            # No view or derived key escapes, and no request/tree state changes.
            left = np.frombuffer(
                self.token_ids, dtype=np.int64, count=size - 64, offset=64 * 8
            )
            right = np.frombuffer(
                other.token_ids,
                dtype=np.int64,
                count=size - 64,
                offset=(offset + 64) * 8,
            )
            differs = left != right
            first = int(differs.argmax())
            matched = first + 64 if differs[first] else size
            if self.is_bigram:
                matched = max(0, min(matched - 1, len(self), len(other) - offset))
                return (matched // page_size) * page_size if page_size > 1 else matched
            matched = min(matched, len(self), len(other) - offset)
            return matched if page_size == 1 else (matched // page_size) * page_size

        target.match_at = match_at
