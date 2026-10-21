# SPDX-License-Identifier: Apache-2.0
"""Private cache-owner references between publication and request admission."""

from __future__ import annotations

import threading

from sglang.srt.managers.kv_control.contracts import ControlError, CopyTo, Location
from sglang.srt.managers.kv_control.controller import ExecutionRequest
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

FULL = ComponentType.FULL


class PreparedPrefix:
    """A bounded internal owner lease, never a public cache retention primitive.

    Register before starting a private GPU COPY. The transfer owner captures the
    complete published GPU scope before releasing its own correctness references.
    The scheduling owner must release this lease after the request takes its own
    cache lock, or on abort, deferral, expiry and task cleanup.
    """

    def __init__(self, executor, ref, pending):
        self.executor, self.cache = executor, executor.cache
        self.ref = ref
        self.work = ExecutionRequest(
            ref=ref,
            target=pending.target,
            payload=CopyTo(
                target=pending.payload.target,
                src=Location(tier="host"),
                dst=Location(tier="gpu"),
            ),
            origin="policy",
        )
        executor.transfers._prefixes(self.work)
        self.locks = []
        self.ready = False
        self.released = False
        self.locked_tokens = 0
        self.code = "prefix_not_ready"
        self.reason = ""

    def capture(self):
        if self.released:
            return False
        self.executor._check(self.work.target)
        if self.ready:
            return True
        acquired = []
        try:
            paths = self.executor.transfers._paths(self.work, source=False)
            nodes = {n.id: n for _, _, rows in paths for n, _, _ in rows}
            if any(
                n.component_data[FULL].value is None
                or n.component_data[FULL].value.numel() != len(n.key)
                or self.executor.transfers._load_pending(n)
                for n in nodes.values()
            ):
                self.code = "prefix_not_ready"
                return False
            anchors = {rows[-1][0].id for _, _, rows in paths}
            for anchor in anchors:
                acquired.append(self.cache.lock(anchor))
            remaining = self.executor.transfers._gpu_available()
            if type(remaining) is not int or remaining < 0:
                self.code = "preparation_capacity"
                return False
            self.locks, acquired = acquired, []
            self.locked_tokens = sum(len(n.key) for n in nodes.values())
            self.ready, self.code = True, "preparation_ready"
            self.reason = ""
            return True
        except ControlError as error:
            self.code = error.code
            self.reason = str(error)[:512]
            return False
        except Exception as error:
            self.code = "preparation_error"
            self.reason = f"{type(error).__name__}: {error}"[:512]
            return False
        finally:
            for lock in reversed(acquired):
                self.cache.unlock(lock)

    def release(self):
        if threading.get_ident() != self.executor._thread:
            raise ControlError("wrong_owner", "Release preparation on the cache owner")
        if self.released:
            return
        for lock in reversed(self.locks):
            self.cache.unlock(lock)
        self.locks.clear()
        self.ready, self.released, self.locked_tokens = False, True, 0
        self.code, self.reason = "preparation_released", ""
        self.executor.preparations.pop(self.ref.operation_id, None)
