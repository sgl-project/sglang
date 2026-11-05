"""Pure integer-nanosecond DAG scheduling; no runtime or model integration.

Resource IDs are hashable, unit-capacity tokens shared across all actors/domains.
Callers supply stream dependencies and FIFO sequence ordinals explicitly. Matching
uses the complete communication key, never layer metadata or arrival sorting.
Times are recomputed after graph changes; consume them only once inputs are ready.
"""

from heapq import heapify, heappop, heappush


class DeadlockError(RuntimeError):
    """Unscheduled events, including unmatched receive keys, in ``wait_graph``."""

    def __init__(self, wait_graph):
        self.wait_graph = wait_graph
        super().__init__(f"Unresolved event wait graph: {wait_graph!r}")


class EventCoordinator:
    def __init__(self, *, defer_cycle_checks=False):
        # Complete-graph adapters can avoid repeated transitive DFS while
        # resolving many forward receive handles. drain still rejects cycles;
        # no partial readiness may be observed in this mode.
        if type(defer_cycle_checks) is not bool:
            raise ValueError("defer_cycle_checks must be a bool")
        self._defer_cycle_checks = defer_cycle_checks
        self._events = []
        self._unresolved = set()
        self._messages = {"send": {}, "recv": {}}
        self._pending_receives = {}
        self._timed_events = []
        self._max_referenced_id = -1
        self._resolved_events = set()
        self._committed_times = []

    def is_resolved(self, event_id):
        if self._defer_cycle_checks:
            raise ValueError(
                "Complete-graph mode requires drain/finish before observation"
            )
        self._event(event_id)
        pending = [(event_id, False)]
        while pending:
            current, visited = pending.pop()
            if current in self._resolved_events:
                continue
            event = self._events[current]
            if event["duration"] is None:
                return False
            if visited:
                self._resolved_events.add(current)
            else:
                pending.append((current, True))
                pending.extend(
                    (dep, False)
                    for dep in event["deps"]
                    if dep not in self._resolved_events
                )
        return True

    def _event(self, event_id):
        if type(event_id) is not int or not 0 <= event_id < len(self._events):
            raise ValueError(f"Missing event/dependency: {event_id!r}")
        return self._events[event_id]

    def _validate(self, duration_ns, deps, resources):
        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("duration_ns must be a non-negative integer")
        deps = tuple(deps)
        for dep in deps:
            self._event(dep)
        return tuple(dict.fromkeys(deps)), tuple(dict.fromkeys(resources))

    def _check_cycle(self, event_id, deps):
        if event_id in deps:
            raise ValueError(f"Dependency cycle through event {event_id}")
        if self._defer_cycle_checks:
            return
        # No existing path can reach an ID beyond all recorded dependency IDs.
        # This makes ordinary appends cheap, but not the unsafe assumption that
        # a newest reserved handle cannot already have forward incoming edges.
        if event_id > self._max_referenced_id:
            return
        stack, seen = list(deps), set()
        while stack:
            dep = stack.pop()
            if dep == event_id:
                raise ValueError(f"Dependency cycle through event {event_id}")
            if dep not in seen:
                seen.add(dep)
                if not self.is_resolved(dep):
                    stack.extend(self._events[dep]["deps"])

    def _invalidate(self):
        for event_id in self._timed_events:
            event = self._events[event_id]
            event["start_ns"] = event["finish_ns"] = None
        self._timed_events.clear()

    def reserve(self, actor, kind, domain="", metadata=None):
        event_id = len(self._events)
        self._events.append(
            dict(
                id=event_id,
                actor=actor,
                kind=kind,
                domain=domain,
                metadata=dict(metadata or {}),
                deps=(),
                resources=(),
                duration=None,
                start_ns=None,
                finish_ns=None,
            )
        )
        self._unresolved.add(event_id)
        self._invalidate()
        return event_id

    def submit(
        self, actor, kind, duration_ns, resources=(), deps=(), metadata=None, domain=""
    ):
        deps, resources = self._validate(duration_ns, deps, resources)
        event_id = self.reserve(actor, kind, domain, metadata)
        self.resolve(event_id, deps, duration_ns, resources)
        return event_id

    def resolve(self, event_id, deps=(), duration_ns=0, resources=()):
        event = self._event(event_id)
        if event["duration"] is not None:
            raise ValueError(f"Event {event_id} already resolved")
        if event_id in self._pending_receives:
            raise ValueError("Receive requires a matching send")
        deps, resources = self._validate(duration_ns, deps, resources)
        self._check_cycle(event_id, deps)
        event.update(deps=deps, duration=duration_ns, resources=resources)
        self._unresolved.remove(event_id)
        self._max_referenced_id = max(self._max_referenced_id, max(deps, default=-1))
        self._invalidate()

    def send(
        self, domain, channel, source, dest, seq, deps=(), duration_ns=0, resources=()
    ):
        return self._transfer(
            "send", domain, channel, source, dest, seq, deps, duration_ns, resources
        )

    def recv(
        self, domain, channel, source, dest, seq, deps=(), duration_ns=0, resources=()
    ):
        return self._transfer(
            "recv", domain, channel, source, dest, seq, deps, duration_ns, resources
        )

    def _transfer(
        self, kind, domain, channel, source, dest, seq, deps, duration_ns, resources
    ):
        key = (domain, channel, source, dest, seq)
        own = self._messages[kind]
        if key in own:
            raise ValueError(f"Duplicate {kind}: {key!r}")
        deps, resources = self._validate(duration_ns, deps, resources)
        peer = self._messages["recv" if kind == "send" else "send"].get(key)
        if kind == "send" and peer is not None:
            self._check_cycle(peer, deps)  # Reject before mutating either handle.
        metadata = {"match_key": key}
        actor = source if kind == "send" else dest
        if kind == "recv" and peer is None:
            event_id = self.reserve(actor, kind, domain, metadata)
            self._events[event_id].update(deps=deps, resources=resources)
            self._max_referenced_id = max(
                self._max_referenced_id, max(deps, default=-1)
            )
            self._pending_receives[event_id] = duration_ns
        else:
            if kind == "recv":
                deps += (peer,)
            event_id = self.submit(
                actor, kind, duration_ns, resources, deps, metadata, domain
            )
        own[key] = event_id
        if kind == "send" and peer is not None:
            event = self._events[peer]
            duration = self._pending_receives.pop(peer)
            self.resolve(
                peer, event["deps"] + (event_id,), duration, event["resources"]
            )
        return event_id

    def drain(self):
        """Schedule all possible events; raise with the remaining wait graph."""
        self._invalidate()
        available = {}
        dependents = [[] for _ in self._events]
        remaining = [len(event["deps"]) for event in self._events]
        dependency_finish = [0] * len(self._events)
        ready = []
        for event in self._events:
            for dep in event["deps"]:
                dependents[dep].append(event["id"])
            if not event["deps"] and event["duration"] is not None:
                ready.append((0, event["id"]))
        heapify(ready)
        while ready:
            earliest, event_id = heappop(ready)
            event = self._events[event_id]
            start = max(
                earliest,
                max((available.get(r, 0) for r in event["resources"]), default=0),
            )
            # Availability only increases: queued times are lower bounds. A
            # stale candidate must be requeued until its full (start, id) key
            # beats every other lower bound, including newly ready events.
            if ready and (start, event_id) > ready[0]:
                heappush(ready, (start, event_id))
                continue
            finish = start + event["duration"]
            event.update(start_ns=start, finish_ns=finish)
            self._timed_events.append(event_id)
            for resource in event["resources"]:
                available[resource] = finish
            for dependent in dependents[event_id]:
                remaining[dependent] -= 1
                dependency_finish[dependent] = max(dependency_finish[dependent], finish)
                if (
                    remaining[dependent] == 0
                    and self._events[dependent]["duration"] is not None
                ):
                    heappush(ready, (dependency_finish[dependent], dependent))
        if len(self._timed_events) != len(self._events):
            raise DeadlockError(
                {
                    i: dict(
                        deps=[
                            d
                            for d in event["deps"]
                            if self._events[d]["finish_ns"] is None
                        ],
                        unresolved=event["duration"] is None,
                        actor=event["actor"],
                        kind=event["kind"],
                        domain=event["domain"],
                        metadata=dict(event["metadata"]),
                    )
                    for i, event in enumerate(self._events)
                    if event["finish_ns"] is None
                }
            )

    @property
    def has_unresolved(self):
        return bool(self._unresolved)

    def checkpoint(self, event_ids):
        """Commit a complete DP step; future steps must depend on this fence.

        Called only when every A rank has returned from the native pipeline.
        Rejects incomplete communication and any rewrite of an observed time.
        """
        fence = self.submit("DP-host", "decode_step_fence", 0, deps=event_ids)
        self.drain()
        for event, expected in zip(self._events, self._committed_times):
            if (event["start_ns"], event["finish_ns"]) != expected:
                raise RuntimeError("ONLINE_CLOCK_HISTORY_REWRITTEN")
        self._committed_times = [(e["start_ns"], e["finish_ns"]) for e in self._events]
        return fence, self._events[fence]["finish_ns"]

    def finish(self, event_id):
        self._event(event_id)
        self.drain()
        return self._events[event_id]["finish_ns"]

    def export(self):
        """Return detached records; times are None until drain (or after edits)."""
        return [
            dict(
                e,
                deps=list(e["deps"]),
                resources=list(e["resources"]),
                metadata=dict(e["metadata"]),
            )
            for e in self._events
        ]

    @property
    def events(self):
        return self.export()
