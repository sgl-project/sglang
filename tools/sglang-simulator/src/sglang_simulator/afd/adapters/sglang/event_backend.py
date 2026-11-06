"""Symbolic device backend: costed compute, paired messages and stream dependencies.

No layer loop, stage split, topology defaults or model geometry live here.
Native orchestration supplies the operations; the backend records their events.
"""

import contextlib
import math
import types

from ...coordinator import EventCoordinator


def make_event_backend(provider):
    clock = EventCoordinator(defer_cycle_checks=True)
    actors, wires, current, active_stream = [], {}, [None], [None]
    c = provider.config
    F = provider.mapping.layout.ffn_ranks
    shapes = provider.mapping.expected_shapes()

    class Actor:
        def __init__(self, role, rank):
            self.role = role
            self.rank = rank
            self.name = role + str(rank)
            self.last = None
            self.position = (0, 0)
            self.streams = {}
            self.packed = set()
            actors.append(self)

        def compute(self, kind, stream=None, **meta):
            data = dict(layer=self.position[0], stage=self.position[1], **meta)
            dep = stream.last if stream else self.last
            e = clock.submit(
                self.name,
                kind,
                provider.quote(kind, role=self.role, **data),
                deps=() if dep is None else (dep,),
                resources=(self.name + ".compute",),
                metadata=data,
            )
            if stream:
                stream.last = e
            else:
                self.last = e
            return e

        def wait(self, event):
            self.last = clock.submit(
                self.name,
                "stream_wait",
                0,
                deps=tuple((x for x in (self.last, event.eid) if x is not None)),
            )

    class Stream:
        def __init__(self, actor, stage, direction):
            self.actor = actor
            self.stage = stage
            self.direction = direction
            self.last = None

    class Event:
        def __init__(self):
            self.eid = None

        def record(self, stream):
            self.eid = stream.last

    @contextlib.contextmanager
    def stream_context(stream):
        before = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = before

    class Tensor:
        def __init__(self, shape, dtype="BF16", label=""):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.label = label
            self.device = "cpu-symbolic"

        @property
        def ndim(self):
            return len(self.shape)

        def numel(self):
            return math.prod(self.shape)

        def element_size(self):
            return {"BF16": 2, "I64": 8, "F32": 4}[self.dtype]

        def __len__(self):
            return self.shape[0]

        def contiguous(self):
            return self

        def to(self, dtype):
            return Tensor(self.shape, dtype)

        def float(self):
            return self.to("F32")

        def __iter__(self):
            return iter(
                [Tensor(self.shape[1:], self.dtype) for _ in range(self.shape[0])]
            )

        def __getitem__(self, key):
            shape = list(self.shape)
            keys = key if isinstance(key, tuple) else (key,)
            for axis, k in reversed(list(enumerate(keys))):
                if isinstance(k, int):
                    shape.pop(axis)
                elif isinstance(k, slice):
                    shape[axis] = len(range(*k.indices(shape[axis])))
            return Tensor(shape, self.dtype, self.label)

        def zero_(self):
            a = current[0]
            if self.label == "packed_hidden" and a.position not in a.packed:
                a.packed.add(a.position)
                a.compute("direct_pack", rows=self.shape[0])
            return self

        def fill_(self, value):
            return self

        def copy_(self, other):
            return self

        def record_stream(self, stream):
            pass

        def flatten(self, start, end):
            return Tensor(
                (math.prod(self.shape[start : end + 1]), *self.shape[end + 1 :]),
                self.dtype,
            )

        def view(self, *shape):
            return Tensor(shape, self.dtype)

        def sum(self, dim, dtype):
            stream = active_stream[0]
            stream.actor.compute("direct_combine", stream=stream, rows=self.shape[1])
            return Tensor(self.shape[1:], dtype)

    def post(stream, peer, fields, send):
        a = stream.actor
        f = peer if a.role == "A" else a.rank
        ar = a.rank if a.role == "A" else peer - F
        key = (stream.direction, ar, f, *a.position)
        kind = "direct_A2F_message" if stream.direction == 0 else "direct_F2A_message"
        bytecount = sum((t.numel() * t.element_size() for t in fields))
        expected = shapes[kind]["bytes"]
        if bytecount != expected:
            raise ValueError(
                "DIRECT_PAYLOAD_BYTES_MISMATCH " + str((bytecount, expected))
            )
        if key not in wires:
            wires[key] = dict(
                event=clock.reserve(
                    "wire",
                    kind,
                    metadata=dict(
                        A=ar,
                        F=f,
                        layer=a.position[0],
                        stage=a.position[1],
                        bytes=bytecount,
                    ),
                ),
                posts={},
            )
        wire = wires[key]
        endpoint = (a.role, "send" if send else "recv")
        if endpoint in wire["posts"]:
            raise ValueError("DUPLICATE_DIRECT_ENDPOINT")
        wire["posts"][endpoint] = stream.last
        if len(wire["posts"]) == 2:
            if {x[1] for x in wire["posts"]} != {"send", "recv"}:
                raise ValueError("DIRECT_ENDPOINT_DIRECTION")
            resources = (f"link.A{ar}.F{f}.{stream.direction}",)
            if c["direct_resource_policy"] == "shared_endpoint_nic":
                resources += (
                    f"A{ar}.nic.{stream.direction}",
                    f"F{f}.nic.{stream.direction}",
                )
            clock.resolve(
                wire["event"],
                deps=tuple((x for x in wire["posts"].values() if x is not None)),
                duration_ns=provider.quote(kind, bytes=bytecount),
                resources=resources,
            )
        return wire["event"]

    class Binding:
        def __init__(self, stream):
            self._stream = stream

        def invoke(self, torch, operation):
            stream = self._stream
            a = stream.actor
            sent = []
            pending = {}

            class Comm:
                def group_start(self):
                    stream.last = clock.submit(
                        a.name,
                        "direct_group_startup",
                        provider.quote("direct_group_startup"),
                        deps=tuple((x for x in (stream.last, a.last) if x is not None)),
                        resources=(a.name + ".group." + str(stream.direction),),
                        metadata=dict(layer=a.position[0], stage=a.position[1]),
                    )

                def send(self, tensor, dst):
                    pending.setdefault((dst, True), []).append(tensor)

                def recv(self, tensor, src):
                    pending.setdefault((src, False), []).append(tensor)

                def group_end(self):
                    for (peer, send), fields in pending.items():
                        sent.append(post(stream, peer, fields, send))
                    stream.last = clock.submit(
                        a.name, "group_completion", 0, deps=tuple(sent)
                    )

            operation(Comm())

    class Transport:
        def __init__(self, a):
            self.actor = a
            self.ffn_backend = None

        def _fork(self, stream):
            stream.last = clock.submit(
                self.actor.name,
                "stream_fork",
                0,
                deps=tuple(
                    (x for x in (stream.last, self.actor.last) if x is not None)
                ),
            )

        def wait(self, event):
            self.actor.wait(event)

    def empty(shape, *, dtype="BF16", device=None):
        return Tensor(shape, dtype)

    torch = types.ModuleType("torch")
    torch.bfloat16, torch.int64, torch.float32 = "BF16", "I64", "F32"
    torch.empty = empty
    torch.empty_like = lambda value, dtype=None: Tensor(
        value.shape, dtype or value.dtype
    )
    torch.cat = lambda values, dim=0: Tensor(
        tuple(
            sum(v.shape[i] for v in values) if i == dim else values[0].shape[i]
            for i in range(values[0].ndim)
        ),
        values[0].dtype,
    )
    torch.cuda = types.SimpleNamespace(Event=Event, stream=stream_context)
    return types.SimpleNamespace(
        clock=clock,
        actors=actors,
        wires=wires,
        current=current,
        Actor=Actor,
        Stream=Stream,
        Binding=Binding,
        Transport=Transport,
        Tensor=Tensor,
        torch=torch,
    )
