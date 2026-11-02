"""Native Direct control with symbolic tensors; conditional costs only.

Runs unchanged attention _run_layers, Direct dispatch/_exchange/run_ffn_step/
run_layer and native two-stage executor. GPU initialization is outside scope.
"""

import contextlib
import copy
import dataclasses
import functools
import math
import types
import typing
from pathlib import Path

from ...contracts import ExecutionTrace
from ...coordinator import EventCoordinator
from .loader import extract, temporary_modules
from .qwen_profile import QwenCostMapping
from .source import verify_source

BASE_FIELDS = ("model", "mode", "ffn_ranks", "batch_size", "effective_kv", "layers")


class DirectMapping:
    """Qwen-specific signatures; no model defaults enter the generic provider."""

    def __init__(self, c):
        required = set(BASE_FIELDS) | {
            "routing",
            "direct_group_alpha_ns",
            "direct_link_bandwidth_Bps",
            "direct_resource_policy",
        }
        optional = {"ffn_hardware", "ffn_recipe"}
        if not required <= c.keys() or c.keys() - required - optional:
            raise ValueError("DIRECT_PROFILE_CONFIG_UNSUPPORTED")
        if c["routing"] != "direct":
            raise ValueError("DIRECT_ROUTING_MISMATCH")
        if type(c["ffn_ranks"]) is not int:
            raise ValueError("DIRECT_PROFILE_WIDTH")
        for name in ("direct_group_alpha_ns", "direct_link_bandwidth_Bps"):
            value = c[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value))
                or (value <= 0)
            ):
                raise ValueError("DIRECT_PROFILE_PARAMETER_INVALID: " + name)
        if c["direct_resource_policy"] not in (
            "shared_endpoint_nic",
            "independent_links",
        ):
            raise ValueError("DIRECT_RESOURCE_POLICY_UNSUPPORTED")
        for name in optional & c.keys():
            if not isinstance(c[name], str) or not c[name]:
                raise ValueError("DIRECT_PROFILE_PARAMETER_INVALID: " + name)
        self.config = copy.deepcopy(c)
        if c["mode"] != "afd" or c["ffn_ranks"] not in (8, 16, 128):
            raise ValueError("DIRECT_PROFILE_WIDTH")
        geometry = {k: c[k] for k in BASE_FIELDS}
        geometry["ffn_ranks"] = 8
        self.base = QwenCostMapping(geometry)
        self.rows = c["batch_size"] // 16
        self.shapes = self.base.expected_shapes()
        self.shapes["routed_experts_local_combine"].update(
            ep=c["ffn_ranks"],
            local_experts=128 // c["ffn_ranks"],
            recipe=c.get("ffn_recipe", "common_block128_ue8m0"),
        )
        self.shapes["ffn_step_submit"] = dict(
            scope="once_per_F_actor_per_step", hardware=c.get("ffn_hardware", "GB300")
        )
        for k in ("all_gatherv", "reduce_scatterv", "p2p_A2F", "p2p_F2A"):
            del self.shapes[k]
        self.shapes["router"]["m"] = self.rows
        self.shapes["topk"]["rows"] = self.rows
        self.shapes.update(
            direct_pack=dict(
                rows=self.rows,
                hidden=4096,
                topk=8,
                boundary="3zero_fill+3copy fixed payload",
            ),
            direct_combine=dict(
                rows=self.rows,
                hidden=4096,
                F=c["ffn_ranks"],
                boundary="FP32 sum of BF16 partials plus BF16 output copy",
            ),
            direct_group_startup=dict(
                alpha_ns=c["direct_group_alpha_ns"],
                scope="once per rank grouped exchange; paired endpoints concurrent",
            ),
            direct_A2F_message=dict(
                bytes=self.rows * (4096 * 2 + 8 * 8 + 8 * 4),
                bandwidth_Bps=c["direct_link_bandwidth_Bps"],
                scope="one A/F peer hidden+ids+weights",
            ),
            direct_F2A_message=dict(
                bytes=self.rows * 4096 * 2,
                bandwidth_Bps=c["direct_link_bandwidth_Bps"],
                scope="one F/A peer weighted BF16 partial",
            ),
        )

    def expected_shapes(self):
        return copy.deepcopy(self.shapes)

    def resolve(self, kind, metadata):
        if kind == "attention_module":
            if metadata["rows"] != self.rows:
                raise ValueError("DIRECT_ATTENTION_ROWS")
            return [
                "input_norm_first" if metadata["layer"] == 0 else "input_norm_residual",
                "qkv",
                "qk_norm_rope",
                "trtllm_decode",
                "o",
                "post_attention_norm",
            ]
        if kind == "graph_replay_overhead":
            return (
                [kind, "trt_metadata"]
                if metadata.get("role") != "F"
                else ["ffn_step_submit"]
            )
        if kind not in self.shapes:
            raise ValueError("DIRECT_UNKNOWN_QUERY " + kind)
        shape = self.shapes[kind]
        if "rows" in metadata and metadata["rows"] != shape.get("rows", shape.get("m")):
            raise ValueError("DIRECT_ROWS_MISMATCH " + kind)
        for k in ("bytes", "world", "ep", "local_experts"):
            if k in metadata and k in shape and (metadata[k] != shape[k]):
                raise ValueError("DIRECT_SHAPE_MISMATCH " + kind)
        return [kind]

    def event_call(self, event):
        meta = dict(event["metadata"])
        actor = event["actor"]
        if actor.startswith(("A", "F")):
            meta["role"] = actor[0]
        return (event["kind"], meta)


class DirectAdapter:
    """Replay native Direct control with symbolic devices and paired events."""

    def __init__(self, source_root, mode):
        if mode != "afd":
            raise ValueError("AFD_ADAPTER_MODE_UNSUPPORTED: " + str(mode))
        if source_root is None:
            raise ValueError("AFD_SOURCE_ROOT_REQUIRED")
        self.root = Path(source_root)
        self.mode = mode

    def verify(self):
        return verify_source(self.root, mode=self.mode, routing="direct")

    def template_config(self, *, batch_size, effective_kv, ffn_ranks):
        return dict(
            model="qwen3_235b",
            mode="afd",
            routing="direct",
            ffn_ranks=ffn_ranks,
            batch_size=batch_size,
            effective_kv=effective_kv,
            layers=94,
            direct_group_alpha_ns=10000,
            direct_link_bandwidth_Bps=900000000000.0,
            direct_resource_policy="shared_endpoint_nic",
        )

    def cost_mapping(self, config):
        return DirectMapping(config)

    def execute(self, provider):
        sources = self.verify()
        c = provider.config
        F = c["ffn_ranks"]
        layers = c["layers"]
        shapes = provider.mapping.expected_shapes()
        # All actors post before time is consumed; final drain validates the DAG.
        clock = EventCoordinator(defer_cycle_checks=True)
        actors = []
        wires = {}
        current = [None]
        active_stream = [None]
        env = {k: v for (k, v) in vars(typing).items() if not k.startswith("__")}
        env.update(
            dataclass=dataclasses.dataclass,
            replace=dataclasses.replace,
            partial=functools.partial,
            nullcontext=contextlib.nullcontext,
            contextmanager=contextlib.contextmanager,
            operations_nvtx_range=lambda **kw: contextlib.nullcontext(),
            stage_forward_context=lambda *_: contextlib.nullcontext(),
            AFDError=ValueError,
            AFDRole=types.SimpleNamespace(ATTENTION="A", FFN="F"),
        )
        mod = types.ModuleType("sglang.srt.afd.expert_exchange")
        mod.__package__ = "sglang.srt.afd"
        mod.__dict__.update(env)
        top = types.ModuleType("sglang.srt.layers.moe.topk")
        top.StandardTopKOutput = lambda weights, ids, x: types.SimpleNamespace(
            topk_weights=weights, topk_ids=ids
        )
        base = types.ModuleType("sglang.srt.afd.model_adapters.base")
        base.stage_forward_context = lambda *_: contextlib.nullcontext()
        layer_mod = types.ModuleType("sglang.srt.afd.layer_executor")

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
                stream.actor.compute(
                    "direct_combine", stream=stream, rows=self.shape[1]
                )
                return Tensor(self.shape[1:], dtype)

        torch = types.ModuleType("torch")
        torch.bfloat16 = "BF16"
        torch.int64 = "I64"
        torch.float32 = "F32"
        torch.cuda = types.SimpleNamespace(Event=Event, stream=stream_context)

        def post(stream, peer, fields, send):
            a = stream.actor
            f = peer if a.role == "A" else a.rank
            ar = a.rank if a.role == "A" else peer - F
            key = (stream.direction, ar, f, *a.position)
            kind = (
                "direct_A2F_message" if stream.direction == 0 else "direct_F2A_message"
            )
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
                            deps=tuple(
                                (x for x in (stream.last, a.last) if x is not None)
                            ),
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

        modules = {
            name: types.ModuleType(name)
            for name in (
                "sglang",
                "sglang.srt",
                "sglang.srt.afd",
                "sglang.srt.layers",
                "sglang.srt.layers.moe",
                "sglang.srt.afd.model_adapters",
            )
        }
        modules.update(
            {
                "torch": torch,
                mod.__name__: mod,
                top.__name__: top,
                base.__name__: base,
                layer_mod.__name__: layer_mod,
            }
        )
        # Restore every replaced import on success, cost failure or source failure.
        with temporary_modules(modules):
            extract(
                self.root / "python/sglang/srt/batch_overlap/operations.py",
                mod.__dict__,
            )
            extract(
                self.root / "python/sglang/srt/afd/layer_executor.py",
                mod.__dict__,
                ["execute_layer_operations"],
            )
            layer_mod.execute_layer_operations = mod.execute_layer_operations
            extract(
                self.root / "python/sglang/srt/afd/pipeline.py",
                mod.__dict__,
                ["_run_layers"],
                "AFDAttentionPipeline",
            )
            attention_run = mod.__dict__.pop("_run_layers")
            extract(
                self.root / "python/sglang/srt/afd/expert_exchange.py",
                mod.__dict__,
                ["combine_partials", "AFDDirectExchange"],
            )
            cls = mod.AFDDirectExchange
            original_run_layer = cls.run_layer

            def traced_layer(obj, *, layer, stage):
                current[0].position = (layer, stage)
                return original_run_layer(obj, layer=layer, stage=stage)

            cls.run_layer = traced_layer
            r = c["batch_size"] // 16
            all_a = []
            all_f = []

            class Experts:
                def __call__(self, hidden, routing):
                    current[0].compute(
                        "routed_experts_local_combine",
                        rows=hidden.shape[0],
                        ep=F,
                        local_experts=128 // F,
                    )
                    return hidden

            for role in ("A", "F"):
                for rank in range(8 if role == "A" else F):
                    a = Actor(role, rank)
                    current[0] = a
                    a.compute("graph_replay_overhead")
                    if role == "A":
                        a.compute("embedding", rows=2 * r)
                    transport = Transport(a)
                    ex = cls.__new__(cls)
                    ex.role = role
                    ex.num_a = 8
                    ex.num_f = F
                    ex.stages = 2
                    ex.rows = r
                    ex.hidden = 4096
                    ex.top_k = 8
                    ex.active_stages = (True, True)
                    ex.transport = transport
                    ex.bindings = [
                        [Binding(Stream(a, s, d)) for d in range(2)] for s in range(2)
                    ]
                    a.streams = {
                        f"{s}:{d}": ex.bindings[s][d]._stream
                        for s in range(2)
                        for d in range(2)
                    }
                    ex.buffers = [
                        (
                            Tensor((F, r, 4096)),
                            Tensor((r, 4096), "BF16", "packed_hidden"),
                            Tensor((r, 8), "I64"),
                            Tensor((r, 8), "F32"),
                        )
                        if role == "A"
                        else (
                            Tensor((8, r, 4096)),
                            Tensor((8, r, 8), "I64"),
                            Tensor((8, r, 8), "F32"),
                        )
                        for s in range(2)
                    ]
                    ex.adapter = types.SimpleNamespace(
                        num_layers=layers,
                        inner=types.SimpleNamespace(
                            layers=[
                                types.SimpleNamespace(
                                    mlp=types.SimpleNamespace(experts=Experts())
                                )
                                for _ in range(layers)
                            ]
                        ),
                    )
                    transport.ffn_backend = ex
                    if role == "F":
                        ex.run_ffn_step()
                        all_f.append(a)
                        continue

                    class AAdapter:
                        requires_token_ids = False
                        num_layers = layers
                        inner = None

                        def prepare_stage(self, stage):
                            pass

                        def local_compute(self, **kw):
                            current[0].position = (kw["layer"], kw["stage"].index)
                            current[0].compute("attention_module", rows=r)
                            return (kw["hidden_states"], None)

                        def finish_layer(self, **kw):
                            return (kw["ffn_output"], None)

                    def route(hidden):
                        current[0].compute("router", rows=r)
                        current[0].compute("topk", rows=r)
                        return types.SimpleNamespace(
                            topk_ids=Tensor((r, 8), "I64"),
                            topk_weights=Tensor((r, 8), "F32"),
                        )

                    adapter = AAdapter()
                    adapter.inner = types.SimpleNamespace(
                        layers=[
                            types.SimpleNamespace(
                                mlp=types.SimpleNamespace(
                                    route=route, shared_experts=None
                                )
                            )
                            for _ in range(layers)
                        ]
                    )
                    stages = [
                        types.SimpleNamespace(
                            index=s,
                            rows=r,
                            forward_batch=None,
                            positions=None,
                            hidden_states=Tensor((r, 4096)),
                            residual=None,
                        )
                        for s in range(2)
                    ]
                    obj = types.SimpleNamespace(
                        _adapter=adapter,
                        _connector=types.SimpleNamespace(transport=transport),
                    )
                    attention_run(
                        obj,
                        stages=stages,
                        recv_buffers=[Tensor((r, 4096)) for _ in range(2)],
                        participates=(True, True),
                        uses_backend=True,
                    )
                    all_a.append(a)
            for a in all_a:
                a.position = (layers, -1)
                a.compute("stage_join", rows=2 * r)
                a.compute("final_norm", rows=2 * r)

            def group(kind):
                return clock.submit(
                    "A-output",
                    kind,
                    provider.quote(kind, world=8, global_rows=c["batch_size"]),
                    deps=tuple((a.last for a in all_a)),
                    resources=tuple((a.name + ".compute" for a in all_a)),
                    metadata=dict(world=8, global_rows=c["batch_size"]),
                )

            gather = group("lm_hidden_gather")
            for a in all_a:
                a.last = gather
                a.compute("lm_head", rows=c["batch_size"])
            exchange = group("logits_exchange")
            for a in all_a:
                a.last = exchange
                a.compute("logits_postprocess", rows=2 * r)
                a.compute("sampling", rows=2 * r)
            if not all((len(w["posts"]) == 2 for w in wires.values())):
                raise ValueError("UNPAIRED_DIRECT_MESSAGE")
            expected = 8 * F * 2 * 2 * layers
            if len(wires) != expected:
                raise ValueError("DIRECT_WIRE_COUNT " + str(len(wires)))
            fence = clock.submit(
                "step", "decode_complete", 0, deps=tuple((a.last for a in actors))
            )
            clock.finish(fence)
        return ExecutionTrace(clock.export(), sources, 8 + F)
