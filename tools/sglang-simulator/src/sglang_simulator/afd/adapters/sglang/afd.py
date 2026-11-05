"""Native A/F control -> existing HiSim coordinator, CPU hooks only.

Kernel hooks consume costs; native pipeline, grouping and TBO bodies are unchanged.
Synthetic providers are regression inputs and never yield production throughput.
"""

import contextlib
import dataclasses
import functools
import types
import typing
from pathlib import Path

from ...coordinator import EventCoordinator
from .loader import extract, temporary_modules
from .source import verify_source


def run(provider, *, source_root, F=8, layers=94, stage_rows=16, full_decode=False):
    verify_source(source_root, mode="afd")
    mod = types.ModuleType("af_native_extracted")
    with temporary_modules({mod.__name__: mod}):
        return _run(
            provider,
            source_root=source_root,
            F=F,
            layers=layers,
            stage_rows=stage_rows,
            full_decode=full_decode,
            env=mod.__dict__,
        )


def _run(provider, *, source_root, F, layers, stage_rows, full_decode, env):
    BASE = Path(source_root) / "python/sglang/srt"
    if F not in (2, 4, 8):
        raise ValueError("unsupported F topology")
    clock = EventCoordinator()
    collectors = {}
    wires = {}
    actors = []
    env.update({k: v for k, v in vars(typing).items() if not k.startswith("__")})
    env.update(
        dataclass=dataclasses.dataclass,
        replace=dataclasses.replace,
        contextmanager=contextlib.contextmanager,
        nullcontext=contextlib.nullcontext,
        partial=functools.partial,
        operations_nvtx_range=lambda **kw: contextlib.nullcontext(),
        stage_forward_context=lambda *_: contextlib.nullcontext(),
        AFDError=ValueError,
        AFDRole=types.SimpleNamespace(ATTENTION="A", FFN="F"),
    )
    extract(BASE / "batch_overlap/operations.py", env)
    extract(BASE / "afd/layer_executor.py", env, ["execute_layer_operations"])
    extract(BASE / "afd/pipeline.py", env, ["_run_layers"], "AFDAttentionPipeline")
    a_run = env.pop("_run_layers")
    extract(BASE / "afd/pipeline.py", env, ["_run_layers"], "AFDFFNPipeline")
    f_run = env.pop("_run_layers")
    extract(
        BASE / "afd/model_adapters/base.py",
        env,
        ["local_compute", "_group_ffn_output", "_compute_ffn_output"],
        "AFDDecoderAdapter",
    )
    current = [None]

    class Tensor:
        def __init__(self, rows, stage=0):
            self.shape = (rows, 4096)
            self.stage = stage
            self.dtype = "bfloat16"
            self.device = "cpu-symbolic"

        def __getitem__(self, key):
            return self

        def __setitem__(self, key, value):
            pass

        def narrow(self, axis, offset, rows):
            return Tensor(rows, self.stage)

    class Actor:
        def __init__(self, role, rank):
            self.role = role
            self.rank = rank
            self.name = f"{role}{rank}"
            self.last = None
            self.streams = {}
            self.position = (0, 0)
            self.recv_count = [0, 0]
            actors.append(self)

        def deps(self, stream=None):
            return tuple(
                dict.fromkeys(
                    e for e in (self.last, self.streams.get(stream)) if e is not None
                )
            )

        def compute(self, kind, **meta):
            eid = clock.submit(
                self.name,
                kind,
                provider.quote(
                    kind,
                    role=self.role,
                    rank=self.rank,
                    layer=self.position[0],
                    stage=self.position[1],
                    **meta,
                ),
                deps=self.deps(),
                resources=(self.name + ".compute",),
                metadata=dict(layer=self.position[0], stage=self.position[1], **meta),
            )
            self.last = eid
            return eid

        def transfer(self, direction, a, layer, stage, endpoint):
            key = (direction, a, layer, stage)
            if key not in wires:
                wires[key] = {
                    "event": clock.reserve(
                        "wire",
                        direction,
                        metadata=dict(A=a, F=a // (8 // F), layer=layer, stage=stage),
                    ),
                    "posts": {},
                }
            item = wires[key]
            if endpoint in item["posts"]:
                raise ValueError("duplicate wire endpoint")
            item["posts"][endpoint] = self.deps(direction)
            eid = item["event"]
            self.streams[direction] = eid
            if len(item["posts"]) == 2:
                deps = tuple(
                    dict.fromkeys(x for v in item["posts"].values() for x in v)
                )
                clock.resolve(
                    eid,
                    deps=deps,
                    duration_ns=provider.quote(
                        "p2p",
                        direction=direction,
                        bytes=stage_rows * 4096 * 2,
                        A=a,
                        F=a // (8 // F),
                    ),
                    resources=(f"link.A{a}.F{a // (8 // F)}.{direction}",),
                )
            return eid

        def collective(self, kind):
            layer, stage = self.position
            key = (kind, layer, stage)
            if key not in collectors:
                collectors[key] = {
                    "event": clock.reserve(
                        "F-group",
                        kind,
                        metadata=dict(layer=layer, stage=stage, world=F),
                    ),
                    "posts": {},
                }
            item = collectors[key]
            if self.rank in item["posts"]:
                raise ValueError("duplicate collective rank")
            item["posts"][self.rank] = self.deps()
            self.last = item["event"]
            if len(item["posts"]) == F:
                clock.resolve(
                    item["event"],
                    deps=tuple(
                        dict.fromkeys(e for d in item["posts"].values() for e in d)
                    ),
                    duration_ns=provider.quote(
                        kind, world=F, global_stage_tokens=8 * stage_rows, hidden=4096
                    ),
                    resources=tuple(f"F{i}.compute" for i in range(F)),
                )

    class Group:
        @property
        def rank_in_group(self):
            return current[0].rank

        def all_gatherv(self, local, sizes):
            current[0].collective("all_gatherv")
            return [Tensor(sum(sizes), current[0].position[1])]

        def reduce_scatterv(self, partial, sizes):
            current[0].collective("reduce_scatterv")
            return Tensor(sizes[current[0].rank], current[0].position[1])

    env["get_parallel"] = lambda: types.SimpleNamespace(tp_group=Group())

    class FAdapter:
        role = "F"
        requires_token_ids = False

        def __init__(self, actor):
            self.actor = actor
            self.num_layers = layers
            self.inner = types.SimpleNamespace(
                layers=[
                    types.SimpleNamespace(compute_ffn_output=self.experts)
                    for _ in range(layers)
                ]
            )

        def experts(self, hidden, batch):
            self.actor.compute(
                "router_deepgemm_module",
                global_stage_tokens=hidden.shape[0],
                ep=F,
                local_experts=128 // F,
                recipe="common_block128_ue8m0",
            )
            return hidden

        def local_compute(self, **kw):
            self.actor.position = (kw["layer"], kw["stage"].index)
            return env["local_compute"](self, **kw)

        _group_ffn_output = env["_group_ffn_output"]
        _compute_ffn_output = env["_compute_ffn_output"]

    class AAdapter:
        requires_token_ids = False
        num_layers = layers

        def __init__(self, actor):
            self.actor = actor

        def prepare_stage(self, **kw):
            pass

        def local_compute(self, **kw):
            self.actor.position = (kw["layer"], kw["stage"].index)
            self.actor.compute("attention_module", rows=stage_rows)
            return kw["hidden_states"], None

        def finish_layer(self, **kw):
            return kw["stage"].hidden_states, None

    class Transport:
        capturing = True

        def __init__(self, a):
            self.a = a

        def wait(self, e):
            self.a.last = clock.submit(
                self.a.name, "stream_wait", 0, deps=self.a.deps() + (e,)
            )

        def dispatch(self, tensor):
            l, s = self.a.position
            self.a.transfer("A2F", self.a.rank, l, s, "send")

        def receive_return(self, buf):
            l, s = self.a.position
            return self.a.transfer("F2A", self.a.rank, l, s, "recv")

        def receive_dispatch(self, bufs):
            s = bufs[0].stage
            l = self.a.recv_count[s]
            self.a.recv_count[s] += 1
            eid = None
            for a in range(self.a.rank * (8 // F), (self.a.rank + 1) * (8 // F)):
                eid = self.a.transfer("A2F", a, l, s, "recv")
            return eid

        def return_result(self, outputs):
            l, s = self.a.position
            assert len(outputs) == 8 // F
            for a in range(self.a.rank * (8 // F), (self.a.rank + 1) * (8 // F)):
                self.a.transfer("F2A", a, l, s, "send")

    fake_torch = types.ModuleType("torch")

    def zeros(shape, **kw):
        current[0].compute("pack", rows=shape[0])
        return Tensor(shape[0], current[0].position[1])

    fake_torch.zeros = zeros
    fake_runtime = types.ModuleType("sglang.srt.runtime_context")
    fake_runtime.get_forward = lambda: types.SimpleNamespace(
        scoped=lambda **kw: contextlib.nullcontext()
    )
    with temporary_modules(
        {"torch": fake_torch, "sglang.srt.runtime_context": fake_runtime}
    ):
        for role, count in (("A", 8), ("F", F)):
            for rank in range(count):
                a = Actor(role, rank)
                current[0] = a
                if full_decode:
                    a.last = clock.submit(
                        a.name,
                        "graph_replay_overhead",
                        provider.quote("graph_replay_overhead", role=role, rank=rank),
                        resources=(a.name + ".host",),
                    )
                    if role == "A":
                        a.compute("embedding", rows=2 * stage_rows, hidden=4096)
                adapter = AAdapter(a) if role == "A" else FAdapter(a)
                obj = types.SimpleNamespace(
                    _adapter=adapter,
                    _connector=types.SimpleNamespace(transport=Transport(a)),
                )
                stages = [
                    types.SimpleNamespace(
                        index=s,
                        rows=stage_rows,
                        forward_batch=None,
                        positions=None,
                        hidden_states=Tensor(stage_rows, s),
                        residual=None,
                        group_widths=(stage_rows,) * (8 // F),
                        merge_sizes=(stage_rows * 8 // F,) * F,
                        merge_lane=rank,
                        graph_output=None,
                    )
                    for s in range(2)
                ]
                if role == "A":
                    a_run(
                        obj,
                        stages=stages,
                        recv_buffers=[Tensor(stage_rows, s) for s in range(2)],
                        participates=(True, True),
                    )
                else:
                    f_run(
                        obj,
                        stages=stages,
                        recv_buffers=[
                            tuple(Tensor(stage_rows, s) for _ in range(8 // F))
                            for s in range(2)
                        ],
                        participates=(True, True),
                    )
    if full_decode:
        aa = [a for a in actors if a.role == "A"]
        for a in aa:
            a.position = (layers, -1)
            a.compute("stage_join", rows=2 * stage_rows, hidden=4096)
            a.compute("final_norm", rows=2 * stage_rows, hidden=4096)
        gather = clock.submit(
            "A-output",
            "lm_hidden_gather",
            provider.quote("lm_hidden_gather", world=8, global_rows=16 * stage_rows),
            deps=tuple(a.last for a in aa),
            resources=tuple(a.name + ".compute" for a in aa),
        )
        for a in aa:
            a.last = gather
            a.compute("lm_head", rows=16 * stage_rows, k=4096)
        exchange = clock.submit(
            "A-output",
            "logits_exchange",
            provider.quote("logits_exchange", world=8, global_rows=16 * stage_rows),
            deps=tuple(a.last for a in aa),
            resources=tuple(a.name + ".compute" for a in aa),
        )
        for a in aa:
            a.last = exchange
            a.compute("logits_postprocess", rows=2 * stage_rows)
            a.compute("sampling", rows=2 * stage_rows)
    assert all(len(w["posts"]) == 2 for w in wires.values())
    assert all(len(c["posts"]) == F for c in collectors.values())
    fence = clock.submit(
        "step",
        "all_stream_join",
        0,
        deps=tuple(
            set(
                e
                for a in actors
                for e in [a.last, *a.streams.values()]
                if e is not None
            )
        ),
    )
    ns = clock.finish(fence)
    return dict(
        events=clock.export(),
        query_log=provider.queries,
        synthetic_step_ns=ns if provider.synthetic else None,
        production_step_ms=None,
        production_tgs=None,
        source_control="native A/F pipeline + native group_ffn_output + TBO",
        kernel_hook="router+DeepGEMM module; no GPU kernel execution",
        pairs=len(wires),
        collective_events=len(collectors),
        communication_cost_rule="once per paired wire; once per collective group; no per-rank multiplication",
        unresolved_validation=[
            "resource-contention model against real service",
            "cost family qualification",
            "full output/head cost and concrete logits exchange dispatch",
        ],
    )
