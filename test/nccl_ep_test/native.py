"""Weight-free native EP hypothesis test, separate from SGLang integration.

There is one group and handle for all graphs in a generation. Host update only
prepares shape descriptors before capture; GPU copies supply replay-time routing.
Passing these tests does not establish SGLang runner integration correctness.
"""

import time
from datetime import timedelta
from itertools import product

import torch
import torch.distributed as dist

from .comparison import compare
from .environment import binding_check, prepare_jit, require_pair
from .oracle import (
    RoutingBatch,
    make_fixture,
    validate_capacity,
)


class NativeEp:
    def __init__(self, comm, capacity, hidden=2048, *, persistent):
        import nccl.ep as ep

        self.ep = ep
        self.capacity = capacity
        self.hidden = hidden
        self.persistent = persistent
        self.handle = None
        self.prepared_size = None
        self.group = ep.Group.create(
            comm,
            ep.GroupConfig(
                algorithm=ep.Algorithm.LOW_LATENCY,
                num_experts=4,
                num_topk=2,
                max_dispatch_tokens_per_rank=capacity,
                max_token_bytes=hidden * 2,
                rdma_buffer_size=0,
                max_num_sms=20,
            ),
        )
        self.x = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
        self.ids = torch.full((capacity, 2), -1, dtype=torch.int64, device="cuda")
        self.weights = torch.zeros(capacity, 2, dtype=torch.float32, device="cuda")
        self.recv = torch.empty(
            2, 2 * capacity, hidden, dtype=torch.bfloat16, device="cuda"
        )
        self.counts = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.combined = torch.empty_like(self.x)
        if persistent:
            # Initialize maximum layout once, so AUTO's RDMA allocation cannot
            # grow after graphs have captured pointers into group storage.
            self.handle = self._create_handle(capacity)
            self.prepared_size = capacity

    def _create_handle(self, size):
        return self.group.create_handle(
            self.ep.Layout.EXPERT_MAJOR,
            self.ep.Tensor(self.ids[:size]),
            stream=torch.cuda.current_stream().cuda_stream,
        )

    def prepare(self, size):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Handle shape update must happen before capture")
        if not 0 < size <= self.capacity:
            raise ValueError("Bucket must fit the frozen capacity")
        if self.persistent:
            self.handle.update(
                self.ep.Tensor(self.ids[:size]),
                stream=torch.cuda.current_stream().cuda_stream,
            )
            self.prepared_size = size

    def run(self, x, ids, weights, *, rank, identity=False):
        size = len(x)
        if self.persistent and self.prepared_size != size:
            raise RuntimeError("Prepare the bucket before warmup/capture")
        if not self.persistent and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The eager group must never enter capture")
        self.x[:size].copy_(x)
        self.ids[:size].copy_(ids)
        self.weights[:size].copy_(weights)
        ep, stream = self.ep, torch.cuda.current_stream().cuda_stream
        handle = self.handle if self.persistent else self._create_handle(size)
        self.counts.zero_()
        # send_only borrows these descriptors for the complete() continuation.
        dispatch_inputs = ep.DispatchInputs(tokens=ep.Tensor(self.x[:size]))
        dispatch_outputs = ep.DispatchOutputs(tokens=ep.Tensor(self.recv))
        layout_info = ep.LayoutInfo(expert_counters=ep.Tensor(self.counts))
        handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=layout_info,
            config=ep.DispatchConfig(send_only=1),
            stream=stream,
        )
        handle.complete(stream=stream)
        # Copies preserve each layer's observations across shared-buffer reuse.
        received, counts = self.recv.clone(), self.counts.clone()
        factors = torch.arange(
            rank * 2 + 1, rank * 2 + 3, device=x.device, dtype=x.dtype
        )
        expert_output = (
            self.recv.clone() if identity else self.recv * factors[:, None, None]
        )
        combine_inputs = ep.CombineInputs(tokens=ep.Tensor(expert_output))
        combine_outputs = ep.CombineOutputs(
            tokens=ep.Tensor(self.combined[:size]),
            topk_weights=ep.Tensor(self.weights[:size]),
        )
        handle.combine(
            combine_inputs,
            combine_outputs,
            config=ep.CombineConfig(send_only=1),
            stream=stream,
        )
        handle.complete(stream=stream)
        output = self.combined[:size].clone()
        if not self.persistent:
            handle.destroy()
        return received, counts, output

    def close(self):
        # Caller drops graph executables only after device completion, then
        # closes the handle before the group. No collective barrier on failure.
        if self.handle is not None:
            self.handle.destroy()
            self.handle = None
        self.group.destroy()


def _inputs(batch, rank):
    return tuple(
        value[rank].cuda() for value in (batch.tokens, batch.expert_ids, batch.weights)
    )


def _compare(batch, rank, outputs, identity=False):
    received, counts, combined = outputs
    compare(batch, rank, received, counts, combined, identity=identity)


def exercise(
    *, mode, buckets=(8, 16, 32), replays=1000, layers=2, generations=2, identity=False
):
    rank, _ = require_pair(ep=True)
    bindings = binding_check()
    jit = prepare_jit()
    import nccl.core as core

    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    shared = [bytes(core.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(shared, src=0)
    comm = core.Communicator.init(2, rank, core.UniqueId.from_bytes(shared[0]))
    capacity = max(buckets)
    cases = ("balanced", "hotspot", "padding", "empty_rank", "all_masked")
    changes = ("tokens", "routing", "weights", "all")
    eager_modes = ("eager", "zero_length", "duplicates")
    eager_capacity = capacity if mode in eager_modes else min(1024, capacity * 2)
    eager_started = time.perf_counter()
    eager = NativeEp(comm, eager_capacity, persistent=False)
    warmup_batch = make_fixture(eager_capacity)
    warmup_inputs = _inputs(warmup_batch, rank)
    for _ in range(2):
        warmup_actual = eager.run(*warmup_inputs, rank=rank, identity=identity)
    torch.cuda.synchronize()
    _compare(warmup_batch, rank, warmup_actual, identity)
    eager_warmup_seconds = time.perf_counter() - eager_started
    checked = 0
    graph_timings = []
    eager_timings = []
    eager_wall_timings = []
    # On GPU/peer failure, leave collective-owned objects to process exit;
    # torchrun plus an external timeout bounds failure handling.
    if mode == "zero_length":
        full = make_fixture(8)
        batch = RoutingBatch(
            (full.tokens[0][:0], full.tokens[1]),
            (full.expert_ids[0][:0], full.expert_ids[1]),
            (full.weights[0][:0], full.weights[1]),
            4,
        )
        validate_capacity(batch, capacity)
        _compare(
            batch,
            rank,
            eager.run(*_inputs(batch, rank), rank=rank, identity=identity),
            identity,
        )
        checked += 1
    elif mode == "duplicates":
        batch = make_fixture(8, case="duplicates")
        validate_capacity(batch, capacity)
        _compare(
            batch,
            rank,
            eager.run(*_inputs(batch, rank), rank=rank, identity=identity),
            identity,
        )
        checked += 1
    elif mode == "eager":
        for bucket, case, change, step in product(buckets, cases, changes, (0, 1, 0)):
            batch = make_fixture(bucket, case=case, step=step, change=change)
            validate_capacity(batch, capacity)
            live_inputs = _inputs(batch, rank)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            wall_start = time.perf_counter()
            actual = eager.run(*live_inputs, rank=rank, identity=identity)
            end.record()
            end.synchronize()
            eager_wall_timings.append((time.perf_counter() - wall_start) * 1000)
            eager_timings.append(start.elapsed_time(end))
            _compare(batch, rank, actual, identity)
            checked += 1
    else:
        for generation in range(generations):
            warmup_started = time.perf_counter()
            graph_ep = NativeEp(comm, capacity, persistent=True)
            graphs, inputs, outputs = {}, {}, {}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for bucket in sorted(buckets, reverse=True):
                inputs[bucket] = [
                    _inputs(make_fixture(bucket, step=layer), rank)
                    for layer in range(layers)
                ]
                # Ensure allocations/copies from the producer are visible.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph_ep.prepare(bucket)
                    for _ in range(2):
                        for item in inputs[bucket]:
                            graph_ep.run(*item, rank=rank, identity=identity)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = [
                        graph_ep.run(*item, rank=rank, identity=identity)
                        for item in inputs[bucket]
                    ]
                graphs[bucket], outputs[bucket] = graph, out
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warmup_started
            for iteration in range(replays if mode == "dynamic" else 3):
                bucket = buckets[iteration % len(buckets)]
                # A -> B -> A at each successive visit, plus bucket alternation.
                visit = iteration // len(buckets)
                step = (0, 1, 0)[visit % 3]
                case = (
                    cases[(visit // 3) % len(cases)]
                    if mode == "dynamic"
                    else "balanced"
                )
                change = (
                    changes[(visit // (3 * len(cases))) % len(changes)]
                    if mode == "dynamic"
                    else "all"
                )
                batches = [
                    make_fixture(bucket, case=case, step=step + layer, change=change)
                    for layer in range(layers)
                ]
                for batch, static in zip(batches, inputs[bucket]):
                    validate_capacity(batch, capacity)
                    for target, source in zip(static, _inputs(batch, rank)):
                        target.copy_(source)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                with torch.cuda.nvtx.range(f"nccl_ep_graph/{bucket}"):
                    start.record()
                    graphs[bucket].replay()
                    end.record()
                end.synchronize()
                graph_timings.append(start.elapsed_time(end))
                for batch, actual in zip(batches, outputs[bucket]):
                    _compare(batch, rank, actual, identity)
                    checked += 1
                if iteration % 17 == 0:
                    # A separate eager group may change its handle shape.
                    batch = make_fixture(eager_capacity, case="hotspot", step=1)
                    validate_capacity(batch, eager_capacity)
                    _compare(
                        batch,
                        rank,
                        eager.run(*_inputs(batch, rank), rank=rank, identity=identity),
                        identity,
                    )
                    checked += 1
            torch.cuda.synchronize()
            for graph in graphs.values():
                graph.reset()
            graphs.clear()
            del graph, out
            outputs.clear()
            graph_ep.close()
            print(
                f"generation={generation} warmup_and_capture_seconds={warmup_seconds:.3f}",
                flush=True,
            )
    torch.cuda.synchronize()
    eager.close()
    comm.destroy()
    dist.destroy_process_group()
    return {
        "implementation": "native",
        "mode": mode,
        "graph_capacity": capacity if mode not in eager_modes else None,
        "eager_capacity": eager_capacity,
        "checked": checked,
        "bindings": bindings,
        "jit": jit,
        "replay_ms": graph_timings,
        "eager_ms": eager_timings,
        "eager_wall_ms": eager_wall_timings,
        "eager_warmup_seconds": eager_warmup_seconds,
        "measurement_scope": "Synthetic full step including buffer copies, expert work and observation snapshots; oracle excluded",
        "buckets": list(buckets),
        "layers": layers if mode not in eager_modes else 1,
        "changed_components": (
            list(changes) if mode in ("eager", "dynamic") else ["all"]
        ),
        "generations": generations if mode not in eager_modes else 0,
        "replays_per_generation": (
            replays if mode == "dynamic" else 3 if mode == "capture" else 0
        ),
        "tolerance": {"rtol": 0, "atol": 0},
    }
