"""Real two-rank SGLang LL dispatcher/full-Graph experiments, without weights.

The expert arithmetic and model/attention constructors are synthetic. EP calls,
FP8 quantization, Graph backend, input registry, and runner execution are real.
"""

import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .comparison import compare
from .dispatcher import dispatchers_for, forward_layer, initialize, run_layer
from .ep_audit import EpAudit
from .oracle import RoutingBatch, expected_combine, make_fixture, validate_capacity

CASES = ("balanced", "hotspot", "padding", "empty_rank", "all_masked")
CHANGES = ("tokens", "routing", "weights", "all")


def backend_for(coordinator, capacity):
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    return FullCudaGraphBackend(
        SimpleNamespace(
            device_module=torch.cuda, model_runner=SimpleNamespace(tp_group=coordinator)
        ),
        nccl_ep_capacity=capacity,
    )


def gpu_inputs(batch, rank):
    return tuple(
        value[rank].cuda() for value in (batch.tokens, batch.expert_ids, batch.weights)
    )


def close_runtime(coordinator, audit):
    import nccl.core as core

    from sglang.srt.distributed.parallel_state import destroy_model_parallel

    # Public production shutdown first closes registered EP resources. This
    # experiment owns its additional PyNccl communicator (no __del__ in the
    # pinned revision), so close that explicitly after observing EP closure.
    destroy_model_parallel()
    evidence = audit.assert_closed()
    core.Communicator(ptr=coordinator.pynccl_comm.comm.value).destroy()
    coordinator.pynccl_comm.available = False
    coordinator.pynccl_comm.disabled = True
    dist.destroy_process_group()
    evidence["experiment_communicator_destroyed"] = True
    return evidence


def exercise_graph(
    *, mode, buckets=(8, 16, 32), replays=1000, layers=2, generations=2, identity=False
):
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    buckets = sorted(set(buckets))
    capacity, eager_capacity = max(buckets), min(1024, 2 * max(buckets))
    rank, coordinator, bindings = initialize(eager_capacity, graph_enabled=True)
    checked, timings, setup_times = 0, [], []
    # Native/peer failures are bounded by torchrun + the external timeout. Do
    # not enter collective cleanup or a success barrier from a failure finally.
    with EpAudit() as audit:
        dispatchers = dispatchers_for(coordinator, layers)
        backend = backend_for(coordinator, capacity)
        static = [
            gpu_inputs(make_fixture(capacity, step=layer), rank)
            for layer in range(layers)
        ]
        capture_stream = torch.cuda.Stream()
        replay_streams = (torch.cuda.Stream(), torch.cuda.Stream())

        def forward(bucket):
            return [
                forward_layer(
                    dispatcher,
                    *(item[:bucket] for item in inputs),
                    rank,
                    identity=identity,
                )
                for dispatcher, inputs in zip(dispatchers, static)
            ]

        for generation in range(generations):
            started = time.perf_counter()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream), backend.capture_session(
                capture_stream
            ):
                for bucket in reversed(buckets):
                    backend.capture_one(
                        ShapeKey(bucket), lambda bucket=bucket: forward(bucket)
                    )
            torch.cuda.synchronize()
            setup_times.append(time.perf_counter() - started)
            update_count = sum(
                item["operation"] == "handle_update" for item in audit.events
            )
            count = replays if mode == "dynamic" else 3
            for iteration in range(count):
                bucket = buckets[iteration % len(buckets)]
                visit = iteration // len(buckets)
                step = (0, 1, 0)[visit % 3]
                case = (
                    CASES[(visit // 3) % len(CASES)]
                    if mode == "dynamic"
                    else "balanced"
                )
                change = (
                    CHANGES[(visit // (3 * len(CASES))) % len(CHANGES)]
                    if mode == "dynamic"
                    else "all"
                )
                batches = [
                    make_fixture(bucket, case=case, step=step + layer, change=change)
                    for layer in range(layers)
                ]
                for batch in batches:
                    validate_capacity(batch, capacity)
                live = [gpu_inputs(batch, rank) for batch in batches]
                stream = replay_streams[iteration % 2]
                stream.wait_stream(torch.cuda.current_stream())
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                with torch.cuda.stream(
                    stream
                ), backend.replay_session(), torch.cuda.nvtx.range(
                    f"sglang_nccl_ep_graph/{bucket}"
                ):
                    start.record()
                    for inputs, sources in zip(static, live):
                        for target, source in zip(inputs, sources):
                            target[:bucket].copy_(source)
                    actual = backend.replay(ShapeKey(bucket), None)
                    end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
                if iteration % 17 == 0:
                    # Keep Graph outputs live across larger eager scratch use.
                    run_layer(
                        dispatchers[0],
                        make_fixture(eager_capacity, case="hotspot", step=1),
                        rank,
                        identity=identity,
                    )
                    checked += 1
                for batch, output in zip(batches, actual):
                    compare(batch, rank, *output, identity=identity)
                    checked += 1
            assert update_count == sum(
                item["operation"] == "handle_update" for item in audit.events
            ), "Python update ran during replay"
            backend.cleanup()
        resources = close_runtime(coordinator, audit)
    return {
        "implementation": "sglang_dispatcher_full_graph",
        "mode": mode,
        "checked": checked,
        "buckets": buckets,
        "layers": layers,
        "generations": generations,
        "replays_per_generation": count,
        "graph_capacity": capacity,
        "eager_capacity": eager_capacity,
        "changed_components": list(CHANGES) if mode == "dynamic" else ["all"],
        "replay_ms": timings,
        "warmup_capture_seconds": setup_times,
        "measurement_scope": "Synthetic full step including GPU input copies, expert work and observation snapshots; oracle and fixture generation excluded",
        "bindings": bindings,
        "resources": resources,
        "fp8_scales_applied": True,
        "tolerance": {"rtol": 0, "atol": 0},
        "model_loaded": False,
    }


def runner_fixture(bucket, values, valid_rows, *, layer=0):
    """Literal CPU routing table, independent of the GPU router arithmetic."""
    tables = (
        {0: [0, 2], 1: [1, 3], 2: [2, 0], 4: [0, 2], 8: [0, 2]},
        {0: [1, 3], 1: [2, 0], 2: [3, 1], 4: [1, 3], 8: [1, 3]},
    )
    tokens, ids, weights = [], [], []
    pattern = torch.tensor([1, 2, 4, 8] * 4).repeat_interleave(128)
    for rank in range(2):
        padded = list(values[rank]) + [0] * (bucket - len(values[rank]))
        tokens.append((torch.tensor(padded)[:, None] * pattern * 2**layer).bfloat16())
        routes = torch.tensor([tables[rank][value] for value in padded])
        routes[valid_rows[rank] :] = -1
        ids.append(routes)
        weights.append(
            torch.tensor(
                [[0.75, 0.25] if value == 1 else [0.25, 0.75] for value in padded]
            )
        )
    batch = RoutingBatch(tuple(tokens), tuple(ids), tuple(weights), 4)
    validate_capacity(batch, bucket)
    return batch


def runner_routing(batch, rank):
    """Synthetic GPU router feeding the real SGLang CUDA padding-mask kernel."""
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    values = batch.input_ids
    pattern = (2.0 ** (torch.arange(2048, device=values.device) // 128 % 4)).bfloat16()
    x = values[:, None].bfloat16() * pattern

    def router(**kwargs):
        first = (values + rank) % 4
        routes = torch.stack((first, (first + 2) % 4), dim=1).int()
        left = torch.where(values % 2 == 1, 0.75, 0.25)
        return torch.stack((left, 1 - left), dim=1), routes

    config = TopKConfig(
        top_k=2, custom_routing_function=router, allow_routed_experts_capture=False
    )
    topk = select_experts(
        x,
        torch.zeros(len(values), 4, device=values.device),
        config,
        num_token_non_padded=batch.num_token_non_padded,
    )
    return x, topk.topk_ids, topk.topk_weights


def exercise_runner(
    *, buckets=(8, 16, 32), layers=2, generations=2, identity=False, **unused
):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    from .runner_inputs import SyntheticDecodeRunner, input_batch

    buckets = sorted(set(buckets))
    if len(buckets) < 2 or buckets[0] < 2:
        raise ValueError("Runner probe requires at least two buckets, smallest >= 2")
    rank, coordinator, bindings = initialize(max(buckets), graph_enabled=True)
    observations, selected, checked = {}, [], 0
    with EpAudit() as audit:
        dispatchers = dispatchers_for(coordinator, layers)

        def forward(batch):
            x, ids, weights = runner_routing(batch, rank)
            result = [
                forward_layer(
                    dispatcher, x * 2**layer, ids, weights, rank, identity=identity
                )
                for layer, dispatcher in enumerate(dispatchers)
            ]
            if torch.cuda.is_current_stream_capturing():
                observations[batch.batch_size] = result
            return LogitsProcessorOutput(
                next_token_logits=torch.cat([item[2] for item in result], dim=1)
            )

        runner = SyntheticDecodeRunner(
            forward,
            coordinator,
            buckets=buckets,
            backend_factory=lambda runner: FullCudaGraphBackend(
                runner, nccl_ep_capacity=max(buckets)
            ),
        )
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        input_addresses = (
            runner.buffers.input_ids.data_ptr(),
            runner.buffers.num_token_non_padded.data_ptr(),
        )
        for generation in range(generations):
            pending = []
            mode = (
                CaptureHiddenMode.NULL
                if generation % 2 == 0
                else CaptureHiddenMode.FULL
            )
            for iteration, raw in enumerate(
                (max(1, buckets[0] - 3), buckets[0] + 1, max(1, buckets[0] - 3))
            ):
                bucket = next(value for value in buckets if value >= raw)
                values = tuple(
                    [
                        [1, 2, 4, 8][(row + (0, 1, 0)[iteration] + rank_index) % 4]
                        for row in range(raw)
                    ]
                    for rank_index in range(2)
                )
                valid = (0 if iteration == 1 else raw, raw)
                fixtures = [
                    runner_fixture(bucket, values, valid, layer=layer)
                    for layer in range(layers)
                ]
                incoming = input_batch(
                    values[rank], valid_rows=valid[rank], hidden_mode=mode
                )
                stream = streams[iteration % 2]
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    output = runner.execute(incoming).next_token_logits
                    # Queue a caller's read after execute/replay_session returns.
                    # The next stream's input writes must wait for this consumer.
                    torch.cuda._sleep(2_000_000)
                    consumed = output.clone()
                    captured = [
                        tuple(item.clone() for item in result)
                        for result in observations[bucket]
                    ]
                pending.append((raw, fixtures, consumed, captured, incoming))
                selected.append(runner.attn_backend.views[-1].batch_size)
                assert selected[-1] == bucket
                assert input_addresses == (
                    runner.buffers.input_ids.data_ptr(),
                    runner.buffers.num_token_non_padded.data_ptr(),
                )
            torch.cuda.synchronize()
            for raw, fixtures, output, captured, incoming in pending:
                for fixture, result in zip(fixtures, captured):
                    compare(fixture, rank, *result, identity=identity)
                    checked += 1
                wanted = torch.cat(
                    [
                        expected_combine(fixture, identity=identity)[rank]
                        for fixture in fixtures
                    ],
                    dim=1,
                )[:raw]
                torch.testing.assert_close(output.cpu().float(), wanted, rtol=0, atol=0)
            assert runner.capture_generations == generation + 1
        runner.backend.cleanup()
        resources = close_runtime(coordinator, audit)
    return {
        "implementation": "sglang_decode_runner",
        "bindings": bindings,
        "checked": checked,
        "selected_buckets": selected,
        "layers": layers,
        "generations": runner.capture_generations,
        "resources": resources,
        "input_addresses_stable": True,
        "delayed_output_consumers": True,
        "actual_topk_padding_mask": True,
        "tolerance": {"rtol": 0, "atol": 0},
        "constructor_and_model": "synthetic",
        "model_loaded": False,
    }


def exercise_cleanup(*, buckets=(8, 16, 32), layers=2, identity=False, **unused):
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    bucket = max(buckets)
    rank, coordinator, bindings = initialize(bucket, graph_enabled=True)
    retained_failures, checked = [], 0
    with EpAudit() as audit:
        dispatchers = dispatchers_for(coordinator, layers)
        backend = backend_for(coordinator, bucket)
        static = [
            gpu_inputs(make_fixture(bucket, step=layer), rank)
            for layer in range(layers)
        ]
        fail = True

        def forward():
            result = [
                forward_layer(dispatcher, *inputs, rank, identity=identity)
                for dispatcher, inputs in zip(dispatchers, static)
            ]
            if fail and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("controlled failure after completed combine")
            return result

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream), backend.capture_session(stream):
                backend.capture_one(ShapeKey(bucket), forward)
        except RuntimeError as error:
            if str(error) != "controlled failure after completed combine":
                raise
            retained_failures.append(
                error
            )  # Keep traceback/executable references alive.
        else:
            raise AssertionError("Controlled capture failure did not occur")
        graph_groups = [
            item for item in audit.groups.values() if item["rdma_buffer_size"] == 0
        ]
        assert len(graph_groups) == 1 and all(item["closed"] for item in graph_groups)
        assert all(graph.audit_reset for graph in audit.graphs)
        fail = False
        with torch.cuda.stream(stream), backend.capture_session(stream):
            backend.capture_one(ShapeKey(bucket), forward)
        torch.cuda.current_stream().wait_stream(stream)
        for step in (0, 1, 0):
            batches = [
                make_fixture(bucket, case="hotspot", step=step + layer)
                for layer in range(layers)
            ]
            with backend.replay_session():
                for batch, inputs in zip(batches, static):
                    for target, source in zip(inputs, gpu_inputs(batch, rank)):
                        target.copy_(source)
                actual = backend.replay(ShapeKey(bucket), None)
            for batch, result in zip(batches, actual):
                compare(batch, rank, *result, identity=identity)
                checked += 1
        backend.cleanup()
        resources = close_runtime(coordinator, audit)
    return {
        "implementation": "sglang_controlled_capture_failure",
        "bindings": bindings,
        "checked": checked,
        "layers": layers,
        "generations": 2,
        "resources": resources,
        "controlled_failures": len(retained_failures),
        "failure_traceback_retained": True,
        "recovery_scope": "host failure after completed combine; GPU/peer faults require process restart",
        "tolerance": {"rtol": 0, "atol": 0},
        "model_loaded": False,
    }
