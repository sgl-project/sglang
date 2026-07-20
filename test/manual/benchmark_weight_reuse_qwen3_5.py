"""Qwen3.5 checkpoint cold-start and Mooncake GPU weight-reuse benchmark.

This is a manual benchmark.  It compares process-cold, page-cache-warm model
loading with a resident source whose CUDA weights are copied into a freshly
allocated target model through Mooncake Transfer Engine.

Example:

PYTHONPATH=/path/to/mooncake-overlay:/path/to/sglang/python \
python test/manual/benchmark_weight_reuse_qwen3_5.py \
  --model /path/to/Qwen3.5-0.8B \
  --hostname 192.168.8.49 \
  --source-gpu 0 \
  --target-gpu 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

_REVISION = "weight-reuse-benchmark"


def _set_visible_gpu(physical_gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def _build_runner(model_path: str, load_format: str):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        load_format=load_format,
        disable_cuda_graph=True,
        enable_weight_runtime_manifest=True,
        mem_fraction_static=0.1,
        tp_size=1,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        ps=ParallelState.trivial(tp_size=1),
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _model_digest(runner) -> str:
    """Hash every distinct parameter storage outside the measured interval."""

    import torch

    torch.cuda.synchronize()
    digest = hashlib.sha256()
    seen = set()
    for name, parameter in runner.model.named_parameters(remove_duplicate=False):
        key = (
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.storage_offset()),
            tuple(int(value) for value in parameter.shape),
            tuple(int(value) for value in parameter.stride()),
            str(parameter.dtype),
        )
        if key in seen:
            continue
        seen.add(key)
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _runtime_manifest(runner, *, model_path: str, instance_id: str, endpoint: str):
    from mooncake.weight_transfer import RuntimeManifest

    inventory = runner.get_weight_runtime_manifest(
        model_id=Path(model_path).name,
        revision=_REVISION,
        instance_id=instance_id,
        worker_id=instance_id,
        endpoint=endpoint,
    )
    return inventory, RuntimeManifest.from_runtime_inventory(inventory)


def _register_manifest(engine, manifest):
    from mooncake.weight_transfer import MemoryRegistrationLease

    ranges = sorted(
        {(fragment.address, fragment.nbytes) for fragment in manifest.fragments}
    )
    registered = []
    try:
        for address, nbytes in ranges:
            result = engine.register_memory(address, nbytes)
            if result != 0:
                raise RuntimeError(
                    f"register_memory failed for {address}+{nbytes}: {result}"
                )
            registered.append((address, nbytes))
    except BaseException:
        for address, _ in reversed(registered):
            engine.unregister_memory(address)
        raise
    leases = tuple(
        MemoryRegistrationLease.from_fragment(fragment)
        for fragment in manifest.fragments
    )
    return registered, leases


def _unregister_ranges(engine, ranges) -> None:
    failures = []
    for address, _ in reversed(ranges):
        result = engine.unregister_memory(address)
        if result != 0:
            failures.append((address, result))
    if failures:
        raise RuntimeError(f"unregister_memory failures: {failures}")


def _send_error(connection) -> None:
    connection.send(("error", traceback.format_exc()))


def _cold_worker(connection, *, model_path: str, physical_gpu: int) -> None:
    _set_visible_gpu(physical_gpu)
    started = time.perf_counter()
    runner = None
    try:
        runner = _build_runner(model_path, "auto")
        import torch

        torch.cuda.synchronize()
        connection.send(
            (
                "ready",
                {
                    "entry_to_weights_ready_s": time.perf_counter() - started,
                    "weight_bytes": sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in runner.model.parameters()
                    ),
                },
            )
        )
        while True:
            command = connection.recv()
            if command[0] == "digest":
                connection.send(("digest", _model_digest(runner)))
            elif command[0] == "stop":
                connection.send(("stopped",))
                break
            else:
                raise RuntimeError(f"unknown cold-worker command: {command[0]}")
    except BaseException:
        _send_error(connection)
    finally:
        _destroy_process_group()
        connection.close()


def _source_worker(
    connection,
    *,
    model_path: str,
    physical_gpu: int,
    hostname: str,
    protocol: str,
    transport_device: str,
) -> None:
    _set_visible_gpu(physical_gpu)
    started = time.perf_counter()
    runner = None
    engine = None
    inventory = None
    registered = []
    active_plan = None
    active_target = None
    active_target_registrations = None
    try:
        from mooncake.engine import TransferEngine
        from mooncake.weight_transfer import (
            MooncakeTransferEngineSink,
            plan_runtime_transfer,
        )

        runner = _build_runner(model_path, "auto")
        import torch

        torch.cuda.synchronize()
        model_ready_at = time.perf_counter()
        engine = TransferEngine()
        result = engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            protocol,
            transport_device,
        )
        if result != 0:
            raise RuntimeError(f"source TransferEngine initialize failed: {result}")
        endpoint = f"{hostname}:{engine.get_rpc_port()}"
        inventory, manifest = _runtime_manifest(
            runner,
            model_path=model_path,
            instance_id="benchmark-source-tp0",
            endpoint=endpoint,
        )
        registered, registrations = _register_manifest(engine, manifest)
        prepared_at = time.perf_counter()
        source_digest = _model_digest(runner)
        connection.send(
            (
                "ready",
                {
                    "entry_to_model_ready_s": model_ready_at - started,
                    "manifest_and_registration_s": prepared_at - model_ready_at,
                    "entry_to_source_ready_s": prepared_at - started,
                    "weight_bytes": sum(
                        fragment.nbytes for fragment in manifest.fragments
                    ),
                    "registered_physical_bytes": sum(
                        nbytes for _, nbytes in registered
                    ),
                    "tensor_count": len(manifest.fragments),
                    "digest": source_digest,
                },
            )
        )
        sink = MooncakeTransferEngineSink(engine)
        while True:
            command = connection.recv()
            if command[0] == "plan":
                active_target = command[1]
                active_target_registrations = command[2]
                plan_started = time.perf_counter()
                active_plan = plan_runtime_transfer((manifest,), (active_target,))
                connection.send(
                    (
                        "planned",
                        {
                            "plan_s": time.perf_counter() - plan_started,
                            "logical_bytes": active_plan.total_bytes,
                            "copy_ranges": len(active_plan.operations),
                        },
                    )
                )
            elif command[0] == "execute":
                if active_plan is None or active_target is None:
                    raise RuntimeError("execute requested before plan")
                transfer_started = time.perf_counter()
                receipts = sink.execute(
                    active_plan,
                    manifest,
                    (active_target,),
                    target_registrations=active_target_registrations,
                    source_pre_registered=True,
                    source_registrations=registrations,
                )
                transfer_s = time.perf_counter() - transfer_started
                connection.send(
                    (
                        "transferred",
                        {
                            "transfer_s": transfer_s,
                            "wire_bytes": sum(item.nbytes for item in receipts),
                            "wire_operations": sum(
                                item.operation_count for item in receipts
                            ),
                        },
                    )
                )
                active_plan = None
                active_target = None
                active_target_registrations = None
            elif command[0] == "stop":
                connection.send(("stopped",))
                break
            else:
                raise RuntimeError(f"unknown source-worker command: {command[0]}")
    except BaseException:
        _send_error(connection)
    finally:
        if runner is not None and inventory is not None:
            runner.release_weight_runtime_manifest(inventory.lease_id)
        if engine is not None and registered:
            _unregister_ranges(engine, registered)
        _destroy_process_group()
        connection.close()


def _reuse_target_worker(
    connection,
    *,
    model_path: str,
    physical_gpu: int,
    hostname: str,
    protocol: str,
    transport_device: str,
    iteration: int,
) -> None:
    _set_visible_gpu(physical_gpu)
    started = time.perf_counter()
    runner = None
    engine = None
    inventory = None
    registered = []
    update_token = None
    try:
        from mooncake.engine import TransferEngine

        runner = _build_runner(model_path, "dummy")
        import torch

        torch.cuda.synchronize()
        model_ready_at = time.perf_counter()
        engine = TransferEngine()
        result = engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            protocol,
            transport_device,
        )
        if result != 0:
            raise RuntimeError(f"target TransferEngine initialize failed: {result}")
        endpoint = f"{hostname}:{engine.get_rpc_port()}"
        inventory, manifest = _runtime_manifest(
            runner,
            model_path=model_path,
            instance_id=f"benchmark-target-{iteration}-tp0",
            endpoint=endpoint,
        )
        manifest_at = time.perf_counter()
        registered, registrations = _register_manifest(engine, manifest)
        registered_at = time.perf_counter()
        connection.send(
            (
                "ready",
                {
                    "manifest": manifest,
                    "registrations": registrations,
                    "entry_to_dummy_ready_s": model_ready_at - started,
                    "manifest_s": manifest_at - model_ready_at,
                    "target_registration_s": registered_at - manifest_at,
                    "entry_to_transfer_ready_s": registered_at - started,
                    "registered_physical_bytes": sum(
                        nbytes for _, nbytes in registered
                    ),
                },
            )
        )
        while True:
            command = connection.recv()
            if command[0] == "begin_update":
                runner.release_weight_runtime_manifest(inventory.lease_id)
                inventory = None
                update_token = runner.weight_snapshot_coordinator.begin_update()
                connection.send(("update_started",))
            elif command[0] == "commit":
                if update_token is None:
                    raise RuntimeError("commit requested before begin_update")
                runner.weight_snapshot_coordinator.finish_update(
                    update_token, success=True
                )
                update_token = None
                generation = runner.commit_weight_runtime_revision()
                torch.cuda.synchronize()
                connection.send(
                    (
                        "committed",
                        {
                            "generation": generation,
                            "entry_to_weights_ready_s": time.perf_counter() - started,
                        },
                    )
                )
            elif command[0] == "digest":
                connection.send(("digest", _model_digest(runner)))
            elif command[0] == "stop":
                connection.send(("stopped",))
                break
            else:
                raise RuntimeError(f"unknown target-worker command: {command[0]}")
    except BaseException:
        if runner is not None and update_token is not None:
            runner.weight_snapshot_coordinator.finish_update(
                update_token, success=False
            )
            update_token = None
        _send_error(connection)
    finally:
        if runner is not None and inventory is not None:
            runner.release_weight_runtime_manifest(inventory.lease_id)
        if engine is not None and registered:
            _unregister_ranges(engine, registered)
        _destroy_process_group()
        connection.close()


def _recv(connection, *, expected: str, timeout_s: float = 900.0):
    if not connection.poll(timeout_s):
        raise TimeoutError(f"timed out waiting for {expected}")
    message = connection.recv()
    if message[0] == "error":
        raise RuntimeError(message[1])
    if message[0] != expected:
        raise RuntimeError(f"expected {expected}, got {message[0]}")
    return message[1] if len(message) > 1 else None


def _stop_process(connection, process) -> None:
    if process.is_alive():
        connection.send(("stop",))
        _recv(connection, expected="stopped", timeout_s=60)
    connection.close()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
        raise RuntimeError(f"worker {process.pid} did not stop")
    if process.exitcode != 0:
        raise RuntimeError(f"worker exited with {process.exitcode}")


def _terminate_process(connection, process) -> None:
    connection.close()
    if process.is_alive():
        process.terminate()
    process.join(timeout=30)


def _start_worker(context, target, kwargs):
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=target, kwargs={"connection": child_connection, **kwargs}
    )
    started = time.perf_counter()
    process.start()
    child_connection.close()
    return parent_connection, process, started


def _median(records: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def run_benchmark(args) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    source_connection = None
    source_process = None
    cold_records = []
    reuse_records = []
    try:
        source_connection, source_process, source_started = _start_worker(
            context,
            _source_worker,
            {
                "model_path": args.model,
                "physical_gpu": args.source_gpu,
                "hostname": args.hostname,
                "protocol": args.protocol,
                "transport_device": args.transport_device,
            },
        )
        source = _recv(source_connection, expected="ready")
        source["spawn_to_source_ready_s"] = time.perf_counter() - source_started

        for iteration in range(args.iterations):
            connection, process, started = _start_worker(
                context,
                _cold_worker,
                {
                    "model_path": args.model,
                    "physical_gpu": args.target_gpu,
                },
            )
            try:
                record = _recv(connection, expected="ready")
                record["spawn_to_weights_ready_s"] = time.perf_counter() - started
                if iteration == 0:
                    connection.send(("digest",))
                    record["digest"] = _recv(connection, expected="digest")
                cold_records.append(record)
                _stop_process(connection, process)
            except BaseException:
                _terminate_process(connection, process)
                raise

        for iteration in range(args.iterations):
            connection, process, started = _start_worker(
                context,
                _reuse_target_worker,
                {
                    "model_path": args.model,
                    "physical_gpu": args.target_gpu,
                    "hostname": args.hostname,
                    "protocol": args.protocol,
                    "transport_device": args.transport_device,
                    "iteration": iteration,
                },
            )
            try:
                record = _recv(connection, expected="ready")
                target_manifest = record.pop("manifest")
                target_registrations = record.pop("registrations")
                source_connection.send(("plan", target_manifest, target_registrations))
                record.update(_recv(source_connection, expected="planned"))
                connection.send(("begin_update",))
                _recv(connection, expected="update_started")
                source_connection.send(("execute",))
                record.update(_recv(source_connection, expected="transferred"))
                connection.send(("commit",))
                record.update(_recv(connection, expected="committed"))
                record["spawn_to_weights_ready_s"] = time.perf_counter() - started
                if iteration == 0:
                    connection.send(("digest",))
                    record["digest"] = _recv(connection, expected="digest")
                reuse_records.append(record)
                _stop_process(connection, process)
            except BaseException:
                _terminate_process(connection, process)
                raise

        digests = {
            source["digest"],
            cold_records[0]["digest"],
            reuse_records[0]["digest"],
        }
        if len(digests) != 1:
            raise RuntimeError(
                "source, checkpoint target, and reused target parameter digests differ"
            )

        return {
            "model": args.model,
            "hostname": args.hostname,
            "protocol": args.protocol,
            "source_gpu": args.source_gpu,
            "target_gpu": args.target_gpu,
            "iterations": args.iterations,
            "cache_boundary": "process-cold, page-cache-warm",
            "measurement_boundary": "process spawn to ModelRunner weights ready",
            "source": source,
            "cold": cold_records,
            "reuse": reuse_records,
            "summary": {
                "cold_spawn_to_weights_ready_median_s": _median(
                    cold_records, "spawn_to_weights_ready_s"
                ),
                "reuse_spawn_to_weights_ready_median_s": _median(
                    reuse_records, "spawn_to_weights_ready_s"
                ),
                "reuse_transfer_median_s": _median(reuse_records, "transfer_s"),
                "reuse_plan_median_s": _median(reuse_records, "plan_s"),
                "reuse_target_registration_median_s": _median(
                    reuse_records, "target_registration_s"
                ),
            },
            "digest_verified": True,
        }
    finally:
        if source_process is not None and source_connection is not None:
            if source_process.is_alive():
                try:
                    _stop_process(source_connection, source_process)
                except BaseException:
                    _terminate_process(source_connection, source_process)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=4)
    parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--transport-device", default="")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.source_gpu == args.target_gpu:
        parser.error("source and target GPUs must differ")
    return args


if __name__ == "__main__":
    benchmark = run_benchmark(parse_args())
    print("WEIGHT_REUSE_BENCHMARK_JSON=" + json.dumps(benchmark, sort_keys=True))
