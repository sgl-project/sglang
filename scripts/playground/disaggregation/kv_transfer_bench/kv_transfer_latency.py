#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|b)?\s*$", re.I)
_UNIT_TO_BYTES = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
}


class TargetInfo:
    def __init__(self, session_id, host, gpu_id, ptr, bytes, ib_device, protocol):
        self.session_id = session_id
        self.host = host
        self.gpu_id = gpu_id
        self.ptr = ptr
        self.bytes = bytes
        self.ib_device = ib_device
        self.protocol = protocol

    def as_dict(self):
        return {
            "session_id": self.session_id,
            "host": self.host,
            "gpu_id": self.gpu_id,
            "ptr": self.ptr,
            "bytes": self.bytes,
            "ib_device": self.ib_device,
            "protocol": self.protocol,
        }

    def __eq__(self, other):
        if not isinstance(other, TargetInfo):
            return NotImplemented
        return self.as_dict() == other.as_dict()


def _time_ns() -> int:
    if hasattr(time, "time_ns"):
        return time.time_ns()
    return int(time.time() * 1_000_000_000)


def _perf_counter_ns() -> int:
    if hasattr(time, "perf_counter_ns"):
        return time.perf_counter_ns()
    return int(time.perf_counter() * 1_000_000_000)


def parse_size(raw: str) -> int:
    match = _SIZE_RE.match(raw)
    if not match:
        raise ValueError(f"invalid size: {raw!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    num_bytes = int(value * _UNIT_TO_BYTES[unit])
    if num_bytes <= 0:
        raise ValueError(f"size must be positive: {raw!r}")
    return num_bytes


def parse_size_list(raw: str) -> List[int]:
    if not raw.strip():
        raise ValueError("size list cannot be empty")

    sizes: List[int] = []
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            sizes.append(parse_size(part))
            continue

        fields = part.split(":")
        if len(fields) != 3 or not fields[2].lower().startswith("x"):
            raise ValueError(
                "range sizes must use START:END:xFACTOR, for example 1MB:1GB:x2"
            )
        start = parse_size(fields[0])
        end = parse_size(fields[1])
        factor = float(fields[2][1:])
        if start > end:
            raise ValueError(f"range start must be <= end: {part!r}")
        if factor <= 1:
            raise ValueError(f"range factor must be > 1: {part!r}")

        value = start
        while value <= end:
            sizes.append(value)
            next_value = int(value * factor)
            if next_value <= value:
                raise ValueError(f"range factor made no progress: {part!r}")
            value = next_value

    if not sizes:
        raise ValueError("size list cannot be empty")
    return sizes


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile for empty values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _bandwidth_gibps(num_bytes: int, latency_ms: Optional[float]) -> Optional[float]:
    if latency_ms is None or latency_ms <= 0:
        return None
    return (num_bytes / 1024**3) / (latency_ms / 1000)


def summarize_latencies_ms(latencies_ms: Iterable[float], num_bytes: int) -> dict:
    values = sorted(float(v) for v in latencies_ms)
    if not values:
        raise ValueError("cannot summarize empty latency list")

    p50 = _percentile(values, 0.50)
    p90 = _percentile(values, 0.90)
    p99 = _percentile(values, 0.99)
    mean = sum(values) / len(values)
    return {
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "latency_ms_p99": p99,
        "latency_ms_min": values[0],
        "latency_ms_max": values[-1],
        "bandwidth_GBps_p50": _bandwidth_gibps(num_bytes, p50),
        "bandwidth_GBps_mean": _bandwidth_gibps(num_bytes, mean),
    }


def write_target_info(path, info: TargetInfo) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info.as_dict(), indent=2) + "\n")


def load_target_info(path) -> TargetInfo:
    data = json.loads(Path(path).read_text())
    return TargetInfo(**data)


def parse_target_info_json(raw: str) -> TargetInfo:
    return TargetInfo(**json.loads(raw))


def write_csv_summary(path, rows: List[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_samples(path, samples: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample, sort_keys=True) + "\n")


def _configure_mooncake_env(protocol: str) -> None:
    os.environ["MOONCAKE_PROTOCOL"] = protocol
    if protocol.lower() == "tcp":
        os.environ["MC_FORCE_TCP"] = "1"


def _init_engine(host: str, gpu_id: int, ib_device: str, protocol: str):
    _configure_mooncake_env(protocol)
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        init_mooncake_transfer_engine,
    )

    return init_mooncake_transfer_engine(
        hostname=host,
        gpu_id=gpu_id,
        ib_device=ib_device,
    )


def _allocate_gpu_buffer(num_bytes: int, gpu_id: int):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this process")
    torch.cuda.set_device(gpu_id)
    tensor = torch.empty((num_bytes,), dtype=torch.uint8, device=f"cuda:{gpu_id}")
    torch.cuda.synchronize(gpu_id)
    return tensor


def _register_buffer(engine, ptr: int, num_bytes: int) -> float:
    start_ns = _perf_counter_ns()
    ret = engine.batch_register([ptr], [num_bytes])
    elapsed_ms = (_perf_counter_ns() - start_ns) / 1e6
    if ret != 0:
        raise RuntimeError(
            f"Mooncake batch_register failed: ret={ret}, ptr=0x{ptr:x}, bytes={num_bytes}"
        )
    return elapsed_ms


def _gbps_to_bytes_per_second(rate_limit_gbps: float) -> float:
    if rate_limit_gbps <= 0:
        raise ValueError(f"rate limit must be positive: {rate_limit_gbps}")
    return rate_limit_gbps * 1_000_000_000 / 8


def _sleep_until_rate(
    *,
    start_ns: int,
    bytes_sent: int,
    bytes_per_second: float,
    now_ns_fn: Callable[[], int],
    sleep_fn: Callable[[float], None],
) -> None:
    target_elapsed_s = bytes_sent / bytes_per_second
    elapsed_s = (now_ns_fn() - start_ns) / 1_000_000_000
    sleep_s = target_elapsed_s - elapsed_s
    if sleep_s > 0:
        sleep_fn(sleep_s)


def _transfer_sync_paced(
    engine,
    target: TargetInfo,
    src_ptr: int,
    num_bytes: int,
    chunk_size: int,
    rate_limit_gbps: Optional[float],
    now_ns_fn: Callable[[], int] = _perf_counter_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if num_bytes <= 0:
        raise ValueError(f"transfer bytes must be positive: {num_bytes}")
    if num_bytes > target.bytes:
        raise ValueError(
            f"requested transfer {format_bytes(num_bytes)} exceeds target "
            f"buffer {format_bytes(target.bytes)}"
        )
    if rate_limit_gbps is None:
        return engine.transfer_sync(target.session_id, src_ptr, target.ptr, num_bytes)
    if chunk_size <= 0:
        raise ValueError(f"chunk size must be positive: {chunk_size}")

    bytes_per_second = _gbps_to_bytes_per_second(rate_limit_gbps)
    start_ns = now_ns_fn()
    bytes_sent = 0
    while bytes_sent < num_bytes:
        length = min(chunk_size, num_bytes - bytes_sent)
        ret = engine.transfer_sync(
            target.session_id,
            src_ptr + bytes_sent,
            target.ptr + bytes_sent,
            length,
        )
        if ret != 0:
            return ret
        bytes_sent += length
        _sleep_until_rate(
            start_ns=start_ns,
            bytes_sent=bytes_sent,
            bytes_per_second=bytes_per_second,
            now_ns_fn=now_ns_fn,
            sleep_fn=sleep_fn,
        )
    return 0


def run_target(args: argparse.Namespace) -> int:
    max_bytes = parse_size(args.max_bytes)
    engine = _init_engine(args.host, args.gpu_id, args.ib_device, args.protocol)
    buffer = _allocate_gpu_buffer(max_bytes, args.gpu_id)
    register_ms = _register_buffer(engine, buffer.data_ptr(), max_bytes)
    resolved_ib_device = engine.get_ib_device()

    info = TargetInfo(
        session_id=engine.get_session_id(),
        host=args.host,
        gpu_id=args.gpu_id,
        ptr=buffer.data_ptr(),
        bytes=max_bytes,
        ib_device=args.ib_device if resolved_ib_device is None else resolved_ib_device,
        protocol=args.protocol,
    )
    write_target_info(args.target_info_file, info)

    print(f"target_info_file={args.target_info_file}", flush=True)
    print(f"target_register_ms={register_ms:.3f}", flush=True)
    print(f"TARGET_INFO_JSON={json.dumps(info.as_dict(), sort_keys=True)}")
    print("target_ready=true", flush=True)

    stop = False

    def _handle_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    while not stop:
        time.sleep(1)
    return 0


def _load_target_from_args(args: argparse.Namespace) -> TargetInfo:
    if args.target_info_json:
        return parse_target_info_json(args.target_info_json)
    return load_target_info(args.target_info_file)


def _sync_cuda(gpu_id: int) -> None:
    import torch

    torch.cuda.synchronize(gpu_id)


def _wait_until_unix_ns(start_at_unix_ns: int) -> None:
    if start_at_unix_ns <= 0:
        return
    while True:
        now_ns = _time_ns()
        remaining_s = (start_at_unix_ns - now_ns) / 1_000_000_000
        if remaining_s <= 0:
            return
        time.sleep(min(remaining_s, 0.050))


def run_background_initiator(
    args: argparse.Namespace,
    target: TargetInfo,
    engine,
    buffer,
    register_ms: float,
    resolved_ib_device: Optional[str],
    background_bytes: int,
    chunk_size: int,
) -> int:
    if args.background_duration_seconds <= 0:
        raise ValueError("--background-duration-seconds must be positive")

    _sync_cuda(args.gpu_id)
    _wait_until_unix_ns(args.start_at_unix_ns)
    start_unix_ns = _time_ns()
    start_ns = _perf_counter_ns()
    end_ns = start_ns + int(args.background_duration_seconds * 1_000_000_000)
    transfer_count = 0
    error_count = 0
    ok_bytes = 0

    while _perf_counter_ns() < end_ns:
        ret = _transfer_sync_paced(
            engine,
            target,
            buffer.data_ptr(),
            background_bytes,
            chunk_size,
            args.rate_limit_gbps,
        )
        if ret == 0:
            transfer_count += 1
            ok_bytes += background_bytes
        else:
            error_count += 1
            break

    _sync_cuda(args.gpu_id)
    end_ns_actual = _perf_counter_ns()
    end_unix_ns = _time_ns()
    elapsed_s = (end_ns_actual - start_ns) / 1_000_000_000
    row = {
        "mode": "background",
        "flow_id": args.flow_id,
        "source_host": args.host,
        "target_host": target.host,
        "source_gpu_id": args.gpu_id,
        "target_gpu_id": target.gpu_id,
        "source_ib_device": (
            args.ib_device if resolved_ib_device is None else resolved_ib_device
        ),
        "target_ib_device": target.ib_device,
        "protocol": args.protocol,
        "bytes_per_transfer": background_bytes,
        "human_bytes_per_transfer": format_bytes(background_bytes),
        "transfer_count": transfer_count,
        "error_count": error_count,
        "total_bytes": ok_bytes,
        "elapsed_s": elapsed_s,
        "start_time_unix_ns": start_unix_ns,
        "end_time_unix_ns": end_unix_ns,
        "start_time_monotonic_ns": start_ns,
        "end_time_monotonic_ns": end_ns_actual,
        "rate_limit_gbps": args.rate_limit_gbps,
        "chunk_size": chunk_size,
        "human_chunk_size": format_bytes(chunk_size),
        "register_ms": register_ms,
        "observed_payload_GBps": (
            (ok_bytes / 1024**3) / elapsed_s if elapsed_s > 0 else None
        ),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    write_csv_summary(args.summary_csv, [row])
    write_jsonl_samples(args.samples_jsonl, [])
    print(f"summary_csv={args.summary_csv}", flush=True)
    print(f"samples_jsonl={args.samples_jsonl}", flush=True)
    return 0 if error_count == 0 else 1


def run_initiator(args: argparse.Namespace) -> int:
    sizes = parse_size_list(args.sizes)
    target = _load_target_from_args(args)
    chunk_size = parse_size(args.chunk_size)
    background_bytes = (
        parse_size(args.background_bytes) if args.background_bytes else max(sizes)
    )
    max_transfer_bytes = (
        background_bytes if args.background_duration_seconds > 0 else max(sizes)
    )
    if args.background_duration_seconds < 0:
        raise ValueError("--background-duration-seconds cannot be negative")
    if max_transfer_bytes > target.bytes:
        raise ValueError(
            f"largest requested size {format_bytes(max_transfer_bytes)} exceeds target "
            f"buffer {format_bytes(target.bytes)}"
        )

    engine = _init_engine(args.host, args.gpu_id, args.ib_device, args.protocol)
    buffer = _allocate_gpu_buffer(max_transfer_bytes, args.gpu_id)
    register_ms = _register_buffer(engine, buffer.data_ptr(), max_transfer_bytes)
    resolved_ib_device = engine.get_ib_device()

    if args.background_duration_seconds > 0:
        return run_background_initiator(
            args,
            target,
            engine,
            buffer,
            register_ms,
            resolved_ib_device,
            background_bytes,
            chunk_size,
        )

    _wait_until_unix_ns(args.start_at_unix_ns)
    rows = []
    samples = []
    for num_bytes in sizes:
        for _ in range(args.warmup):
            ret = _transfer_sync_paced(
                engine,
                target,
                buffer.data_ptr(),
                num_bytes,
                chunk_size,
                args.rate_limit_gbps,
            )
            if ret != 0:
                raise RuntimeError(
                    f"warmup transfer failed: ret={ret}, bytes={num_bytes}, "
                    f"target_session={target.session_id}"
                )

        ok_latencies = []
        error_count = 0
        for iteration in range(args.repeat):
            _sync_cuda(args.gpu_id)
            start_unix_ns = _time_ns()
            start_ns = _perf_counter_ns()
            ret = _transfer_sync_paced(
                engine,
                target,
                buffer.data_ptr(),
                num_bytes,
                chunk_size,
                args.rate_limit_gbps,
            )
            _sync_cuda(args.gpu_id)
            end_ns = _perf_counter_ns()
            end_unix_ns = _time_ns()
            latency_ms = (end_ns - start_ns) / 1e6
            if ret == 0:
                ok_latencies.append(latency_ms)
            else:
                error_count += 1
            samples.append(
                {
                    "flow_id": args.flow_id,
                    "source_host": args.host,
                    "target_host": target.host,
                    "source_gpu_id": args.gpu_id,
                    "target_gpu_id": target.gpu_id,
                    "bytes": num_bytes,
                    "human_bytes": format_bytes(num_bytes),
                    "iteration": iteration,
                    "latency_ms": latency_ms,
                    "start_time_unix_ns": start_unix_ns,
                    "end_time_unix_ns": end_unix_ns,
                    "start_time_monotonic_ns": start_ns,
                    "end_time_monotonic_ns": end_ns,
                    "ret": ret,
                    "rate_limit_gbps": args.rate_limit_gbps,
                    "chunk_size": chunk_size,
                }
            )

        row = {
            "flow_id": args.flow_id,
            "source_host": args.host,
            "target_host": target.host,
            "source_gpu_id": args.gpu_id,
            "target_gpu_id": target.gpu_id,
            "source_ib_device": (
                args.ib_device if resolved_ib_device is None else resolved_ib_device
            ),
            "target_ib_device": target.ib_device,
            "protocol": args.protocol,
            "bytes": num_bytes,
            "human_bytes": format_bytes(num_bytes),
            "warmup_count": args.warmup,
            "repeat_count": args.repeat,
            "error_count": error_count,
            "register_ms": register_ms,
            "rate_limit_gbps": args.rate_limit_gbps,
            "chunk_size": chunk_size,
            "human_chunk_size": format_bytes(chunk_size),
        }
        if ok_latencies:
            row.update(summarize_latencies_ms(ok_latencies, num_bytes))
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    write_csv_summary(args.summary_csv, rows)
    write_jsonl_samples(args.samples_jsonl, samples)
    print(f"summary_csv={args.summary_csv}", flush=True)
    print(f"samples_jsonl={args.samples_jsonl}", flush=True)
    return 0 if all(row["error_count"] == 0 for row in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure SGLang/Mooncake GPU-buffer transfer latency by data size."
    )
    parser.add_argument("--role", choices=("target", "initiator"), required=True)
    parser.add_argument("--host", required=True, help="Local host IP reachable by peer.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ib-device", default=os.environ.get("IB_DEVICE", "mlx5_0"))
    parser.add_argument(
        "--protocol",
        default=os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
        help="Mooncake protocol, usually rdma or tcp.",
    )
    parser.add_argument(
        "--target-info-file",
        default="/tmp/kv-transfer-bench/target-info.json",
    )
    parser.add_argument(
        "--target-info-json",
        default=os.environ.get("TARGET_INFO_JSON"),
        help="Inline target JSON. Overrides --target-info-file for initiator.",
    )
    parser.add_argument("--max-bytes", default="2GB")
    parser.add_argument("--sizes", default="1MB:1GB:x2")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--flow-id",
        default=os.environ.get("FLOW_ID", ""),
        help="Optional logical flow identifier copied into summary and samples.",
    )
    parser.add_argument(
        "--start-at-unix-ns",
        type=int,
        default=int(os.environ.get("START_AT_UNIX_NS", "0") or "0"),
        help=(
            "Optional wall-clock Unix timestamp in nanoseconds. When set, the "
            "initiator waits until this time after setup/registration and before "
            "starting measured or background transfers."
        ),
    )
    parser.add_argument(
        "--rate-limit-gbps",
        type=float,
        default=None,
        help=(
            "Optional application-level Mooncake payload rate limit in Gbit/s. "
            "When set, each logical transfer is split into --chunk-size chunks "
            "and paced with sleeps."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        default="16MB",
        help="Chunk size used when --rate-limit-gbps is set.",
    )
    parser.add_argument(
        "--background-duration-seconds",
        type=float,
        default=0.0,
        help=(
            "When > 0, initiator runs a paced Mooncake background stream for "
            "this many seconds instead of latency measurements."
        ),
    )
    parser.add_argument(
        "--background-bytes",
        default=None,
        help=(
            "Logical transfer size for background mode. Defaults to the largest "
            "value from --sizes."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default="/tmp/kv-transfer-bench/summary.csv",
    )
    parser.add_argument(
        "--samples-jsonl",
        default="/tmp/kv-transfer-bench/samples.jsonl",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.role == "target":
        return run_target(args)
    if args.role == "initiator":
        return run_initiator(args)
    raise AssertionError(f"unhandled role: {args.role}")


if __name__ == "__main__":
    sys.exit(main())
