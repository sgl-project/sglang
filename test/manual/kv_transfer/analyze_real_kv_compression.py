"""Offline nvCOMP experiment on a bounded, exported real-KV byte batch.

This does not generate synthetic KV or change the online object granularity.
Run in the GPU image with --input /path/to/kv-PID.pt --output result.json.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-pages", default="1,4,16,64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from sglang.srt.kv_compression.backend import NvcompLZ4Backend

    capture = torch.load(args.input, map_location="cpu", weights_only=True)
    raw_cpu = capture["raw"]
    if raw_cpu.dtype != torch.uint8 or raw_cpu.ndim != 2 or not len(raw_cpu):
        raise ValueError("Expected a nonempty page-major uint8 capture")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    backend = NvcompLZ4Backend(device, stream)
    results = []
    with torch.cuda.stream(stream):
        raw = raw_cpu.to(device)
        for group in map(int, args.group_pages.split(",")):
            if group <= 0:
                raise ValueError("Object group sizes must be positive")
            if group > len(raw):
                continue
            sources = [raw[i : i + group].flatten() for i in range(0, len(raw), group)]
            capacities = [backend.max_output_bytes(t) for t in sources]
            encoded = [
                torch.empty(n, dtype=torch.uint8, device=device) for n in capacities
            ]
            restored = [torch.empty_like(t) for t in sources]
            for repeat in range(-1, args.repeats):  # one unreported warmup
                begin, compressed, decompressed = [
                    torch.cuda.Event(enable_timing=True) for _ in range(3)
                ]
                begin.record(stream)
                started = time.perf_counter()
                lengths = backend.compress_batch(sources, encoded)
                compressed.record(stream)
                compress_wall_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                backend.decompress_batch(
                    [t[:n] for t, n in zip(encoded, lengths)], restored
                )
                decompressed.record(stream)
                stream.synchronize()
                decompress_wall_ms = (time.perf_counter() - started) * 1000
                if not all(torch.equal(a, b) for a, b in zip(sources, restored)):
                    raise AssertionError("Real KV byte restoration mismatch")
                if repeat >= 0:
                    # Actual sizes, capacity bounds, and hypothetical aligned
                    # transport sizes are distinct quantities.
                    wire = (
                        sum((n + 255) // 256 * 256 for n in lengths[:-1]) + lengths[-1]
                    )
                    results.append(
                        dict(
                            group_pages=group,
                            repeat=repeat,
                            raw_bytes=raw.numel(),
                            actual_encoded_bytes=sum(lengths),
                            aligned_payload_bytes=wire,
                            capacity_bytes=sum(capacities),
                            actual_lengths=lengths,
                            byte_equal=True,
                            compress_wall_ms=compress_wall_ms,
                            decompress_wall_ms=decompress_wall_ms,
                            compress_stream_ms=begin.elapsed_time(compressed),
                            decompress_stream_ms=compressed.elapsed_time(decompressed),
                        )
                    )
    record = dict(
        input=str(args.input),
        raw_sha256=hashlib.sha256(raw_cpu.numpy().tobytes()).hexdigest(),
        layout=capture["layout"],
        source_pages=len(raw_cpu),
        page_bytes=raw_cpu.shape[1],
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device),
        nvcomp=importlib.metadata.version(
            "nvidia-nvcomp-cu" + torch.version.cuda.split(".")[0]
        ),
        note="Diagnostic capture; no serving speedup claim. Aligned bytes exclude descriptor JSON.",
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
