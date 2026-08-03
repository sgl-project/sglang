import statistics
import time

import torch

from sglang.srt.layers.moe.dwdp.hsa_copy import HsaSdmaCopyEngine


def _measure(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def main():
    peer_count = min(torch.cuda.device_count() - 1, 3)
    if peer_count < 1:
        raise RuntimeError("HSA copy benchmark requires at least two GPUs")

    size_bytes = 256 * 1024 * 1024
    sources = [
        torch.randint(
            0,
            256,
            (size_bytes,),
            dtype=torch.uint8,
            device=f"cuda:{peer}",
        )
        for peer in range(1, peer_count + 1)
    ]
    destinations = [
        torch.empty(size_bytes, dtype=torch.uint8, device="cuda:0") for _ in sources
    ]
    streams = [torch.cuda.Stream(device="cuda:0") for _ in sources]

    # Establish peer access before bypassing HIP memcpy.
    for destination, source in zip(destinations, sources):
        destination.copy_(source)
    torch.cuda.synchronize()

    engine = HsaSdmaCopyEngine()
    selected_engines = {
        peer: engine.engine_for_devices(0, peer) for peer in range(1, peer_count + 1)
    }

    def run_hsa():
        tickets = [
            engine.submit(
                destination,
                source,
                destination_device=0,
                source_device=peer,
            )
            for peer, (destination, source) in enumerate(
                zip(destinations, sources), start=1
            )
        ]
        engine.wait_all(tickets)

    def run_hip():
        events = []
        for stream, destination, source in zip(streams, destinations, sources):
            with torch.cuda.stream(stream):
                destination.copy_(source, non_blocking=True)
                events.append(stream.record_event())
        for event in events:
            event.synchronize()

    hsa_ms = _measure(run_hsa)
    hip_ms = _measure(run_hip)
    total_gib = size_bytes * peer_count / (1024**3)
    print(
        {
            "peers": peer_count,
            "bytes_per_peer": size_bytes,
            "selected_engines": selected_engines,
            "hsa_median_ms": hsa_ms,
            "hsa_aggregate_gib_s": total_gib / (hsa_ms / 1000),
            "hip_median_ms": hip_ms,
            "hip_aggregate_gib_s": total_gib / (hip_ms / 1000),
            "speedup": hip_ms / hsa_ms,
        }
    )


if __name__ == "__main__":
    main()
