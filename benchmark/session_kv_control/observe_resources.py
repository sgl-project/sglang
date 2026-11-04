#!/usr/bin/env python3
"""Read-only resource/Prometheus sampler for externally managed benchmark services."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request


def fetch(url):
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read().decode()
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass
        return {"url": url, "body": body, "elapsed_s": time.monotonic() - start}
    except Exception as error:
        return {"url": url, "error": str(error), "elapsed_s": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", action="append", default=[])
    parser.add_argument("--master", help="Mooncake Master metrics URL")
    parser.add_argument("--gpu-indices", default="0,1")
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    urls = [url.rstrip("/") + path for url in args.engine for path in ("/metrics", "/v1/loads")]
    if args.master:
        urls.append(args.master)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as output, concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
        while not stopped.is_set():
            start = time.monotonic()
            pending = list(pool.map(fetch, urls))
            try:
                gpu = subprocess.run(
                    ["nvidia-smi", "-i", args.gpu_indices,
                     "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                ).stdout
            except Exception as error:
                gpu = {"error": str(error)}
            output.write(json.dumps({"wall_time": time.time(), "monotonic": start,
                                     "sample_elapsed_s": time.monotonic() - start,
                                     "loadavg": os.getloadavg(), "gpu_csv": gpu, "http": pending}) + "\n")
            stopped.wait(max(0, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
