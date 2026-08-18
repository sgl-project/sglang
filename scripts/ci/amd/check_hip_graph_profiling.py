#!/usr/bin/env python3
"""Check that torch.profiler records device kernels launched by HIP/CUDA graph replay.

ROCm 7.2.0 (roctracer 4.1.70200) loses kernel-dispatch events for work submitted
through hipGraphLaunch: a trace captured while SGLang decodes -- which replays HIP
graphs -- shows the host-side launch but no GPU kernels under it. ROCm resolved the
roctracer reporting failure in 7.2.2 (https://github.com/ROCm/ROCm/issues/6102), so
this probe is what qualifies a candidate ROCm image before it is published.

On 7.2.0 the same combination can also wedge the HIP runtime inside
hipGraphLaunch instead of losing events, so each phase runs in a child process
under a timeout rather than inline.

Run inside the image under evaluation, on a machine with at least one GPU:

    python3 scripts/ci/amd/check_hip_graph_profiling.py

The eager phase is a control: it separates "this runtime cannot trace graph replay"
from "this runtime cannot trace anything". Exit status is 0 when graph-replay
kernels reach the trace, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile

RESULT_MARKER = "PROBE_RESULT "
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
TRACING_LIB_PATTERN = re.compile(
    r"lib(amdhip64|roctracer64|rocprofiler-sdk|rocprofiler-register|hsa-runtime64)[^/\s]*\.so[^/\s]*"
)


def loaded_tracing_libs() -> list[str]:
    """Sonames of the HIP/tracing libraries mapped into this process."""
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
    except OSError:
        return []
    return sorted({m.group(0) for m in TRACING_LIB_PATTERN.finditer(maps)})


def rocm_version() -> str | None:
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def phase_env() -> dict:
    import torch

    info: dict = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "rocm_version": rocm_version(),
        "device_count": 0,
    }
    if not torch.cuda.is_available():
        info["error"] = "no GPU visible to torch"
        return info

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    info.update(
        device_count=torch.cuda.device_count(),
        device_name=torch.cuda.get_device_name(0),
        gcn_arch=getattr(props, "gcnArchName", None),
        libs=loaded_tracing_libs(),
    )
    return info


def count_device_events(events: list[dict]) -> dict[str, int]:
    """Device-side event counts per Chrome-trace category, as Perfetto would show them."""
    counts: dict[str, int] = {}
    for event in events:
        category = str(event.get("cat", "")).lower()
        if category in DEVICE_CATEGORIES:
            counts[category] = counts.get(category, 0) + 1
    return counts


def phase_profile(use_graph: bool, replays: int, size: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"error": "no GPU visible to torch"}

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    a = torch.randn((size, size), device=device, dtype=torch.float16)
    b = torch.randn((size, size), device=device, dtype=torch.float16)
    out = torch.empty_like(a)

    # Graph capture requires the workload to have run on a non-default stream first.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            torch.matmul(a, b, out=out)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                torch.matmul(a, b, out=out)
        launch = graph.replay
    else:

        def launch() -> None:
            for _ in range(4):
                torch.matmul(a, b, out=out)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "probe.json")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(replays):
                launch()
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            events = json.load(f)["traceEvents"]

    counts = count_device_events(events)
    device_time_us = sum(
        getattr(item, "self_device_time_total", 0)
        or getattr(item, "self_cuda_time_total", 0)
        for item in prof.key_averages()
    )
    return {
        "kernels": counts.get("kernel", 0),
        "event_counts": counts,
        "trace_events": len(events),
        "self_device_time_us": device_time_us,
        "libs": loaded_tracing_libs(),
    }


def run_child(phase: str, args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--phase",
        phase,
        "--replays",
        str(args.replays),
        "--matmul-size",
        str(args.matmul_size),
    ]
    # New session so a HIP runtime wedged past SIGTERM can still be killed as a group.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output = proc.communicate(timeout=args.timeout)[0]
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output = proc.communicate()[0]
        return {"status": "timeout", "output": output}

    result = {"status": "ok" if proc.returncode == 0 else "error", "output": output}
    for line in output.splitlines():
        if line.startswith(RESULT_MARKER):
            result.update(json.loads(line[len(RESULT_MARKER) :]))
    return result


def report(label: str, result: dict) -> None:
    prefix = f"{label:<14}"
    if result.get("status") == "timeout":
        print(f"{prefix}TIMEOUT (no result within the phase timeout)")
    elif result.get("kernels") is None:
        print(f"{prefix}ERROR {result.get('error', '')}".rstrip())
    else:
        print(
            f"{prefix}{result['kernels']} kernel events, "
            f"{result['self_device_time_us']:.0f} us device time, "
            f"{result['trace_events']} trace events"
        )
        return
    if result.get("output"):
        print("--- phase output ---")
        print(result["output"].rstrip())
        print("--- end phase output ---")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "env", "eager", "graph"),
        default="all",
        help="internal: 'all' orchestrates the others in child processes",
    )
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--matmul-size", type=int, default=1024)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds allowed per phase; a wedged graph launch is reported as a timeout",
    )
    args = parser.parse_args()

    if args.phase == "env":
        info = phase_env()
        print(RESULT_MARKER + json.dumps(info))
        return 1 if "error" in info else 0
    if args.phase in ("eager", "graph"):
        result = phase_profile(args.phase == "graph", args.replays, args.matmul_size)
        print(RESULT_MARKER + json.dumps(result))
        return 1 if "error" in result else 0

    env = run_child("env", args)
    if env.get("status") != "ok" or not env.get("device_count"):
        reason = env.get("error") or "the environment phase did not report a usable GPU"
        print(f"environment: unusable -- {reason}")
        if env.get("output"):
            print(env["output"].rstrip())
        return 1

    print(f"torch {env['torch']} (hip {env['torch_hip']}), rocm {env['rocm_version']}")
    print(f"device: {env.get('device_name')!r} ({env.get('gcn_arch')})")
    print("hip/tracing libs: " + ", ".join(env.get("libs") or ["<none mapped>"]))
    if not (env.get("device_name") or "").strip():
        print(
            "warning: the device reports no marketing name, so device-name-keyed tuned "
            "configs will not resolve. The image is missing libdrm-amdgpu-common "
            "(https://github.com/ROCm/ROCm/issues/5992)."
        )

    eager = run_child("eager", args)
    report("eager launch:", eager)
    graph = run_child("graph", args)
    report("graph replay:", graph)
    if graph.get("libs"):
        print("tracing libs while profiling: " + ", ".join(graph["libs"]))

    if not eager.get("kernels"):
        print(
            "\nVERDICT: FAIL -- the profiler records no kernels even for eager launches, "
            "so this runtime cannot trace GPU work at all and the graph phase says nothing "
            "about graph support."
        )
        return 1
    if graph.get("status") == "timeout":
        print(
            "\nVERDICT: FAIL -- graph replay under the profiler never finished. "
            "ROCm 7.2.0 can deadlock in hipGraphLaunch while a profiler session is "
            "attached; use a ROCm 7.2.2 or newer runtime."
        )
        return 1
    if graph.get("kernels") is None:
        print(
            "\nVERDICT: FAIL -- graph replay under the profiler did not complete; see the "
            "phase output above."
        )
        return 1
    if graph["kernels"] == 0:
        print(
            "\nVERDICT: FAIL -- eager kernels are traced but graph-replay kernels are "
            "missing, which is the roctracer reporting failure fixed in ROCm 7.2.2 "
            "(https://github.com/ROCm/ROCm/issues/6102). Profile with --disable-cuda-graph "
            "on this image, or move to a ROCm 7.2.2 or newer runtime."
        )
        return 1

    print("\nVERDICT: PASS -- graph-replay kernels reach the trace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
