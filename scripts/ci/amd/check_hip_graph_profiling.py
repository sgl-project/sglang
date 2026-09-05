#!/usr/bin/env python3
"""Check that torch.profiler records device kernels launched by HIP/CUDA graph replay.

ROCm 7.2.0 (roctracer 4.1.70200) loses kernel-dispatch events for work submitted
through hipGraphLaunch: a trace captured while SGLang decodes -- which replays HIP
graphs -- comes back missing kernels. ROCm resolved the roctracer reporting failure
in 7.2.2 (https://github.com/ROCm/ROCm/issues/6102).

Having the fix in the image is not the same as running it. The torch wheels vendor
their own HIP and roctracer in torch/lib, and libtorch_hip.so carries RPATH $ORIGIN,
so a ROCm 7.2.4 image can still profile through 7.2.0 libraries; LD_LIBRARY_PATH
does not override that. This probe reports which libraries torch actually mapped and
prints the LD_PRELOAD that forces the ROCm install's copies.

Run inside the image under evaluation, on a machine with at least one GPU:

    python3 scripts/ci/amd/check_hip_graph_profiling.py

The loss is partial, not total: on ROCm 7.2.0 a 4-node graph is traced correctly
while a 64-node graph drops events, so the check counts every dispatch it asked
for rather than looking for a non-empty trace. The eager phase is a control,
separating "this runtime cannot trace graph replay" from "this runtime cannot
trace anything", and each phase runs in a child process under a timeout because
7.2.0 can also wedge inside hipGraphLaunch instead of losing events.

Exit status is 0 when every graph-replay kernel reaches the trace, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
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
    r"\S*lib(?:amdhip64|roctracer64|rocprofiler-sdk|rocprofiler-register|hsa-runtime64)[^/\s]*\.so[^/\s]*"
)
# The libraries whose version decides whether graph replay is traced.
RUNTIME_LIBS = ("libamdhip64", "libroctracer64")
# Options that shape the workload, so the phase children need all of them.
PHASE_OPTIONS = ("replays", "matmul_size", "graph_nodes")


def loaded_tracing_libs() -> list[str]:
    """Paths of the HIP/tracing libraries mapped into this process.

    Paths, not sonames: a ROCm 7.2.4 image can still be running the 7.2.0 copies
    that the torch wheel vendors in torch/lib, and only the path shows that.
    """
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
    except OSError:
        return []
    return sorted({m.group(0) for m in TRACING_LIB_PATTERN.finditer(maps)})


def vendored_runtime_libs(
    libs: list[str], rocm_lib_dir: str = "/opt/rocm/lib"
) -> list[str]:
    """Mapped HIP/tracing libraries that are not the ROCm install's build.

    `libtorch_hip.so` carries `RPATH $ORIGIN` and needs `libamdhip64.so` /
    `libroctracer64.so`, so the loader takes the wheel's copies and
    LD_LIBRARY_PATH does not override them. The image's ROCm version then says
    nothing about what torch.profiler is actually using.

    Neither the path nor the file name settles it, so both are checked against
    the ROCm install: a preload maps its copy alongside the wheel's and
    interposes, and an image can have the ROCm library copied over the wheel's
    path, where only the size gives it away.
    """
    rocm_sizes: dict[str, set[int]] = {}
    for name in RUNTIME_LIBS:
        sizes = set()
        for path in glob.glob(os.path.join(rocm_lib_dir, f"{name}.so.*")):
            if not os.path.islink(path):
                try:
                    sizes.add(os.path.getsize(path))
                except OSError:
                    pass
        rocm_sizes[name] = sizes

    flagged = []
    for name in RUNTIME_LIBS:
        mapped = [lib for lib in libs if name in lib]
        if not mapped or any(lib.startswith("/opt/rocm") for lib in mapped):
            continue
        for lib in mapped:
            try:
                is_rocm_build = os.path.getsize(lib) in rocm_sizes[name]
            except OSError:
                is_rocm_build = False
            if not is_rocm_build:
                flagged.append(lib)
    return flagged


def rocm_runtime_preload(lib_dir: str = "/opt/rocm/lib") -> str | None:
    """LD_PRELOAD value that forces the ROCm install's HIP and roctracer."""
    resolved = []
    for name in RUNTIME_LIBS:
        # The unversioned names are symlinks; preload needs the real files, and
        # naming them keeps the value correct across ROCm patch releases.
        versioned = [
            path
            for path in sorted(glob.glob(os.path.join(lib_dir, f"{name}.so.*")))
            if not os.path.islink(path)
        ]
        if not versioned:
            return None
        resolved.append(versioned[-1])
    return ":".join(resolved)


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


def phase_profile(use_graph: bool, replays: int, size: int, nodes: int) -> dict:
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
            for _ in range(nodes):
                torch.matmul(a, b, out=out)
        launch = graph.replay
    else:

        def launch() -> None:
            for _ in range(nodes):
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
        # One matmul is one dispatch, so a healthy runtime reports every one of
        # them. Loss under graph replay is partial -- 448 of 512 in the ROCm
        # 7.2.0 measurement on #35390 -- so a threshold of "more than zero"
        # would call a broken runtime healthy.
        "expected_kernels": nodes * replays,
        "event_counts": counts,
        "trace_events": len(events),
        "self_device_time_us": device_time_us,
        "libs": loaded_tracing_libs(),
    }


def child_command(phase: str, args: argparse.Namespace) -> list[str]:
    """The child invocation for one phase, carrying every workload option.

    Built from PHASE_OPTIONS rather than by hand: a knob that the parent accepts
    and forgets to forward is silently ignored, and the child's own default then
    stands in for it.
    """
    cmd = [sys.executable, os.path.abspath(__file__), "--phase", phase]
    for name in PHASE_OPTIONS:
        cmd += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
    return cmd


def run_child(phase: str, args: argparse.Namespace) -> dict:
    cmd = child_command(phase, args)
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
            f"{prefix}{result['kernels']}/{result['expected_kernels']} kernel events, "
            f"{result['self_device_time_us']:.0f} us device time, "
            f"{result['trace_events']} trace events"
        )
        return
    if result.get("output"):
        print("--- phase output ---")
        print(result["output"].rstrip())
        print("--- end phase output ---")


def traced_everything(result: dict) -> bool:
    kernels, expected = result.get("kernels"), result.get("expected_kernels")
    return kernels is not None and expected is not None and kernels >= expected


PARENT_ONLY_OPTIONS = ("phase", "timeout", "print_ld_preload")


def build_parser() -> argparse.ArgumentParser:
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
        "--graph-nodes",
        type=int,
        default=64,
        help="dispatches per graph. 64 is the calibrated value: ROCm 7.2.0 traces a "
        "4-node graph completely and only starts dropping around 16, while 7.2.4 is "
        "complete at 64 across repeated runs. Raising it much further is not a "
        "stricter test -- 7.2.4 has been seen dropping events at 256 -- so a failure "
        "there does not mean the same thing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds allowed per phase; a wedged graph launch is reported as a timeout",
    )
    parser.add_argument(
        "--print-ld-preload",
        action="store_true",
        help="print the LD_PRELOAD that forces the ROCm install's HIP and roctracer, "
        "for callers that have to set it before starting the process, and exit",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.print_ld_preload:
        preload = rocm_runtime_preload()
        if not preload:
            print("no versioned HIP/roctracer under /opt/rocm/lib", file=sys.stderr)
            return 1
        print(preload)
        return 0

    if args.phase == "env":
        info = phase_env()
        print(RESULT_MARKER + json.dumps(info))
        return 1 if "error" in info else 0
    if args.phase in ("eager", "graph"):
        result = phase_profile(
            args.phase == "graph", args.replays, args.matmul_size, args.graph_nodes
        )
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

    vendored = vendored_runtime_libs(graph.get("libs") or env.get("libs") or [])
    if graph.get("libs") and graph["libs"] != env.get("libs"):
        print("tracing libs while profiling: " + ", ".join(graph["libs"]))
    if vendored:
        preload = rocm_runtime_preload()
        print(
            "note: torch is using its own HIP/roctracer, not the ROCm install's: "
            + ", ".join(vendored)
        )
        if preload:
            print(
                "      to fix the image, overwrite those two files with the ROCm ones:"
            )
            for name, source in zip(RUNTIME_LIBS, preload.split(":")):
                for destination in (lib for lib in vendored if name in lib):
                    print(f"        cp -a {source} {destination}")
            print(f'      to check this one run only: LD_PRELOAD="{preload}"')
            print(
                "      a preload leaves both copies mapped, which is fine here but has"
            )
            print("      been seen to break a full server run during graph capture")

    if not traced_everything(eager):
        print(
            "\nVERDICT: FAIL -- the profiler did not record every eager launch, so this "
            "runtime cannot trace GPU work reliably at all and the graph phase says "
            "nothing about graph support."
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
    if not traced_everything(graph):
        missing = graph["expected_kernels"] - graph["kernels"]
        print(
            f"\nVERDICT: FAIL -- eager launches are traced in full but {missing} of "
            f"{graph['expected_kernels']} graph-replay kernels never reached the trace. "
            "That is the roctracer reporting failure fixed in ROCm 7.2.2 "
            "(https://github.com/ROCm/ROCm/issues/6102)."
        )
        if vendored:
            print(
                "The ROCm install is not what torch loaded, so replace the wheel's copies "
                "as printed above and re-run before concluding the image is unfixable."
            )
        else:
            print(
                "Profile with --disable-cuda-graph on this image, or move to a ROCm 7.2.2 "
                "or newer runtime."
            )
        return 1

    print("\nVERDICT: PASS -- every graph-replay kernel reached the trace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
