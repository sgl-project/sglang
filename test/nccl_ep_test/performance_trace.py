"""Read nsys SQLite: per-step kernel activity and replay allocation checks.

Kernel interval overlap does not measure networking progress between send and
complete. The nsys GUI is still needed for a critical-path diagnosis.
"""

import bisect
import collections
import functools
import json
import re
import sqlite3


def annotate_replays():
    import torch

    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    original = FullCudaGraphBackend.replay

    @functools.wraps(original)
    def replay(self, *args, **kwargs):
        with torch.cuda.nvtx.range("nccl_ep_performance/replay"):
            return original(self, *args, **kwargs)

    FullCudaGraphBackend.replay = replay


def union_ns(intervals):
    total, previous = 0, None
    for start, end in sorted(intervals):
        if previous is None or start > previous:
            total += end - start
        else:
            total += max(0, end - previous)
        previous = max(previous or end, end)
    return total


def analyze(path, expected_steps=8):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        windows = collections.defaultdict(list)
        replays = collections.defaultdict(list)
        for start, end, tid, label, text_id in db.execute(
            "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS ORDER BY start"
        ):
            label = label or names.get(text_id, "")
            if label == "nccl_ep_performance/replay":
                replays[tid].append((start, end))
            if label.startswith(("nccl_ep_model/", "nccl_ep_pipeline/")):
                rank = int(re.search(r"rank=(\d+)", label)[1])
                if end is None or end < start:
                    raise ValueError("Incomplete NVTX step")
                windows[rank].append((start, end, label))
        if set(windows) != {0, 1} or any(
            len(w) != expected_steps for w in windows.values()
        ):
            raise ValueError("Missing dual-rank profiled steps")
        starts = {tid: [x[0] for x in spans] for tid, spans in replays.items()}
        runtime_calls = collections.Counter()
        for start, end, tid, name in db.execute(
            "SELECT start,end,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            if tid in starts:
                index = bisect.bisect_right(starts[tid], start) - 1
                if index >= 0 and end <= replays[tid][index][1]:
                    runtime_calls[names[name]] += 1
        launches = sum(v for k, v in runtime_calls.items() if "GraphLaunch" in k)
        if (
            launches != expected_steps * 2
            or sum(map(len, replays.values())) != launches
        ):
            raise ValueError("Missing Graph replay calls")
        forbidden = {
            k: v
            for k, v in runtime_calls.items()
            if re.search(
                r"Malloc|MemAlloc|Free|GraphInstantiate|GraphDestroy|ModuleLoad|LibraryLoad",
                k,
            )
        }
        if forbidden:
            raise ValueError(f"Unexpected replay resource operations: {forbidden}")
        kernels = collections.defaultdict(list)
        for device, start, end, name, graph in db.execute(
            "SELECT deviceId,start,end,shortName,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        ):
            kernels[device].append((start, end, names[name], graph))
        rows = []
        for rank, spans in windows.items():
            for start, end, label in spans:
                selected = [k for k in kernels[rank] if k[0] >= start and k[1] <= end]
                counts = collections.Counter(k[2] for k in selected)
                graph_counts = collections.Counter(k[2] for k in selected if k[3])
                for operation in ("dispatch", "combine"):
                    if not any(f"ll_{operation}_kernel" in k for k in graph_counts):
                        raise ValueError("Native EP missing from profiled graph")
                if not any("fused_moe" in k for k in graph_counts):
                    raise ValueError("Expert GEMM missing from profiled graph")
                ep = [
                    (k[0], k[1]) for k in selected if k[2].startswith("nccl_ep_jit_ll_")
                ]
                other = [
                    (k[0], k[1])
                    for k in selected
                    if not k[2].startswith("nccl_ep_jit_ll_")
                ]
                busy = union_ns(ep + other)
                rows.append(
                    dict(
                        rank=rank,
                        label=label,
                        step_ns=end - start,
                        kernel_busy_ns=busy,
                        no_kernel_ns=end - start - busy,
                        ep_kernel_ns=union_ns(ep),
                        other_kernel_ns=union_ns(other),
                        ep_other_kernel_overlap_ns=union_ns(ep)
                        + union_ns(other)
                        - busy,
                        kernel_counts=dict(counts),
                        graph_kernel_counts=dict(graph_counts),
                    )
                )
        return dict(
            passed=True,
            steps=rows,
            graph_launches=launches,
            forbidden_calls_in_replay=forbidden,
            scope="Profiled kernel intervals inside synchronized step ranges, not unprofiled latency estimates. Other kernels include attention, GEMMs, copies and collectives. Network progress between send/complete is not measured by kernel overlap.",
        )
    finally:
        db.close()


def export_and_analyze(directory):
    import subprocess

    database = directory / "trace.sqlite"
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--output",
            str(database),
            str(directory / "trace.nsys-rep"),
        ],
        check=True,
        timeout=180,
    )
    result = analyze(database)
    (directory / "trace-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
