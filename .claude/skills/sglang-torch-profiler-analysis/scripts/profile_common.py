"""Shared helpers for SGLang torch-profiler skill scripts."""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request

STAGE_ORDER = {"extend": 0, "prefill": 0, "decode": 1, "all": 2}
TRACE_METADATA_NAMES = {
    "process_name",
    "thread_name",
    "process_sort_index",
    "thread_sort_index",
}
NON_KERNEL_TRACE_CATEGORIES = ("python_function", "cpu_op", "trace")
PYTHON_SCOPE_NAME_PREFIXES = ("python/", "nn.module:")


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_repo_relative_path(path: object) -> str:
    text = normalize_text(path).replace("\\", "/")
    for marker in ("python/sglang/", "sgl_kernel/"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].lstrip("/")
    idx = text.find("sglang/")
    if idx != -1:
        return ("python/" + text[idx:]).lstrip("/")
    return text.lstrip("/")


def contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def coerce_optional_int(value: object) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def extract_trace_events(trace: object) -> Sequence[dict]:
    if isinstance(trace, dict):
        events = trace.get("traceEvents", [])
        return events if isinstance(events, list) else []
    if isinstance(trace, list):
        return trace
    return []


def is_trace_metadata_name(name: object) -> bool:
    return str(name) in TRACE_METADATA_NAMES


def is_complete_duration_event(event: dict) -> bool:
    if event.get("ph") != "X":
        return False
    dur = event.get("dur")
    ts = event.get("ts")
    if dur is None or ts is None:
        return False
    try:
        return float(dur) > 0
    except (TypeError, ValueError):
        return False


def is_annotation_event(name: object, category: object) -> bool:
    lowered_name = normalize_text(name).lower()
    lowered_category = normalize_text(category).lower()
    return "annotation" in lowered_category or lowered_name.startswith("## call ")


def is_non_kernel_trace_category(category: object) -> bool:
    lowered_category = normalize_text(category).lower()
    return any(token in lowered_category for token in NON_KERNEL_TRACE_CATEGORIES)


def looks_like_python_scope_name(name: object) -> bool:
    lowered_name = normalize_text(name).lower()
    return ".py(" in lowered_name or lowered_name.startswith(PYTHON_SCOPE_NAME_PREFIXES)


def has_stream_marker(args: Optional[dict]) -> bool:
    trace_args = args or {}
    return "stream" in trace_args or "cuda_stream" in trace_args


def load_trace_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_server_args(path: Path) -> Optional[dict]:
    resolved = path.resolve()
    candidate_dirs: List[Path] = []
    if resolved.is_file():
        candidate_dirs.extend([resolved.parent, resolved.parent.parent])
    else:
        candidate_dirs.extend([resolved, resolved.parent])

    seen: set[Path] = set()
    for candidate_dir in candidate_dirs:
        if candidate_dir in seen:
            continue
        seen.add(candidate_dir)
        candidate = candidate_dir / "server_args.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def parse_stage(path: Path) -> str:
    name = path.name.lower()
    if "-extend" in name or "-prefill" in name:
        return "extend"
    if "-decode" in name:
        return "decode"
    return "all"


def parse_tp_rank(path: Path) -> Optional[int]:
    match = re.search(r"TP-(\d+)", path.name)
    return int(match.group(1)) if match else None


def newest_trace_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    direct = list(path.glob("*.trace.json")) + list(path.glob("*.trace.json.gz"))
    if direct:
        return path
    child_candidates = [item for item in path.rglob("*") if item.is_dir()]
    trace_dirs = [
        candidate
        for candidate in child_candidates
        if list(candidate.glob("*.trace.json"))
        or list(candidate.glob("*.trace.json.gz"))
    ]
    if not trace_dirs:
        raise FileNotFoundError(f"No trace files found under {path}")
    trace_dirs.sort(key=lambda item: item.stat().st_mtime)
    return trace_dirs[-1]


def discover_trace_targets(
    path: Path, all_traces: bool
) -> Tuple[List[Path], Optional[dict]]:
    if path.is_file():
        return [path], load_server_args(path)

    trace_dir = newest_trace_dir(path)
    traces = sorted(
        list(trace_dir.glob("*.trace.json")) + list(trace_dir.glob("*.trace.json.gz")),
        key=lambda item: item.stat().st_mtime,
    )
    if not traces:
        raise FileNotFoundError(f"No trace files found under {trace_dir}")

    non_merged = [trace for trace in traces if not trace.name.startswith("merged-")]
    selected = non_merged or traces
    if not all_traces:
        ranks = sorted(
            {
                rank
                for rank in (parse_tp_rank(trace) for trace in selected)
                if rank is not None
            }
        )
        if ranks:
            rank = 0 if 0 in ranks else ranks[0]
            selected = [trace for trace in selected if parse_tp_rank(trace) == rank]
        grouped: Dict[str, List[Path]] = defaultdict(list)
        for trace in selected:
            grouped[parse_stage(trace)].append(trace)
        selected = [
            sorted(group, key=lambda item: item.stat().st_mtime)[-1]
            for group in grouped.values()
        ]

    selected.sort(key=lambda item: (STAGE_ORDER.get(parse_stage(item), 99), item.name))
    return selected, load_server_args(trace_dir)


def post_json(url: str, payload: dict, timeout: float = 60.0) -> Optional[dict]:
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else None


def send_probe_request(
    url: str, prompt: str, max_new_tokens: int, sampling_seed: int
) -> None:
    payload = {
        "text": prompt,
        "sampling_params": {
            "sampling_seed": sampling_seed,
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }
    post_json(url.rstrip("/") + "/generate", payload, timeout=300.0)


def run_profiler(
    url: str,
    output_dir: Optional[str],
    num_steps: int,
    profile_by_stage: bool,
    merge_profiles: bool,
    profile_prefix: Optional[str],
    probe_requests: int,
    probe_prompt: str,
    probe_max_new_tokens: Optional[int],
    probe_delay: float,
    start_step: Optional[int] = None,
) -> Path:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sglang-torch-profile-")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "sglang.profiler",
        "--url",
        url,
        "--output-dir",
        str(output_path),
        "--num-steps",
        str(num_steps),
        "--cpu",
        "--gpu",
        "--merge-profiles" if merge_profiles else "--no-merge-profiles",
        "--profile-by-stage" if profile_by_stage else "--no-profile-by-stage",
    ]
    if profile_prefix:
        cmd.extend(["--profile-prefix", profile_prefix])
    if start_step is not None:
        cmd.extend(["--start-step", str(start_step)])

    profiler_proc = subprocess.Popen(cmd)
    try:
        if probe_requests > 0:
            time.sleep(max(0.0, probe_delay))
            effective_max_new_tokens = probe_max_new_tokens or max(64, num_steps * 8)
            for request_idx in range(probe_requests):
                send_probe_request(
                    url=url,
                    prompt=probe_prompt,
                    max_new_tokens=effective_max_new_tokens,
                    sampling_seed=request_idx,
                )
                if profiler_proc.poll() is not None:
                    break
        return_code = profiler_proc.wait()
    finally:
        if profiler_proc.poll() is None:
            profiler_proc.kill()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    deadline = time.time() + 15.0
    while time.time() < deadline:
        child_dirs = [path for path in output_path.iterdir() if path.is_dir()]
        if child_dirs:
            child_dirs.sort(key=lambda path: path.stat().st_mtime)
            newest_child = child_dirs[-1]
            if any(newest_child.glob("*.trace.json*")):
                return newest_child
        time.sleep(0.5)

    child_dirs = [path for path in output_path.iterdir() if path.is_dir()]
    if child_dirs:
        child_dirs.sort(key=lambda path: path.stat().st_mtime)
        return child_dirs[-1]
    return output_path


def select_heaviest_pid(
    events: Sequence[dict],
    event_filter: Callable[[dict], bool],
    pid_substring: Optional[str] = None,
    preferred_substrings: Iterable[str] = (),
) -> Optional[str]:
    durations: Counter = Counter()
    for event in events:
        if not event_filter(event):
            continue
        pid = str(event.get("pid"))
        if pid_substring and pid_substring not in pid:
            continue
        durations[pid] += float(event["dur"])
    if not durations:
        return None

    for substring in preferred_substrings:
        preferred = [pid for pid in durations if substring in pid]
        if preferred:
            return max(preferred, key=lambda pid: durations[pid])
    return max(durations, key=lambda pid: durations[pid])
