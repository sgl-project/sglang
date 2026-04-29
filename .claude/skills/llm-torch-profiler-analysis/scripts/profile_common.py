"""Shared helpers for unified LLM torch-profiler skill scripts."""

from __future__ import annotations

import gzip
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request

STAGE_ORDER = {"extend": 0, "prefill": 0, "decode": 1, "all": 2}
FRAMEWORK_LABELS = {
    "auto": "auto",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "trtllm": "TensorRT-LLM",
}
TRACE_FILE_PATTERNS = (
    "*.trace.json",
    "*.trace.json.gz",
    "*.pt.trace.json",
    "*.pt.trace.json.gz",
    "*.json",
    "*.json.gz",
)
TRACE_FILE_IGNORE_NAMES = {
    "server_args.json",
    "metadata.json",
    "config.json",
}
TRACE_METADATA_NAMES = {
    "process_name",
    "thread_name",
    "process_sort_index",
    "thread_sort_index",
}
NON_KERNEL_TRACE_CATEGORIES = ("python_function", "cpu_op", "trace")
PYTHON_SCOPE_NAME_PREFIXES = ("python/", "nn.module:")


@lru_cache(maxsize=65536)
def _normalize_text_cached(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    for token in (" ", "\t", "\n", "\r", "\v", "\f"):
        if token in text:
            return " ".join(text.split())
    return text


def normalize_text(value: object) -> str:
    return _normalize_text_cached(value if isinstance(value, str) else str(value))


def canonicalize_framework(value: object) -> str:
    lowered = normalize_text(value).lower().replace("_", "-")
    aliases = {
        "": "auto",
        "auto": "auto",
        "sglang": "sglang",
        "sgl": "sglang",
        "vllm": "vllm",
        "trt": "trtllm",
        "tllm": "trtllm",
        "trtllm": "trtllm",
        "tensorrt-llm": "trtllm",
        "tensorrtllm": "trtllm",
    }
    return aliases.get(lowered, "auto")


def framework_display_name(value: object) -> str:
    return FRAMEWORK_LABELS.get(canonicalize_framework(value), str(value))


@lru_cache(maxsize=65536)
def _normalize_repo_relative_path_cached(text: str) -> str:
    text = text.replace("\\", "/")
    lowered = text.lower()
    for marker, normalized_marker in (
        ("python/sglang/", "python/sglang/"),
        ("sgl_kernel/", "sgl_kernel/"),
        ("vllm/", "vllm/"),
        ("tensorrt_llm/", "tensorrt_llm/"),
        ("tensorrt-llm/", "tensorrt_llm/"),
    ):
        idx = lowered.find(marker)
        if idx != -1:
            suffix = text[idx + len(marker) :].lstrip("/")
            return f"{normalized_marker}{suffix}".lstrip("/")
    idx = lowered.find("sglang/")
    if idx != -1:
        return ("python/" + text[idx:]).lstrip("/")
    return text.lstrip("/")


def normalize_repo_relative_path(path: object) -> str:
    return _normalize_repo_relative_path_cached(normalize_text(path))


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


def try_get_json(url: str, timeout: float = 60.0) -> Optional[object]:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            raw = response.read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _flatten_chat_text_parts(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            parts.extend(_flatten_chat_text_parts(item))
        return parts
    if isinstance(value, dict):
        parts: List[str] = []
        text_keys = (
            "text",
            "content",
            "reasoning_content",
            "reasoning",
            "output_text",
        )
        if any(key in value for key in text_keys):
            for key in text_keys:
                parts.extend(_flatten_chat_text_parts(value.get(key)))
            if parts:
                return parts
        item_type = normalize_text(value.get("type")).lower()
        if item_type in {"text", "output_text", "input_text"}:
            for key in ("text", "content", "value"):
                parts.extend(_flatten_chat_text_parts(value.get(key)))
        elif item_type in {"reasoning", "thinking"}:
            for key in ("text", "content", "reasoning_content", "reasoning"):
                parts.extend(_flatten_chat_text_parts(value.get(key)))
        return parts
    return []


def flatten_chat_text(value: object) -> str:
    return "\n".join(_flatten_chat_text_parts(value)).strip()


def extract_openai_chat_text(body: object) -> Tuple[str, str]:
    if not isinstance(body, dict):
        return "", "invalid_body"

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        fallback = flatten_chat_text(body.get("output_text"))
        if fallback:
            return fallback, "body.output_text"
        return "", "missing_choices"

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return "", "invalid_choice"

    message = first_choice.get("message")
    if isinstance(message, dict):
        for key in ("content", "reasoning_content", "reasoning"):
            text = flatten_chat_text(message.get(key))
            if text:
                return text, f"message.{key}"

    for key in ("text", "content", "reasoning_content", "reasoning"):
        text = flatten_chat_text(first_choice.get(key))
        if text:
            return text, f"choice.{key}"

    delta = first_choice.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "reasoning_content", "reasoning"):
            text = flatten_chat_text(delta.get(key))
            if text:
                return text, f"delta.{key}"

    fallback = flatten_chat_text(body.get("output_text"))
    if fallback:
        return fallback, "body.output_text"
    return "", "empty"


def detect_framework_from_text(text: object) -> Optional[str]:
    lowered = normalize_text(text).lower()
    if not lowered:
        return None
    if any(
        token in lowered
        for token in (
            "tensorrt_llm",
            "tensorrt-llm",
            "trtllm",
            "pyexecutor",
        )
    ):
        return "trtllm"
    if "vllm" in lowered:
        return "vllm"
    if any(token in lowered for token in ("python/sglang/", "sgl_kernel/", "sglang/")):
        return "sglang"
    return None


def detect_framework_from_server_args(server_args: Optional[dict]) -> Optional[str]:
    if not isinstance(server_args, dict) or not server_args:
        return None
    lowered_keys = {normalize_text(key).lower() for key in server_args}
    if lowered_keys & {
        "attention_backend",
        "sampling_backend",
        "disable_cuda_graph",
        "disable_piecewise_cuda_graph",
        "chunked_prefill_size",
        "schedule_policy",
    }:
        return "sglang"
    return detect_framework_from_text(json.dumps(server_args, sort_keys=True))


def detect_framework_from_trace(trace: object) -> Optional[str]:
    text_samples: List[str] = []
    for event in extract_trace_events(trace)[:256]:
        text_samples.extend(
            [
                str(event.get("name", "")),
                str(event.get("cat", "")),
                str(event.get("pid", "")),
            ]
        )
        trace_args = event.get("args")
        if isinstance(trace_args, dict):
            for key, value in list(trace_args.items())[:8]:
                text_samples.append(str(key))
                if isinstance(value, str):
                    text_samples.append(value)
    return detect_framework_from_text(" ".join(text_samples))


def detect_framework_from_path(path: Path) -> Optional[str]:
    hint = detect_framework_from_text(str(path))
    if hint:
        return hint
    server_args = load_server_args(path)
    hint = detect_framework_from_server_args(server_args)
    if hint:
        return hint
    if path.is_file():
        try:
            return detect_framework_from_trace(load_trace_json(path))
        except Exception:
            return None
    trace_files = discover_trace_files(path, recursive=True, limit=3)
    for trace_file in trace_files:
        try:
            hint = detect_framework_from_trace(load_trace_json(trace_file))
        except Exception:
            hint = None
        if hint:
            return hint
    return None


def detect_framework_from_url(
    url: str, output_dir: Optional[str] = None
) -> Optional[str]:
    hint = detect_framework_from_text(output_dir or "")
    if hint:
        return hint
    server_info = try_get_json(url.rstrip("/") + "/server_info")
    if isinstance(server_info, dict) and (
        "internal_states" in server_info
        or "tokenizer_path" in server_info
        or "prefill" in server_info
        or "decode" in server_info
    ):
        return "sglang"
    models = try_get_json(url.rstrip("/") + "/v1/models")
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        return "vllm"
    return None


def resolve_framework(
    requested: object,
    *,
    input_path: Optional[Path] = None,
    url: Optional[str] = None,
    server_args: Optional[dict] = None,
) -> str:
    explicit = canonicalize_framework(requested)
    if explicit != "auto":
        return explicit
    for hint in (
        detect_framework_from_server_args(server_args),
        detect_framework_from_path(input_path) if input_path else None,
        (
            detect_framework_from_url(url, str(input_path) if input_path else None)
            if url
            else None
        ),
    ):
        if hint:
            return hint
    return "sglang"


def parse_stage(path: Path) -> str:
    name = path.name.lower()
    if "-extend" in name or "-prefill" in name:
        return "extend"
    if "-decode" in name:
        return "decode"
    return "all"


def parse_tp_rank(path: Path) -> Optional[int]:
    for pattern in (
        r"(?:^|[_-])tp(\d+)(?:[_.-]|$)",
        r"TP-(\d+)",
        r"(?:^|[_-])rank(\d+)(?:[_.-]|$)",
        r"(?:^|[_-])worker(\d+)(?:[_.-]|$)",
    ):
        match = re.search(pattern, path.name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def file_looks_like_trace(path: Path) -> bool:
    name = path.name.lower()
    if name in TRACE_FILE_IGNORE_NAMES:
        return False
    if path.is_dir():
        return False
    if any(name.endswith(suffix) for suffix in (".trace.json", ".trace.json.gz")):
        return True
    if ".pt.trace.json" in name:
        return True
    if not any(name.endswith(suffix) for suffix in (".json", ".json.gz")):
        return False
    try:
        trace = load_trace_json(path)
    except Exception:
        return False
    if isinstance(trace, dict):
        return isinstance(trace.get("traceEvents"), list)
    if isinstance(trace, list):
        return bool(trace) and all(isinstance(item, dict) for item in trace[:8])
    return False


def discover_trace_files(
    path: Path,
    *,
    recursive: bool,
    limit: Optional[int] = None,
) -> List[Path]:
    if path.is_file():
        return [path] if file_looks_like_trace(path) else []

    candidates: List[Path] = []
    seen: set[Path] = set()
    for pattern in TRACE_FILE_PATTERNS:
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        for candidate in iterator:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    candidates = [
        candidate
        for candidate in candidates
        if candidate.exists() and file_looks_like_trace(candidate)
    ]
    candidates.sort(key=lambda item: item.stat().st_mtime)
    if limit is not None and limit >= 0:
        return candidates[-limit:] if limit else []
    return candidates


def newest_trace_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent
    direct = discover_trace_files(path, recursive=False)
    if direct:
        return path
    traces = discover_trace_files(path, recursive=True)
    trace_dirs = list({trace.parent for trace in traces})
    if not trace_dirs:
        raise FileNotFoundError(f"No trace files found under {path}")
    trace_dirs.sort(
        key=lambda item: max(
            trace.stat().st_mtime for trace in traces if trace.parent == item
        )
    )
    return trace_dirs[-1]


def discover_trace_targets(
    path: Path, all_traces: bool
) -> Tuple[List[Path], Optional[dict]]:
    if path.is_file():
        return [path], load_server_args(path)

    trace_dir = newest_trace_dir(path)
    traces = discover_trace_files(trace_dir, recursive=False)
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


def post_json(
    url: str, payload: Optional[dict] = None, timeout: float = 60.0
) -> Optional[dict]:
    req = request.Request(
        url=url,
        data=(None if payload is None else json.dumps(payload).encode("utf-8")),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else None


def send_probe_request(
    url: str,
    prompt: str,
    max_new_tokens: int,
    sampling_seed: int,
    framework: str,
    model: Optional[str] = None,
) -> None:
    framework = canonicalize_framework(framework)
    if framework == "sglang":
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
        return

    resolved_model = model or discover_openai_model(url)
    chat_payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    try:
        post_json(url.rstrip("/") + "/v1/chat/completions", chat_payload, timeout=300.0)
        return
    except Exception:
        completion_payload = {
            "model": resolved_model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        post_json(
            url.rstrip("/") + "/v1/completions",
            completion_payload,
            timeout=300.0,
        )


def discover_openai_model(url: str) -> str:
    payload = try_get_json(url.rstrip("/") + "/v1/models", timeout=60.0)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Could not read {url.rstrip('/')}/v1/models")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No models returned by {url.rstrip('/')}/v1/models")
    first = data[0]
    if isinstance(first, dict) and first.get("id"):
        return str(first["id"])
    raise RuntimeError(f"Malformed /v1/models payload from {url.rstrip('/')}")


def ensure_remote_profiler_output_path(
    output_dir: Optional[str], framework: str
) -> Path:
    if not output_dir:
        raise ValueError(
            f"{framework_display_name(framework)} live capture requires --output-dir "
            "to point at the server-side torch profiler trace path that is visible "
            "from this machine."
        )
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.suffix in {".json", ".gz"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def wait_for_profiler_artifact(path: Path, timeout_s: float = 60.0) -> Path:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and file_looks_like_trace(path):
            return path
        if path.exists():
            trace_files = discover_trace_files(path, recursive=True)
            if trace_files:
                return newest_trace_dir(path)
            if path.is_dir():
                child_dirs = [item for item in path.iterdir() if item.is_dir()]
                if child_dirs:
                    child_dirs.sort(key=lambda item: item.stat().st_mtime)
                    newest_child = child_dirs[-1]
                    child_traces = discover_trace_files(newest_child, recursive=True)
                    if child_traces:
                        return newest_child
        time.sleep(0.5)
    return path


def start_remote_profiler(url: str, framework: str) -> None:
    try:
        post_json(url.rstrip("/") + "/start_profile", timeout=60.0)
    except Exception as exc:
        if framework == "vllm":
            raise RuntimeError(
                "vLLM live torch profiling requires the server to be launched with "
                '--profiler-config \'{"profiler":"torch","torch_profiler_dir":"..."}\' '
                "and to expose POST /start_profile."
            ) from exc
        if framework == "trtllm":
            raise RuntimeError(
                "TensorRT-LLM live torch profiling requires "
                "a server build that exposes POST /start_profile plus the env vars "
                "TLLM_PROFILE_START_STOP=1 and TLLM_TORCH_PROFILE_TRACE=/shared/path."
            ) from exc
        raise


def stop_remote_profiler(url: str, framework: str) -> None:
    try:
        post_json(url.rstrip("/") + "/stop_profile", timeout=300.0)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to stop {framework_display_name(framework)} profiler via "
            f"{url.rstrip('/')}/stop_profile"
        ) from exc


def run_remote_profiler(
    url: str,
    output_dir: Optional[str],
    framework: str,
    probe_requests: int,
    probe_prompt: str,
    probe_max_new_tokens: Optional[int],
    probe_delay: float,
    num_steps: int,
) -> Path:
    framework = canonicalize_framework(framework)
    output_path = ensure_remote_profiler_output_path(output_dir, framework)
    start_remote_profiler(url, framework)
    stop_error: Optional[BaseException] = None
    try:
        if probe_requests > 0:
            # Some profiler endpoints need a brief setup window after
            # POST /start_profile. A very short delay can send probes too early
            # and miss the profiling window entirely.
            time.sleep(max(5.0, probe_delay))
            effective_max_new_tokens = probe_max_new_tokens or max(64, num_steps * 8)
            model = (
                discover_openai_model(url) if framework in {"vllm", "trtllm"} else None
            )
            for request_idx in range(probe_requests):
                send_probe_request(
                    url=url,
                    prompt=probe_prompt,
                    max_new_tokens=effective_max_new_tokens,
                    sampling_seed=request_idx,
                    framework=framework,
                    model=model,
                )
    finally:
        try:
            stop_remote_profiler(url, framework)
        except BaseException as exc:  # pragma: no cover - preserve original failure
            stop_error = exc
    if stop_error is not None:
        raise stop_error
    return wait_for_profiler_artifact(output_path)


def run_sglang_profiler(
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
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / str(time.time())
    output_path.mkdir(parents=True, exist_ok=True)

    server_args = try_get_json(url.rstrip("/") + "/server_info", timeout=60.0)
    if server_args is not None:
        with open(output_path / "server_args.json", "w", encoding="utf-8") as handle:
            json.dump(server_args, handle)

    payload = {
        "output_dir": str(output_path),
        "num_steps": str(num_steps),
        "activities": ["CPU", "GPU"],
        "profile_by_stage": profile_by_stage,
        "merge_profiles": merge_profiles,
        "profile_prefix": profile_prefix,
    }
    if start_step is not None:
        payload["start_step"] = str(start_step)

    req = request.Request(
        url.rstrip("/") + "/start_profile",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=300.0):
        pass

    if probe_requests > 0:
        time.sleep(max(0.0, probe_delay))
        effective_max_new_tokens = probe_max_new_tokens or max(64, num_steps * 8)
        for request_idx in range(probe_requests):
            send_probe_request(
                url=url,
                prompt=probe_prompt,
                max_new_tokens=effective_max_new_tokens,
                sampling_seed=request_idx,
                framework="sglang",
            )

    return wait_for_profiler_artifact(output_path, timeout_s=180.0)


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
    framework: str = "auto",
    framework_hint_path: Optional[str] = None,
) -> Path:
    resolved_framework = resolve_framework(
        framework,
        url=url,
        input_path=(
            Path(framework_hint_path).expanduser().resolve()
            if framework_hint_path
            else None
        ),
    )
    if resolved_framework == "sglang":
        return run_sglang_profiler(
            url=url,
            output_dir=output_dir,
            num_steps=num_steps,
            profile_by_stage=profile_by_stage,
            merge_profiles=merge_profiles,
            profile_prefix=profile_prefix,
            probe_requests=probe_requests,
            probe_prompt=probe_prompt,
            probe_max_new_tokens=probe_max_new_tokens,
            probe_delay=probe_delay,
            start_step=start_step,
        )
    if start_step is not None:
        raise ValueError("--start-step is only supported for SGLang live capture.")
    if profile_by_stage:
        raise ValueError(
            "--profile-by-stage is only supported for SGLang live capture. "
            "Disable it when profiling vLLM or TensorRT-LLM."
        )
    if merge_profiles:
        raise ValueError(
            "--merge-profiles is only supported for SGLang live capture. "
            "Disable it when profiling vLLM or TensorRT-LLM."
        )
    if profile_prefix:
        print(
            f"Note: {framework_display_name(resolved_framework)} ignores "
            "--profile-prefix on the HTTP profiler control path.",
            file=sys.stderr,
        )
    return run_remote_profiler(
        url=url,
        output_dir=output_dir,
        framework=resolved_framework,
        probe_requests=probe_requests,
        probe_prompt=probe_prompt,
        probe_max_new_tokens=probe_max_new_tokens,
        probe_delay=probe_delay,
        num_steps=num_steps,
    )


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
