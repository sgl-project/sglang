# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trace captured graph operands against PyTorch allocation lifetimes.

``sitecustomize`` injects the tracer into serving workers without production
hooks. CUDA tracing starts lazily at the first capture, on one device per
process. Graphs use ``keep_graph=True`` and weak references: the tracer never
keeps a graph or its operand buffers alive. Breakable segments share a forward
window so eager bridges remain distinguishable from graph-pool temporaries.

Inspect graph operands once, then check their lifetimes before the first replay,
periodically, and on a nonce-bearing flush request or SIGUSR1. Nested replays
defer checks until capture ends: synchronization, GC and allocator snapshots
are unsafe inside capture. Non-clean findings remain in the status history.

SGLang wrappers label supported paths and observe their replays, including direct
dedup launches. Reports identify the run, worker, physical GPU and imported
source; a final flush echoes the harness nonce. Failed or incomplete inspections
raise unless fail-fast is disabled.

Outside scope: non-PyTorch allocations, device-side pointer tables, stale-but-live
metadata, unexercised traffic, and raw graphs mutated after their first inspection.
Graphs without a path wrapper are inspected under generic labels.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import gc
import json
import os
import signal
import socket
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TRACER_VERSION = 3
_SEVERITY = {
    "clean": 0,
    "no-captures": 1,
    "unchecked": 1,
    "incomplete": 2,
    "failed": 3,
    "dangling": 4,
}
ENV_REPORT_DIR = "SGLANG_GRAPH_LIFETIME_REPORT_DIR"
ENV_RUN_ID = "SGLANG_GRAPH_LIFETIME_RUN_ID"
ENV_CHECK_EVERY = "SGLANG_GRAPH_LIFETIME_CHECK_EVERY"
ENV_FAIL_FAST = "SGLANG_GRAPH_LIFETIME_FAIL_FAST"
_DEFAULT_CHECK_EVERY = 10


@dataclass
class _Record:
    label: str
    graph_ref: Any  # weakref.ref to the CUDAGraph
    forward_begin: Any  # marker tensors: their alloc events bound the windows
    begin: Any
    end: Any = None
    raw_graph: Optional[int] = None
    pools: Optional[frozenset] = None
    nodes: int = 0
    operands: Dict[int, List[str]] = field(default_factory=dict)
    scanned: bool = False
    failure: Optional[str] = None
    inspected: bool = False


def _current_device_index() -> int:
    import torch

    return torch.cuda.current_device()


_PARALLEL_FIELDS = (
    "tp_rank",
    "pp_rank",
    "dp_rank",
    "attn_dp_rank",
    "tp_size",
    "pp_size",
    "dp_size",
    "attn_dp_size",
    "nnodes",
    "node_rank",
)


def _parallel_identity() -> Optional[dict]:
    """The scheduler ranks and widths SGLang published in this process, or
    None in a process that is no scheduler worker (nothing published)."""
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
    except Exception:  # noqa: BLE001 - not an SGLang process
        return None
    identity: Dict[str, Any] = {}
    for name in _PARALLEL_FIELDS:
        try:
            value = getattr(parallel, name)
        except Exception:  # noqa: BLE001 - not written in this process
            value = None
        identity[name] = (
            value if value is None or isinstance(value, int) else str(value)
        )
    if all(value is None for value in identity.values()):
        return None
    return identity


def _device_identity(device) -> dict:
    """The physical device behind a logical name: distinct visibility
    mappings give different GPUs the same ``cuda:0``."""
    import torch

    try:
        props = torch.cuda.get_device_properties(device)
    except Exception as e:  # noqa: BLE001
        return {"device_uuid": None, "pci_bus_id": None, "error": repr(e)}
    uuid = getattr(props, "uuid", None)
    return {
        "device_uuid": None if uuid is None else str(uuid),
        "pci_bus_id": getattr(props, "pci_bus_id", None),
    }


_source: Optional[dict] = None


def _source_identity() -> dict:
    """The sglang and torch this process imported, with the git commit and
    tree of the sglang source directory when it is a checkout: the identity
    of the code under test as the worker itself sees it, which the harness
    compares with its own working directory."""
    global _source
    if _source is not None:
        return _source
    identity: Dict[str, Any] = {}
    try:
        import sglang

        identity["sglang_file"] = sglang.__file__
        try:
            from sglang.version import __version__

            identity["sglang_version"] = __version__
        except Exception:  # noqa: BLE001
            identity["sglang_version"] = None
        import subprocess

        root = os.path.dirname(os.path.abspath(sglang.__file__))
        probe = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        identity["git"] = probe.stdout.split() if probe.returncode == 0 else None
        if identity["git"]:
            # Uncommitted changes make the commit a weaker identity: the
            # harness reports them, since the files on disk are what ran.
            status = subprocess.run(
                ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            identity["dirty"] = (
                bool(status.stdout.strip()) if status.returncode == 0 else None
            )
    except Exception as e:  # noqa: BLE001 - identity is best effort, never fatal
        identity["error"] = repr(e)
    try:
        import torch

        identity["torch"] = torch.__version__
        identity["torch_file"] = torch.__file__
    except Exception:  # noqa: BLE001
        pass
    _source = identity
    return identity


class LifetimeTracer:
    def __init__(self) -> None:
        self.installed_at = time.time()
        self.report_dir = os.environ.get(ENV_REPORT_DIR) or None
        self.run_id = os.environ.get(ENV_RUN_ID) or None
        self.flush_nonce: Optional[str] = None  # from the last flush request file
        self.check_every = max(
            1, int(os.environ.get(ENV_CHECK_EVERY, _DEFAULT_CHECK_EVERY))
        )
        self.fail_fast = os.environ.get(ENV_FAIL_FAST, "1") != "0"
        self.device = None
        self.records: Dict[int, _Record] = {}
        self.retired: int = 0
        self.captures = 0
        self.replays = 0
        self.replays_since_check = 0
        self.checks = 0
        self.deferred_checks = 0
        self.deferred = False
        self.flush_requested = False
        self.label_captures: Dict[str, int] = {}
        self.label_replays: Dict[str, int] = {}
        self._label_stack: List[str] = []
        self._forward_stack: List[Any] = []
        self._pending_begin: Dict[int, tuple] = {}
        self.status = "no-captures"  # the latest check; the report shows the worst
        self.status_history: List[dict] = []  # every non-clean check, never erased
        self.last_report: Optional[dict] = None
        self._last_findings = None  # kept so an exit report still shows them
        self.errors: List[str] = []
        self._lock = threading.RLock()
        self._recording = False
        self._traces: List[dict] = []
        self.hostname = socket.gethostname()

    # -- labels and forward windows (set by the SGLang wrappers) ---------------

    @contextlib.contextmanager
    def labelled(self, label: str):
        self._label_stack.append(label)
        try:
            yield
        finally:
            self._label_stack.pop()

    @contextlib.contextmanager
    def forward(self):
        """One captured forward whose segments share a forward window."""
        marker = self._mark()
        self._forward_stack.append(marker)
        try:
            yield
        finally:
            self._forward_stack.pop()

    def count_label_replay(self, label: str) -> None:
        self.label_replays[label] = self.label_replays.get(label, 0) + 1

    # -- CUDAGraph hooks ---------------------------------------------------------

    def _mark(self):
        import torch

        from sglang.test.cuda_graph_lifetime import classifier

        if self.device is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=classifier._MAX_TRACE_ENTRIES,
                device=self.device,
            )
            self._recording = True
        return torch.empty(
            classifier._MARKER_BYTES, dtype=torch.uint8, device=self.device
        )

    def on_init(self, graph) -> None:
        key = id(graph)
        weakref.finalize(graph, self._retire, key)

    def on_capture_begin(self, graph) -> None:
        with self._lock:
            label = (
                self._label_stack[-1] if self._label_stack else f"graph#{self.captures}"
            )
            forward_begin = self._forward_stack[-1] if self._forward_stack else None
            begin = self._mark()
            current = _current_device_index()
            if current != self.device.index:
                # One capture device per process: the allocator trace, the
                # markers and every lifetime belong to the first device, so a
                # capture elsewhere could only be classified against the wrong
                # trace. Fail explicitly instead.
                message = (
                    f"capture of {label!r} on cuda:{current} but the tracer is bound "
                    f"to {self.device}: one capture device per process is supported"
                )
                self.errors.append(message)
                self._record_failure("device conflict", "failed", message)
                self.write_report("device conflict")
                raise RuntimeError(f"CUDA graph lifetime tracer: {message}")
            if forward_begin is None:
                forward_begin = begin
            self._pending_begin[id(graph)] = (label, forward_begin, begin)

    def on_capture_end(self, graph) -> None:
        with self._lock:
            pending = self._pending_begin.pop(id(graph), None)
            if pending is None:
                return
            label, forward_begin, begin = pending
            record = _Record(label, weakref.ref(graph), forward_begin, begin)
            record.end = self._mark()
            try:
                record.raw_graph = int(graph.raw_cuda_graph())
                record.pools = frozenset([tuple(graph.pool())])
            except Exception as e:  # keep_graph unsupported or pool unreadable
                record.failure = f"raw graph or pool unavailable: {e!r}"
            if self._forward_stack:
                # Segments of one forward share the forward's pool set.
                for other in self.records.values():
                    if (
                        other.forward_begin is forward_begin
                        and other.pools
                        and record.pools
                    ):
                        record.pools = record.pools | other.pools
            self.records[id(graph)] = record
            self.captures += 1
            self.label_captures[label] = self.label_captures.get(label, 0) + 1
            if self.status == "no-captures":
                self.status = "unchecked"
                self.write_report("first capture")
            if self.deferred:
                self._maybe_check("deferred check after an outer capture ended")

    def on_capture_abandoned(self, graph) -> None:
        with self._lock:
            self._pending_begin.pop(id(graph), None)

    def flush_file(self) -> Optional[str]:
        if not self.report_dir:
            return None
        return os.path.join(self.report_dir, f"flush-{os.getpid()}.request")

    def _take_flush_nonce(self) -> None:
        """The harness writes a nonce into the request file; the report of the
        check it triggers carries it back, so a stale report cannot pass for the
        acknowledgement of this run's final flush."""
        path = self.flush_file()
        try:
            with open(path) as f:
                self.flush_nonce = f.read().strip() or None
        except OSError:
            self.flush_nonce = None

    def before_replay(self, graph=None, raw_graph: Optional[int] = None) -> None:
        import torch

        with self._lock:
            record = None
            if graph is not None:
                record = self.records.get(id(graph))
            elif raw_graph is not None:
                record = next(
                    (r for r in self.records.values() if r.raw_graph == raw_graph), None
                )
            self.replays += 1
            self.replays_since_check += 1
            if torch.cuda.is_current_stream_capturing():
                # A replay recorded into another graph's capture: checking here
                # would synchronize inside the capture, so defer to its end.
                self.deferred = True
                self.deferred_checks += 1
                return
            path = self.flush_file()
            if path is not None and os.path.exists(path):
                self.flush_requested = True
                self._take_flush_nonce()
                self._maybe_check("flush-request")
                return
            due = (
                (record is not None and not record.inspected)
                or self.replays_since_check >= self.check_every
                or self.flush_requested
                or self.deferred
            )
            if due:
                self._maybe_check("before replay")

    def _retire(self, key: int) -> None:
        with self._lock:
            if self.records.pop(key, None) is not None:
                self.retired += 1
            self._pending_begin.pop(key, None)

    # -- analysis --------------------------------------------------------------

    def _record_failure(self, reason: str, status: str, summary: str) -> None:
        self.status_history.append(
            {
                "reason": reason,
                "status": status,
                "time": time.time(),
                "summary": summary,
            }
        )

    def effective_status(self) -> str:
        """The worst status ever observed: a graph retired after a dangling
        check does not make the process clean again."""
        worst = self.status
        for entry in self.status_history:
            if _SEVERITY.get(entry["status"], 0) > _SEVERITY.get(worst, 0):
                worst = entry["status"]
        return worst

    def _maybe_check(self, reason: str) -> None:
        import torch

        if torch.cuda.is_current_stream_capturing():
            self.deferred = True
            return
        self.deferred = False
        self.flush_requested = False
        self.replays_since_check = 0
        findings = self.analyze()
        self.checks += 1
        self.status = findings.status
        self._last_findings = findings
        if findings.status != "clean":
            from sglang.test.cuda_graph_lifetime import classifier

            diagnostic = classifier.format_report(findings, self._traces)
            self._record_failure(reason, findings.status, diagnostic.splitlines()[0])
        self.write_report(reason, findings)
        path = self.flush_file()
        if path is not None and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        if findings.status != "clean" and self.fail_fast:
            raise RuntimeError(f"CUDA graph lifetime tracer ({reason}): " + diagnostic)

    def analyze(self):
        import torch

        from sglang.test.cuda_graph_lifetime import classifier

        torch.cuda.synchronize(self.device)
        gc.collect()  # capture-time garbage must reach the trace before it is read
        torch.cuda.synchronize(self.device)
        snapshot = torch.cuda.memory._snapshot(self.device)
        traces = snapshot["device_traces"][self.device.index]
        self._traces = traces
        lifetimes = classifier._Lifetimes(traces)
        findings = classifier.Findings()
        if len(traces) >= classifier._MAX_TRACE_ENTRIES - 1:
            findings.incomplete.append(
                f"allocator trace ring is full ({len(traces)} entries); "
                "older captures may be misattributed"
            )
        seen: set = set()
        unknown_pools: set = set()
        for record in list(self.records.values()):
            if record.graph_ref() is None:
                continue  # retiring
            if not record.scanned and record.failure is None:
                record.scanned = True
                try:
                    record.nodes = classifier._graph_operands(
                        record.raw_graph, record.operands
                    )
                except classifier.InspectionError as e:
                    record.failure = str(e)
            if record.failure is not None:
                findings.failed.append(f"{record.label}: {record.failure}")
                continue
            classifier.classify_record(record, lifetimes, findings, seen, unknown_pools)
            record.inspected = True
        for label, base in sorted(unknown_pools):
            findings.incomplete.append(
                f"{label}: operand lifetime at {base:#x} has no allocator pool information"
            )
        return findings

    # -- reporting -------------------------------------------------------------

    def report_dict(self, reason: str, findings=None) -> dict:
        import torch

        from sglang.test.cuda_graph_lifetime import classifier

        rank = None
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
        except Exception:  # noqa: BLE001 - identity is best effort
            rank = None
        report = {
            "tracer_version": TRACER_VERSION,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": self.hostname,
            "rank": rank,
            "parallel": _parallel_identity(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": None if self.device is None else str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device is not None
                else None
            ),
            **(
                _device_identity(self.device)
                if self.device is not None
                else {"device_uuid": None, "pci_bus_id": None}
            ),
            "source": _source_identity(),
            "torch": torch.__version__,
            "status": self.effective_status(),
            "current_status": self.status,
            "status_history": list(self.status_history),
            "reason": reason,
            "flush_nonce": self.flush_nonce if reason == "flush-request" else None,
            "time": time.time(),
            "counts": {
                "captures": self.captures,
                "live_records": len(self.records),
                "retired": self.retired,
                "replays": self.replays,
                "checks": self.checks,
                "deferred_checks": self.deferred_checks,
                "check_every": self.check_every,
            },
            "label_captures": dict(sorted(self.label_captures.items())),
            "label_replays": dict(sorted(self.label_replays.items())),
            "errors": list(self.errors),
        }
        if findings is None:
            findings = self._last_findings
        if findings is not None:
            report["findings"] = classifier.findings_dict(findings, self._traces)
        return report

    def write_report(self, reason: str, findings=None) -> None:
        self.last_report = self.report_dict(reason, findings)
        if not self.report_dir:
            return
        os.makedirs(self.report_dir, exist_ok=True)
        path = os.path.join(self.report_dir, f"worker-{os.getpid()}.json")
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.last_report, f, indent=1)
        os.replace(tmp, path)

    def write_installed_marker(self) -> None:
        if not self.report_dir:
            return
        try:
            os.makedirs(self.report_dir, exist_ok=True)
            path = os.path.join(self.report_dir, f"installed-{os.getpid()}.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "tracer_version": TRACER_VERSION,
                        "run_id": self.run_id,
                        "pid": os.getpid(),
                        "argv": sys.argv[:4],
                        "time": self.installed_at,
                    },
                    f,
                )
        except (
            OSError
        ) as e:  # the report dir is the harness's problem, not the server's
            self.errors.append(f"installed marker: {e!r}")

    def flush(self, reason: str = "flush") -> None:
        """Run a check now if outside a capture, else at the next safe point."""
        with self._lock:
            if self.device is None:
                self.write_report(reason)
                return
            self.flush_requested = True
            self._maybe_check(reason)


_tracer: Optional[LifetimeTracer] = None
_patched: set = set()


def get() -> LifetimeTracer:
    global _tracer
    if _tracer is None:
        _tracer = LifetimeTracer()
    return _tracer


def reset_for_tests() -> LifetimeTracer:
    """Drop every record and start a fresh tracer (the torch wrappers stay)."""
    global _tracer
    if _tracer is not None and _tracer._recording:
        import torch

        torch.cuda.memory._record_memory_history(enabled=None, device=_tracer.device)
    _tracer = LifetimeTracer()
    return _tracer


# -- torch.cuda.CUDAGraph wrappers --------------------------------------------


def _wrap(cls, name: str, make):
    original = getattr(cls, name)
    if getattr(original, "_lifetime_tracer_wrapped", False):
        return
    wrapper = make(original)
    functools.update_wrapper(wrapper, original)
    wrapper._lifetime_tracer_wrapped = True
    setattr(cls, name, wrapper)


def patch_torch() -> None:
    """Wrap torch.cuda.CUDAGraph in place (methods only; the class identity,
    subclasses and signatures are untouched)."""
    import torch

    if "torch" in _patched:
        return
    _patched.add("torch")
    cls = torch.cuda.CUDAGraph

    def init(original):
        def __init__(self, *args, **kwargs):
            kwargs["keep_graph"] = True
            original(self, *args, **kwargs)
            get().on_init(self)

        return __init__

    def capture_begin(original):
        def capture_begin(self, *args, **kwargs):
            get().on_capture_begin(self)
            try:
                return original(self, *args, **kwargs)
            except BaseException:
                get().on_capture_abandoned(self)
                raise

        return capture_begin

    def capture_end(original):
        def capture_end(self, *args, **kwargs):
            try:
                result = original(self, *args, **kwargs)
            except BaseException:
                get().on_capture_abandoned(self)
                raise
            get().on_capture_end(self)
            return result

        return capture_end

    def replay(original):
        def replay(self, *args, **kwargs):
            get().before_replay(graph=self)
            return original(self, *args, **kwargs)

        return replay

    def reset(original):
        def reset(self, *args, **kwargs):
            get()._retire(id(self))
            return original(self, *args, **kwargs)

        return reset

    _wrap(cls, "__init__", init)
    _wrap(cls, "capture_begin", capture_begin)
    _wrap(cls, "capture_end", capture_end)
    _wrap(cls, "replay", replay)
    _wrap(cls, "reset", reset)


# -- SGLang wrappers (applied when the modules are imported) --------------------

_SGLANG_MODULES = {
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph": "_patch_breakable_capture",
    "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin": "_patch_dedup",
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend": "_patch_full_backend",
    "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend": "_patch_breakable_backend",
    "sglang.srt.model_executor.runner.row_bucket_graph_runner": "_patch_row_bucket",
    "sglang.srt.model_executor.runner.metadata_glue_graph": "_patch_glue",
}


def _shape_label(backend, shape_key) -> str:
    runner = getattr(backend, "_cuda_graph_runner", None)
    parts = [type(runner).__name__ if runner is not None else type(backend).__name__]
    parts.append(f"size={getattr(shape_key, 'size', shape_key)}")
    for name in ("variant_label", "attention_variant"):
        value = getattr(shape_key, name, None)
        if value is not None:
            parts.append(f"{name}={value}")
    return " ".join(parts)


def _patch_breakable_capture(module) -> None:
    cls = module.BreakableCUDAGraphCapture

    def enter(original):
        def __enter__(self):
            self._lifetime_forward = get().forward()
            self._lifetime_forward.__enter__()
            try:
                return original(self)
            except BaseException:
                self._lifetime_forward.__exit__(None, None, None)
                raise

        return __enter__

    def exit_(original):
        def __exit__(self, *args):
            try:
                return original(self, *args)
            finally:
                forward = getattr(self, "_lifetime_forward", None)
                if forward is not None:
                    forward.__exit__(None, None, None)

        return __exit__

    _wrap(cls, "__enter__", enter)
    _wrap(cls, "__exit__", exit_)


def _patch_dedup(module) -> None:
    cls = module.DedupedCudaGraph

    def replay(original):
        def replay(self, *args, **kwargs):
            get().before_replay(raw_graph=int(self.raw_graph))
            return original(self, *args, **kwargs)

        return replay

    _wrap(cls, "replay", replay)


def _patch_capture_and_replay(cls) -> None:
    def capture_one(original):
        def capture_one(self, shape_key, *args, **kwargs):
            with get().labelled(_shape_label(self, shape_key)):
                return original(self, shape_key, *args, **kwargs)

        return capture_one

    def replay(original):
        def replay(self, shape_key, *args, **kwargs):
            get().count_label_replay(_shape_label(self, shape_key))
            return original(self, shape_key, *args, **kwargs)

        return replay

    _wrap(cls, "capture_one", capture_one)
    _wrap(cls, "replay", replay)


def _patch_full_backend(module) -> None:
    _patch_capture_and_replay(module.FullCudaGraphBackend)


def _patch_breakable_backend(module) -> None:
    _patch_capture_and_replay(module.BreakableCudaGraphBackend)


def _patch_row_bucket(module) -> None:
    cls = module.RowBucketGraphRunner

    def capture(original):
        def _capture(self, rows, *args, **kwargs):
            with get().labelled(f"{type(self).__name__} rows={rows}"):
                return original(self, rows, *args, **kwargs)

        return _capture

    _wrap(cls, "_capture", capture)


def _patch_glue(module) -> None:
    cls = module.MetadataGlueGraph

    def run(original):
        def run(self, *args, **kwargs):
            with get().labelled(type(self).__name__):
                return original(self, *args, **kwargs)

        return run

    _wrap(cls, "run", run)


def patch_sglang_module(name: str, module) -> None:
    patcher = _SGLANG_MODULES.get(name)
    if patcher is None or name in _patched:
        return
    _patched.add(name)
    try:
        globals()[patcher](module)
    except Exception as e:  # a changed internal: record, keep serving
        get().errors.append(f"could not wrap {name}: {e!r}")


def patch_already_imported() -> None:
    for name in _SGLANG_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            patch_sglang_module(name, module)


# -- process-level installation ---------------------------------------------------


def _on_sigusr1(signum, frame) -> None:
    try:
        get().flush("SIGUSR1")
    except Exception as e:
        get().errors.append(f"flush: {e!r}")
        get().write_report("flush failed")
        raise


def install() -> LifetimeTracer:
    """Called by sitecustomize: cheap, no CUDA, torch patched when it loads."""
    tracer = get()
    tracer.write_installed_marker()
    if "torch" in sys.modules:
        patch_torch()
    patch_already_imported()
    try:
        signal.signal(signal.SIGUSR1, _on_sigusr1)
    except (ValueError, OSError):  # not the main thread
        pass
    atexit.register(lambda: tracer.write_report("atexit") if tracer.device else None)
    return tracer
