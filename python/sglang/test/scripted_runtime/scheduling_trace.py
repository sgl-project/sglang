"""Per-step scheduling traces of scripted runs, and a diff of two trace dirs.

With SGLANG_TEST_SCRIPTED_RUNTIME_TRACE_DIR set, each executed script writes one
JSONL trace there. Records hold only what two runs of one commit reproduce (rids
relabeled in first-seen order, no timestamps), so traces of two commits diff
cleanly: ``python -m sglang.test.scripted_runtime.scheduling_trace OLD NEW``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot

DIFF_CONTEXT_RECORDS: int = 3

# Bumped whenever records gain, drop or change fields, so a trace written in
# an older format shows up as a config mismatch instead of a diff.
TRACE_FORMAT: int = 3

# Server args that decide scheduling, recorded in each trace's header.
HEADER_SERVER_ARGS = (
    "model_path",
    "disable_overlap_schedule",
    "enable_mixed_chunk",
    "chunked_prefill_size",
    "page_size",
    "max_running_requests",
    "schedule_policy",
)

# Test-only env vars that change scheduling, recorded in each trace's header.
HEADER_ENVS = (
    "SGLANG_TEST_RETRACT",
    "SGLANG_TEST_RETRACT_INTERVAL",
    "SGLANG_TEST_RETRACT_NO_PREFILL_BS",
)


class SchedulingTraceWriter:
    def __init__(self, path: Path, header: Dict[str, Any]) -> None:
        # Line-buffered so written records survive the server dying mid-script.
        self._file = path.open("w", buffering=1)
        self._step = 0
        self._rid_labels: Dict[str, str] = {}
        self._retraction_counts: Dict[str, int] = {}
        self._readers: Dict[str, Any] = {}
        self._write({"kind": "header", **header})

    @classmethod
    def maybe_open(
        cls, scheduler: Scheduler, fn_path: str, args: Sequence[Any]
    ) -> Optional[SchedulingTraceWriter]:
        trace_dir = envs.SGLANG_TEST_SCRIPTED_RUNTIME_TRACE_DIR.get()
        if not trace_dir:
            return None
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        extra_kwargs = envs.SGLANG_TEST_SCRIPTED_RUNTIME_EXTRA_ENGINE_KWARGS.get()
        header = {
            "format": TRACE_FORMAT,
            "extra_engine_kwargs": json.loads(extra_kwargs) if extra_kwargs else {},
            "server_args": {
                name: getattr(scheduler.parts.server_args, name)
                for name in HEADER_SERVER_ARGS
            },
            "envs": {name: getattr(envs, name).get() for name in HEADER_ENVS},
        }
        return cls(Path(trace_dir) / trace_file_name(fn_path, args), header)

    def close(self) -> None:
        self._file.close()

    def observe_readers(self, scheduler: Scheduler) -> None:
        """Snapshot reader values at the top of an iteration for the next record.

        Not taken in record_schedule: mid-iteration, the requests a step
        admits are in the waiting queue until its launch, a state no reader sees.
        """
        load = scheduler.hub.load_inquirer.get_loads()
        self._readers = {
            "idle": scheduler.hub.request_registry.is_fully_idle(),
            "idle_ignore_waiting": scheduler.hub.request_registry.is_fully_idle(
                ignore_waiting=True
            ),
            # A set, so sorted; each req was labeled by an earlier schedule record.
            "inflight": sorted(
                self._labels(scheduler.hub.request_registry.collect_inflight_reqs())
            ),
            "load": {
                "running": load.num_running_reqs,
                "waiting": load.num_waiting_reqs,
            },
        }
        if scheduler.parts.flags._engine_paused:
            self._write({"kind": "paused", "step": self._step, **self._readers})

    def record_schedule(
        self, scheduler: Scheduler, batch: Optional[ScheduleBatch], slot: ScheduleSlot
    ) -> None:
        self._step += 1
        batch_reqs = batch.reqs if batch is not None else []
        # The launch moves the requests the step prefills out of the waiting
        # queue, and its chunked request into the chunked holder.
        prefilled = batch.prefill_reqs() if batch is not None else []
        admitted = set(prefilled)
        waiting = [r for r in scheduler.parts.lists.waiting_queue if r not in admitted]
        chunked = batch.chunked_req if prefilled else scheduler.parts.lists.chunked_req
        record: Dict[str, Any] = {
            "kind": "schedule",
            "step": self._step,
            "mode": _mode_name(batch),
            "rids": self._labels(batch_reqs),
        }
        if batch is not None and batch.forward_mode.is_extend():
            record["extend_lens"] = list(batch.extend_lens)
            record["prefix_lens"] = list(batch.prefix_lens)
        record.update(
            chunked=self._label(chunked.rid) if chunked is not None else None,
            running=self._labels(scheduler.parts.lists.decoding),
            waiting=self._labels(waiting),
            retracted=self._labels(
                self._newly_retracted(
                    [
                        *batch_reqs,
                        *scheduler.parts.lists.decoding,
                        *waiting,
                        *([chunked] if chunked is not None else []),
                    ]
                )
            ),
            kv_available=scheduler.parts.kv.token_to_kv_pool_allocator.available_size(),
            kv_evictable=scheduler.parts.kv.tree_cache.evictable_size(),
            new_token_ratio=scheduler.parts.new_token_ratio_tracker.current,
            **self._readers,
        )
        self._write(record)

    def record_process_result(self, batch: ScheduleBatch) -> None:
        self._write(
            {
                "kind": "result",
                "step": self._step,
                "mode": _mode_name(batch),
                "rids": self._labels(batch.reqs),
                "output_lens": [len(req.output_ids) for req in batch.reqs],
                "finished": [
                    type(req.finished_reason).__name__ if req.finished() else None
                    for req in batch.reqs
                ],
            }
        )

    def _newly_retracted(self, reqs: Iterable) -> List:
        # A retracted req can be re-admitted by the plan that first sees it (on
        # resume after pause_generation), so scan every req, not only waiting.
        retracted = []
        for req in reqs:
            if req.retraction_count > self._retraction_counts.get(req.rid, 0):
                self._retraction_counts[req.rid] = req.retraction_count
                retracted.append(req)
        return retracted

    def _label(self, rid: str) -> str:
        return self._rid_labels.setdefault(rid, f"r{len(self._rid_labels)}")

    def _labels(self, reqs: Iterable) -> List[str]:
        return [self._label(req.rid) for req in reqs]

    def _write(self, record: Dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")


def trace_file_name(fn_path: str, args: Sequence[Any]) -> str:
    name = fn_path.replace(":", ".") + "".join(f"-{arg}" for arg in args)
    return re.sub(r"[^\w.-]+", "_", name) + ".jsonl"


def _mode_name(batch: Optional[ScheduleBatch]) -> Optional[str]:
    return batch.forward_mode.name.lower() if batch is not None else None


def load_trace(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def first_divergence(
    old: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    ignore: Iterable[str] = (),
    as_set: Iterable[str] = (),
) -> Optional[int]:
    ignore, as_set = set(ignore), set(as_set)
    for i, (a, b) in enumerate(zip(old, new)):
        if _comparable(a, ignore, as_set) != _comparable(b, ignore, as_set):
            return i
    if len(old) != len(new):
        return min(len(old), len(new))
    return None


def _header(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if records and records[0].get("kind") == "header":
        return records[0]
    return None


def _comparable(record: Dict[str, Any], ignore: set, as_set: set) -> Dict[str, Any]:
    """The record without the ignored fields, the as_set list fields sorted."""
    return {
        k: sorted(v, key=json.dumps) if k in as_set and isinstance(v, list) else v
        for k, v in record.items()
        if k not in ignore
    }


def _differing_fields(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def _print_all_differences(old, new, ignore: set, as_set: set, out) -> None:
    """Each differing record's fields with both values, then counts per field."""
    field_counts: Counter = Counter()
    for i, (a, b) in enumerate(zip(old, new)):
        a, b = _comparable(a, ignore, as_set), _comparable(b, ignore, as_set)
        fields = _differing_fields(a, b)
        if not fields:
            continue
        field_counts.update(fields)
        print(f"    [{i}] fields: {', '.join(fields)}", file=out)
        for k in fields:
            print(f"      - old {k}: {json.dumps(a.get(k))}", file=out)
            print(f"      + new {k}: {json.dumps(b.get(k))}", file=out)
    if len(old) != len(new):
        print(f"    lengths: old {len(old)}, new {len(new)} records", file=out)
    counts = ", ".join(f"{k}={n}" for k, n in sorted(field_counts.items()))
    print(f"    counts: {counts or '-'}", file=out)


def diff_trace_dirs(
    old_dir: Path,
    new_dir: Path,
    *,
    ignore: Iterable[str] = (),
    as_set: Iterable[str] = (),
    all_records: bool = False,
    out=sys.stdout,
) -> int:
    ignore, as_set = set(ignore), set(as_set)
    old_names = {p.name for p in old_dir.glob("*.jsonl")}
    new_names = {p.name for p in new_dir.glob("*.jsonl")}
    num_differ = 0
    for name in sorted(old_names | new_names):
        if name not in new_names or name not in old_names:
            side = "old" if name in old_names else "new"
            print(f"ONLY IN {side.upper()}  {name}", file=out)
            num_differ += 1
            continue
        old = load_trace(old_dir / name)
        new = load_trace(new_dir / name)
        old_header, new_header = _header(old), _header(new)
        if old_header is None or old_header != new_header:
            num_differ += 1
            print(f"CONFIG       {name} headers differ, not compared", file=out)
            print(f"  - old {json.dumps(old_header)}", file=out)
            print(f"  + new {json.dumps(new_header)}", file=out)
            continue
        idx = first_divergence(old, new, ignore, as_set)
        if idx is None:
            print(f"IDENTICAL    {name} ({len(old)} records)", file=out)
            continue
        num_differ += 1
        print(f"DIFFERS      {name} at record {idx}", file=out)
        if all_records:
            _print_all_differences(old, new, ignore, as_set, out)
            continue
        for i in range(max(0, idx - DIFF_CONTEXT_RECORDS), idx):
            print(f"    [{i}]     {json.dumps(old[i])}", file=out)
        a = old[idx] if idx < len(old) else None
        b = new[idx] if idx < len(new) else None
        for sign, side, record in (("-", "old", a), ("+", "new", b)):
            text = json.dumps(record) if record is not None else "<end of trace>"
            print(f"  {sign} {side} [{idx}] {text}", file=out)
        if a is not None and b is not None:
            fields = _differing_fields(
                _comparable(a, ignore, as_set), _comparable(b, ignore, as_set)
            )
            print(f"    fields: {', '.join(fields)}", file=out)
        else:
            # One trace ended early: what the other one goes on with.
            record = a if a is not None else b
            print(f"    lengths: old {len(old)}, new {len(new)} records", file=out)
            print(f"    fields: {', '.join(sorted(record.keys() - ignore))}", file=out)
    print(
        f"{len(old_names | new_names) - num_differ} identical, {num_differ} differ",
        file=out,
    )
    return num_differ


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two dirs of scripted-runtime scheduling traces."
    )
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="FIELD",
        help="record field to leave out of the comparison (repeatable)",
    )
    parser.add_argument(
        "--as-set",
        action="append",
        default=[],
        metavar="FIELD",
        help="list field to compare as a set, ignoring its order (repeatable)",
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="print every differing record of a trace, with counts per field",
    )
    args = parser.parse_args(argv)
    for trace_dir in (args.old_dir, args.new_dir):
        if not any(trace_dir.glob("*.jsonl")):
            print(f"error: no *.jsonl traces in {trace_dir}", file=sys.stderr)
            return 2
    num_differ = diff_trace_dirs(
        args.old_dir,
        args.new_dir,
        ignore=args.ignore,
        as_set=args.as_set,
        all_records=args.all_records,
    )
    return 1 if num_differ else 0


if __name__ == "__main__":
    sys.exit(main())
