"""Per-step scheduling traces of scripted runs, and a diff of two trace dirs.

With SGLANG_TEST_SCRIPTED_RUNTIME_TRACE_DIR set, every executed script writes
one JSONL file there: a ``schedule`` record each time the scheduler picks the
next batch and a ``result`` record each time a batch result is processed.
Records hold only what two runs of one commit reproduce: rids are relabeled
r0, r1, ... in first-seen order, and there are no timestamps or forward_ct.

Two commits are compared by running the same scripted tests on each with a
different trace dir, then:

    python -m sglang.test.scripted_runtime.scheduling_trace OLD_DIR NEW_DIR

SGLANG_TEST_SCRIPTED_RUNTIME_EXTRA_ENGINE_KWARGS (a JSON object such as
'{"disable_overlap_schedule": true}' or '{"enable_mixed_chunk": true}') runs
the same tests under another server config.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

DIFF_CONTEXT_RECORDS: int = 3


class SchedulingTraceWriter:
    def __init__(self, path: Path) -> None:
        # Line-buffered: each record is on disk once written, also when the
        # server dies mid-script.
        self._file = path.open("w", buffering=1)
        self._step = 0
        self._rid_labels: Dict[str, str] = {}
        self._retraction_counts: Dict[str, int] = {}

    @classmethod
    def maybe_open(
        cls, fn_path: str, args: Sequence[Any]
    ) -> Optional[SchedulingTraceWriter]:
        trace_dir = envs.SGLANG_TEST_SCRIPTED_RUNTIME_TRACE_DIR.get()
        if not trace_dir:
            return None
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        return cls(Path(trace_dir) / trace_file_name(fn_path, args))

    def close(self) -> None:
        self._file.close()

    def record_schedule(self, scheduler: Scheduler, plan: NextBatchPlan) -> None:
        self._step += 1
        batch = plan.batch_to_run
        batch_reqs = batch.reqs if batch is not None else []
        chunked = scheduler.chunked_req
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
            running=self._labels(plan.running_batch.reqs),
            waiting=self._labels(scheduler.waiting_queue),
            batch_is_full=plan.running_batch.batch_is_full,
            retracted=self._labels(
                self._newly_retracted(
                    [
                        *batch_reqs,
                        *plan.running_batch.reqs,
                        *scheduler.waiting_queue,
                        *([chunked] if chunked is not None else []),
                    ]
                )
            ),
            kv_available=scheduler.token_to_kv_pool_allocator.available_size(),
            kv_evictable=scheduler.tree_cache.evictable_size(),
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
        # A retracted req can be re-admitted in the same plan that first sees
        # it (e.g. on resume after pause_generation), so look at every req the
        # record sees, not only the waiting queue.
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
    old: List[Dict[str, Any]], new: List[Dict[str, Any]], ignore: Iterable[str] = ()
) -> Optional[int]:
    ignore = set(ignore)
    for i, (a, b) in enumerate(zip(old, new)):
        if _strip(a, ignore) != _strip(b, ignore):
            return i
    if len(old) != len(new):
        return min(len(old), len(new))
    return None


def _strip(record: Dict[str, Any], ignore: set) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if k not in ignore}


def diff_trace_dirs(
    old_dir: Path, new_dir: Path, *, ignore: Iterable[str] = (), out=sys.stdout
) -> int:
    """Print the per-script comparison; return the number of scripts that differ."""
    ignore = set(ignore)
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
        idx = first_divergence(old, new, ignore)
        if idx is None:
            print(f"IDENTICAL    {name} ({len(old)} records)", file=out)
            continue
        num_differ += 1
        print(f"DIFFERS      {name} at record {idx}", file=out)
        for i in range(max(0, idx - DIFF_CONTEXT_RECORDS), idx):
            print(f"    [{i}]     {json.dumps(old[i])}", file=out)
        a = old[idx] if idx < len(old) else None
        b = new[idx] if idx < len(new) else None
        for sign, side, record in (("-", "old", a), ("+", "new", b)):
            text = json.dumps(record) if record is not None else "<end of trace>"
            print(f"  {sign} {side} [{idx}] {text}", file=out)
        if a is not None and b is not None:
            fields = sorted(
                k for k in (a.keys() | b.keys()) - ignore if a.get(k) != b.get(k)
            )
            print(f"    fields: {', '.join(fields)}", file=out)
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
    args = parser.parse_args(argv)
    for trace_dir in (args.old_dir, args.new_dir):
        if not any(trace_dir.glob("*.jsonl")):
            print(f"error: no *.jsonl traces in {trace_dir}", file=sys.stderr)
            return 2
    return 1 if diff_trace_dirs(args.old_dir, args.new_dir, ignore=args.ignore) else 0


if __name__ == "__main__":
    sys.exit(main())
