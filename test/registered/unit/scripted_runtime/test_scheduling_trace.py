from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.engine_state import EngineFlags
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.scheduling_trace import (
    SchedulingTraceWriter,
    diff_trace_dirs,
    first_divergence,
    load_trace,
    main,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_HEADER = {"kind": "header", "extra_engine_kwargs": {}}


def _schedule(step, **fields):
    return {"kind": "schedule", "step": step, "rids": ["r0"], **fields}


class _Req(SimpleNamespace):
    # Hashable by identity: the writer keeps the step's requests in a set.
    __hash__ = object.__hash__


class TestSchedulingTraceWriter(CustomTestCase):
    def test_retraction_seen_after_readmission(self):
        # pause_generation(retract) skips planning while paused, so the first
        # plan after resume already re-admits the retracted req into the batch.
        req = _Req(rid="a", retraction_count=0)
        scheduler = SimpleNamespace(
            parts=SimpleNamespace(
                lists=SimpleNamespace(
                    chunked_req=None, waiting_queue=[req], decoding=[]
                ),
                kv=SimpleNamespace(
                    token_to_kv_pool_allocator=SimpleNamespace(
                        available_size=lambda: 8
                    ),
                    tree_cache=SimpleNamespace(evictable_size=lambda: 0),
                ),
                new_token_ratio_tracker=SimpleNamespace(current=0.5),
            )
        )
        step = SimpleNamespace(
            reqs=[req],
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[4],
            prefix_lens=[9],
            chunked_req=None,
            prefill_reqs=lambda: [req],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "t.jsonl")
            writer = SchedulingTraceWriter(path, {})
            writer.record_schedule(scheduler, step, ScheduleSlot())
            req.retraction_count = 1
            writer.record_schedule(scheduler, step, ScheduleSlot())
            writer.record_schedule(scheduler, step, ScheduleSlot())
            writer.close()
            records = load_trace(path)

        self.assertEqual([r["retracted"] for r in records[1:]], [[], ["r0"], []])

    def test_paused_iterations_record_reader_values(self):
        inflight = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
        scheduler = SimpleNamespace(
            parts=SimpleNamespace(flags=EngineFlags()),
            hub=SimpleNamespace(
                load_inquirer=SimpleNamespace(
                    get_loads=lambda: SimpleNamespace(
                        num_running_reqs=0, num_waiting_reqs=1
                    )
                ),
                request_registry=SimpleNamespace(
                    is_fully_idle=lambda ignore_waiting=False: ignore_waiting,
                    # The record sorts the labels.
                    collect_inflight_reqs=lambda: [inflight[1], inflight[0]],
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "t.jsonl")
            writer = SchedulingTraceWriter(path, {})
            writer._labels(inflight)
            writer.observe_readers(scheduler)
            scheduler.parts.flags._engine_paused = True
            writer.observe_readers(scheduler)
            writer.close()
            records = load_trace(path)

        self.assertEqual(
            records[1:],
            [
                {
                    "kind": "paused",
                    "step": 0,
                    "idle": False,
                    "idle_ignore_waiting": True,
                    "inflight": ["r0", "r1"],
                    "load": {"running": 0, "waiting": 1},
                }
            ],
        )


class TestFirstDivergence(CustomTestCase):
    def test_reports_first_differing_record(self):
        old = [_schedule(1), _schedule(2, waiting=[]), _schedule(3)]
        new = [_schedule(1), _schedule(2, waiting=["r1"]), _schedule(3, mode="x")]

        self.assertIsNone(first_divergence(old, list(old)))
        self.assertEqual(first_divergence(old, new), 1)

    def test_truncated_trace_diverges_at_its_end(self):
        old = [_schedule(1), _schedule(2)]

        self.assertEqual(first_divergence(old, old[:1]), 1)
        self.assertEqual(first_divergence(old[:1], old), 1)

    def test_ignored_fields_do_not_diverge(self):
        old = [_schedule(1, kv_available=8)]
        new = [_schedule(1, kv_available=0)]

        self.assertEqual(first_divergence(old, new), 0)
        self.assertIsNone(first_divergence(old, new, ignore=["kv_available"]))


class TestDiffTraceDirs(CustomTestCase):
    def test_counts_differing_and_missing_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir, new_dir = Path(tmp, "old"), Path(tmp, "new")
            old_dir.mkdir()
            new_dir.mkdir()
            other_header = {**_HEADER, "extra_engine_kwargs": {"x": 1}}
            traces = {
                "same.jsonl": ([_HEADER, _schedule(1)], [_HEADER, _schedule(1)]),
                "differs.jsonl": (
                    [_HEADER, _schedule(1)],
                    [_HEADER, _schedule(1, mode="decode")],
                ),
                "config.jsonl": ([_HEADER, _schedule(1)], [other_header, _schedule(1)]),
                "only_old.jsonl": ([_HEADER, _schedule(1)], None),
            }
            for name, (old, new) in traces.items():
                _write(old_dir / name, old)
                if new is not None:
                    _write(new_dir / name, new)

            out = io.StringIO()
            num_differ = diff_trace_dirs(old_dir, new_dir, out=out)

        self.assertEqual(num_differ, 3)
        self.assertIn("IDENTICAL    same.jsonl", out.getvalue())
        self.assertIn("DIFFERS      differs.jsonl at record 1", out.getvalue())
        self.assertIn("CONFIG       config.jsonl", out.getvalue())
        self.assertIn("fields: mode", out.getvalue())
        self.assertIn("ONLY IN OLD  only_old.jsonl", out.getvalue())

    def test_all_records_set_fields_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir, new_dir = Path(tmp, "old"), Path(tmp, "new")
            old_dir.mkdir()
            new_dir.mkdir()
            old = [
                _HEADER,
                _schedule(1, waiting=["r1", "r2"], mode="extend"),
                _schedule(2, mode="decode"),
                _schedule(3, mode="decode"),
            ]
            new = [
                _HEADER,
                _schedule(1, waiting=["r2", "r1"], mode="extend"),
                _schedule(2, mode="extend"),
            ]
            _write(old_dir / "t.jsonl", old)
            _write(new_dir / "t.jsonl", new)

            out = io.StringIO()
            diff_trace_dirs(old_dir, new_dir, as_set=["waiting"], out=out)
            # Waiting order is not a difference; the first one is the mode.
            self.assertIn("DIFFERS      t.jsonl at record 2", out.getvalue())

            out = io.StringIO()
            diff_trace_dirs(old_dir, new_dir, all_records=True, out=out)
            text = out.getvalue()
            self.assertIn("[1] fields: waiting", text)
            self.assertIn("[2] fields: mode", text)
            self.assertIn("lengths: old 4, new 3 records", text)
            self.assertIn("counts: mode=1, waiting=1", text)

            _write(new_dir / "t.jsonl", old[:3])
            out = io.StringIO()
            diff_trace_dirs(old_dir, new_dir, out=out)
            self.assertIn("lengths: old 4, new 3 records", out.getvalue())
            self.assertIn("fields: kind, mode, rids, step", out.getvalue())

    def test_empty_or_missing_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main([tmp, str(Path(tmp, "missing"))]), 2)


def _write(path: Path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    unittest.main()
