from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.scheduling_trace import (
    diff_trace_dirs,
    first_divergence,
    main,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _schedule(step, **fields):
    return {"kind": "schedule", "step": step, "rids": ["r0"], **fields}


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
        old = [_schedule(1, batch_is_full=False)]
        new = [_schedule(1, batch_is_full=True)]

        self.assertEqual(first_divergence(old, new), 0)
        self.assertIsNone(first_divergence(old, new, ignore=["batch_is_full"]))


class TestDiffTraceDirs(CustomTestCase):
    def test_counts_differing_and_missing_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir, new_dir = Path(tmp, "old"), Path(tmp, "new")
            old_dir.mkdir()
            new_dir.mkdir()
            traces = {
                "same.jsonl": ([_schedule(1)], [_schedule(1)]),
                "differs.jsonl": ([_schedule(1)], [_schedule(1, mode="decode")]),
                "only_old.jsonl": ([_schedule(1)], None),
            }
            for name, (old, new) in traces.items():
                _write(old_dir / name, old)
                if new is not None:
                    _write(new_dir / name, new)

            out = io.StringIO()
            num_differ = diff_trace_dirs(old_dir, new_dir, out=out)

        self.assertEqual(num_differ, 2)
        self.assertIn("IDENTICAL    same.jsonl", out.getvalue())
        self.assertIn("DIFFERS      differs.jsonl at record 0", out.getvalue())
        self.assertIn("fields: mode", out.getvalue())
        self.assertIn("ONLY IN OLD  only_old.jsonl", out.getvalue())

    def test_empty_or_missing_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main([tmp, str(Path(tmp, "missing"))]), 2)


def _write(path: Path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    unittest.main()
