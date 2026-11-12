"""Unit tests for the overlap scheduler's ScheduleBatch field snapshot helpers."""

import dataclasses
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    ScheduleBatch,
    schedule_batch_field_names,
    schedule_batch_field_values,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScheduleBatchFieldSnapshot(CustomTestCase):
    def test_values_follow_the_dataclass_fields(self):
        batch = ScheduleBatch(reqs=[])
        batch.seq_lens_sum = 17
        names = schedule_batch_field_names(ScheduleBatch)
        self.assertEqual(names, tuple(f.name for f in dataclasses.fields(batch)))
        values = schedule_batch_field_values(batch)
        self.assertEqual(len(values), len(names))
        for name, value in zip(names, values):
            with self.subTest(field=name):
                self.assertIs(value, batch.__dict__[name])

    def test_an_unassigned_field_reads_as_none(self):
        batch = ScheduleBatch(reqs=[])
        del batch.__dict__["seq_lens_sum"]
        snapshot = dict(
            zip(
                schedule_batch_field_names(ScheduleBatch),
                schedule_batch_field_values(batch),
            )
        )
        self.assertIsNone(snapshot["seq_lens_sum"])
        self.assertEqual(snapshot["reqs"], [])


if __name__ == "__main__":
    unittest.main()
