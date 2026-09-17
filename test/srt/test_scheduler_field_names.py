"""ScheduleBatch snapshots resolve their field list once per class."""

import dataclasses
import unittest

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import _dataclass_field_names
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDataclassFieldNames(unittest.TestCase):
    def test_matches_dataclasses_fields(self):
        expected = tuple(f.name for f in dataclasses.fields(ScheduleBatch))
        self.assertEqual(_dataclass_field_names(ScheduleBatch), expected)

    def test_resolved_once_per_class(self):
        first = _dataclass_field_names(ScheduleBatch)
        self.assertIs(_dataclass_field_names(ScheduleBatch), first)

    def test_subclass_gets_its_own_fields(self):
        @dataclasses.dataclass
        class Extended(ScheduleBatch):
            extra_field: int = 0

        names = _dataclass_field_names(Extended)
        self.assertIn("extra_field", names)
        self.assertNotIn("extra_field", _dataclass_field_names(ScheduleBatch))


if __name__ == "__main__":
    unittest.main()
