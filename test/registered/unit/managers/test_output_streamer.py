import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.output_streamer import (  # noqa: E402
    _append_optional_per_request_field,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOutputStreamerOptionalFields(CustomTestCase):
    def test_optional_field_stays_absent_when_no_request_returns_it(self):
        field = None
        field = _append_optional_per_request_field(field, False, "ignored", 1)
        field = _append_optional_per_request_field(field, False, "ignored", 2)

        self.assertIsNone(field)

    def test_optional_field_backfills_prior_requests(self):
        field = None
        field = _append_optional_per_request_field(field, False, "ignored", 1)
        field = _append_optional_per_request_field(field, True, "routed-b", 2)

        self.assertEqual(field, [None, "routed-b"])

    def test_optional_field_appends_none_for_later_disabled_requests(self):
        field = None
        field = _append_optional_per_request_field(field, True, "routed-a", 1)
        field = _append_optional_per_request_field(field, False, "ignored", 2)
        field = _append_optional_per_request_field(field, True, "routed-c", 3)

        self.assertEqual(field, ["routed-a", None, "routed-c"])


if __name__ == "__main__":
    unittest.main()
