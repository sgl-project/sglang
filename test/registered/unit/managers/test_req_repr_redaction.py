"""SGLANG_REDACT_REQUEST_DATA keeps request data out of Req.__repr__, which
the scheduler watchdog and assertion messages format into logs, and refuses
startup options that log or dump request data. Pure CPU."""

import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.validation_hook import check_request_data_redaction
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req():
    req = Req(
        rid="rid-abc",
        origin_input_text="",
        origin_input_ids=array("q", [90001, 90002, 90003]),
        sampling_params=SamplingParams(max_new_tokens=8, stop="secret-stop"),
        vocab_size=100_000,
    )
    req.output_ids = array("q", [90004])
    return req


class TestReqReprRedaction(CustomTestCase):
    def test_redacted_repr_has_rid_and_lengths_only(self):
        with envs.SGLANG_REDACT_REQUEST_DATA.override(True):
            text = repr(_make_req())
        self.assertEqual(text, "Req(rid=rid-abc, input_len=3, output_len=1)")

    def test_redacted_repr_covers_lists_of_reqs(self):
        # The watchdog formats the batch's request list with an f-string `=`.
        reqs = [_make_req()]
        with envs.SGLANG_REDACT_REQUEST_DATA.override(True):
            text = f"{reqs=}"
        for secret in ("90001", "90004", "secret-stop"):
            self.assertNotIn(secret, text)

    def test_default_repr_is_unchanged(self):
        text = repr(_make_req())
        self.assertIn("90001", text)
        self.assertIn("secret-stop", text)


def _server_args(**overrides):
    args = dict(log_requests=False, crash_dump_folder=None, log_level="info")
    args.update(overrides)
    return SimpleNamespace(**args)


class TestRedactionStartupCheck(CustomTestCase):
    def test_request_data_options_are_refused(self):
        args = _server_args(
            log_requests=True, crash_dump_folder="/tmp/dump", log_level="DEBUG"
        )
        with envs.SGLANG_REDACT_REQUEST_DATA.override(True):
            with self.assertRaises(ValueError) as ctx:
                check_request_data_redaction(args)
        for option in ("--log-requests", "--crash-dump-folder", "--log-level debug"):
            self.assertIn(option, str(ctx.exception))

    def test_other_options_are_allowed(self):
        with envs.SGLANG_REDACT_REQUEST_DATA.override(True):
            check_request_data_redaction(_server_args(log_level="warning"))

    def test_options_are_allowed_without_the_flag(self):
        check_request_data_redaction(_server_args(log_requests=True))


if __name__ == "__main__":
    unittest.main()
