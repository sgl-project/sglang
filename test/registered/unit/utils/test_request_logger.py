"""Unit tests for srt/utils/request_logger.py — no server, no model loading.

Covers the derived properties and contracts of RequestLogger:
- truncation boundary math in the string/list elision helpers,
- the per-level skip-name sets (what each --log-requests-level redacts),
- header whitelisting (non-whitelisted headers must never reach the log),
- the SGLANG_LOG_REQUEST_EXCEEDED_MS latency gate (seconds -> ms conversion),
- configure() recomputing metadata so a stale skip set is never reused,
- the input_ids -> text backfill that only level >= 2 performs.
"""

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.utils.request_logger import (
    RequestLogger,
    _dataclass_to_string_truncated,
    _extract_whitelisted_headers,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@dataclass
class _FakeGenerateReq:
    """Duck-typed stand-in for GenerateReqInput: RequestLogger only touches
    rid/text/input_ids plus whatever dataclasses.fields() exposes."""

    rid: str = "req-1"
    text: Optional[Any] = "hello"
    input_ids: Optional[List[Any]] = None
    sampling_params: Any = field(default_factory=lambda: {"temperature": 0.7})


def _make_logger(
    *,
    log_requests: bool = True,
    level: int = 0,
    fmt: str = "text",
) -> RequestLogger:
    logger = RequestLogger(
        log_requests=log_requests,
        log_requests_level=level,
        log_requests_format=fmt,
        log_requests_target=None,
    )
    # Replace the real stdout logger with a mock so tests can inspect output.
    logger.targets = [Mock()]
    return logger


def _logged_text(logger: RequestLogger) -> str:
    return "".join(str(call.args[0]) for call in logger.targets[0].info.call_args_list)


class TestTruncationBoundaries(CustomTestCase):
    def test_string_at_exact_max_length_is_not_truncated(self):
        """Derived property: elision uses a strict `>` comparison, so a string
        of exactly max_length chars must round-trip unelided. Guards against
        an off-by-one rewrite (`>=`) silently elides full-size payloads."""
        s = "x" * 10
        self.assertEqual(_dataclass_to_string_truncated(s, 10), repr(s))
        self.assertEqual(_transform_data_for_logging(s, 10), s)

    def test_string_over_max_length_keeps_head_and_tail_halves(self):
        """Derived property: an elided string keeps max_length//2 chars from
        each end, so both the prompt prefix and the generation suffix stay
        visible in logs."""
        s = "abcdefghijk"  # 11 chars, max_length 10 -> halves of 5
        self.assertEqual(
            _dataclass_to_string_truncated(s, 10),
            f"{repr(s[:5])} ... {repr(s[-5:])}",
        )
        self.assertEqual(_transform_data_for_logging(s, 10), s[:5] + "..." + s[-5:])

    def test_long_list_elides_middle_with_marker(self):
        """Derived property: a list longer than max_length is logged as
        head + ellipsis marker + tail instead of dumping every element."""
        data = list(range(11))
        transformed = _transform_data_for_logging(data, 10)
        self.assertEqual(transformed, [0, 1, 2, 3, 4, "...", 6, 7, 8, 9, 10])

    def test_skip_names_only_apply_to_top_level_keys(self):
        """Derived property: skip_names redact top-level request fields; the
        same key nested deeper is kept. Guards against a rewrite that starts
        redacting recursively and hides nested payload structure."""
        data = {"text": "secret", "meta": {"text": "nested"}}
        transformed = _transform_data_for_logging(data, 100, skip_names={"text"})
        self.assertNotIn("text", transformed)
        self.assertEqual(transformed["meta"]["text"], "nested")


class TestMetadataLevels(CustomTestCase):
    def test_level_0_redacts_sampling_params_but_level_1_keeps_them(self):
        """Bookkeeping: the only difference between level 0 and level 1 is
        that level 0 also redacts sampling_params. If someone edits the level
        tables and loses that distinction, this goes red."""
        _, skip_names_l0, _ = _make_logger(level=0).metadata
        _, skip_names_l1, _ = _make_logger(level=1).metadata
        self.assertIn("sampling_params", skip_names_l0)
        self.assertNotIn("sampling_params", skip_names_l1)
        # Both levels still redact the bulky token fields.
        for skip_names in (skip_names_l0, skip_names_l1):
            self.assertIn("text", skip_names)
            self.assertIn("input_ids", skip_names)

    def test_level_2_truncates_at_2048_with_no_redaction(self):
        max_length, skip_names, out_skip_names = _make_logger(level=2).metadata
        self.assertEqual(max_length, 2048)
        self.assertIsNone(skip_names)
        self.assertIsNone(out_skip_names)

    def test_invalid_level_raises_but_only_when_logging_enabled(self):
        """Negative-branch contract: a bad level fails fast at construction
        when logging is on, yet a disabled logger tolerates any level (the
        gate short-circuits before validation)."""
        with self.assertRaises(ValueError):
            _make_logger(level=99)
        _make_logger(log_requests=False, level=99)  # must not raise

    def test_configure_recomputes_metadata(self):
        """Field lifecycle: configure() must rebuild the skip set, otherwise
        a runtime level change keeps redacting with the stale set."""
        logger = _make_logger(level=0)
        logger.configure(log_requests_level=2)
        max_length, skip_names, _ = logger.metadata
        self.assertEqual(max_length, 2048)
        self.assertIsNone(skip_names)


class TestHeaderWhitelisting(CustomTestCase):
    def test_non_whitelisted_headers_never_reach_the_log(self):
        """Sensitive-header guard: only whitelisted headers are extracted;
        an authorization header must not leak into log output."""
        request = SimpleNamespace(
            headers={"x-smg-routing-key": "route-1", "authorization": "secret"}
        )
        self.assertEqual(
            _extract_whitelisted_headers(request), {"x-smg-routing-key": "route-1"}
        )

        logger = _make_logger(level=0)
        logger.log_received_request(_FakeGenerateReq(), request=request)
        text = _logged_text(logger)
        self.assertIn("route-1", text)
        self.assertNotIn("secret", text)

    def test_empty_header_values_and_missing_request_are_dropped(self):
        """Negative branch: a whitelisted header with an empty value is
        omitted, and a missing raw request yields None instead of {}."""
        request = SimpleNamespace(headers={"x-smg-routing-key": ""})
        self.assertEqual(_extract_whitelisted_headers(request), {})
        self.assertIsNone(_extract_whitelisted_headers(None))


class TestRequestLogging(CustomTestCase):
    def test_disabled_logger_emits_nothing(self):
        """Negative gate: log_requests=False suppresses both received and
        finished logs entirely."""
        logger = _make_logger(log_requests=False)
        obj = _FakeGenerateReq()
        out = {"meta_info": {"e2e_latency": 1.0}}
        logger.log_received_request(obj)
        logger.log_finished_request(obj, out)
        logger.targets[0].info.assert_not_called()

    def test_level_0_redacts_sampling_params_from_text_output(self):
        logger = _make_logger(level=0)
        logger.log_received_request(_FakeGenerateReq())
        self.assertNotIn("sampling_params", _logged_text(logger))

    def test_json_format_event_names_and_payload(self):
        """Bookkeeping: downstream log consumers parse the request.received /
        request.finished event names and the rid field; renaming either
        silently breaks them."""
        logger = _make_logger(level=0, fmt="json")
        obj = _FakeGenerateReq(rid="rid-7")
        out = {"meta_info": {"e2e_latency": 1.0}, "text": "generated"}
        with patch("sglang.srt.utils.request_logger.log_json") as mock_log_json:
            logger.log_received_request(obj)
            logger.log_finished_request(obj, out)

        events = [call.args[1] for call in mock_log_json.call_args_list]
        self.assertEqual(events, ["request.received", "request.finished"])
        received_payload = mock_log_json.call_args_list[0].args[2]
        self.assertEqual(received_payload["rid"], "rid-7")
        # out_skip_names at level 0 redacts the generated text from the
        # finished event.
        finished_payload = mock_log_json.call_args_list[1].args[2]
        self.assertNotIn("text", finished_payload["out"])


class TestExceededMsGate(CustomTestCase):
    def test_latency_gate_converts_seconds_to_ms(self):
        """Derived property: e2e_latency is reported in seconds while the env
        threshold is in ms. 0.05 s must compare as 50 ms (suppressed by a
        100 ms threshold), 0.2 s as 200 ms (logged). Guards the x1000
        conversion."""
        with envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.override(100):
            logger = _make_logger(level=0)
            obj = _FakeGenerateReq()
            logger.log_finished_request(obj, {"meta_info": {"e2e_latency": 0.05}})
            logger.targets[0].info.assert_not_called()
            logger.log_finished_request(obj, {"meta_info": {"e2e_latency": 0.2}})
            logger.targets[0].info.assert_called_once()

    def test_gate_disabled_by_default_threshold(self):
        """Negative branch: the default threshold (-1) disables the gate, so
        even a 0-latency request is logged."""
        logger = _make_logger(level=0)
        logger.log_finished_request(
            _FakeGenerateReq(), {"meta_info": {"e2e_latency": 0.0}}
        )
        logger.targets[0].info.assert_called_once()


class TestInputIdsBackfill(CustomTestCase):
    def test_level_2_backfills_text_from_input_ids(self):
        """Contract: at level >= 2 a request that arrived pre-tokenized gets
        its text reconstructed via tokenizer.decode(skip_special_tokens=False)
        so full-payload logs stay readable."""
        logger = _make_logger(level=2)
        tokenizer = Mock()
        tokenizer.decode.return_value = "decoded"
        obj = _FakeGenerateReq(text=None, input_ids=[1, 2, 3])
        logger.log_received_request(obj, tokenizer=tokenizer)
        self.assertEqual(obj.text, "decoded")
        tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=False)

    def test_level_2_backfills_batched_input_ids_per_entry(self):
        """PD-disaggregation warmup sends a batch of token lists; each entry
        must be decoded separately instead of crashing or joining them."""
        logger = _make_logger(level=2)
        tokenizer = Mock()
        tokenizer.decode.side_effect = ["first", "second"]
        obj = _FakeGenerateReq(text=None, input_ids=[[1, 2], [3]])
        logger.log_received_request(obj, tokenizer=tokenizer)
        self.assertEqual(obj.text, ["first", "second"])

    def test_level_below_2_does_not_backfill(self):
        """Negative branch: redacting levels must not mutate the request."""
        logger = _make_logger(level=0)
        tokenizer = Mock()
        obj = _FakeGenerateReq(text=None, input_ids=[1, 2, 3])
        logger.log_received_request(obj, tokenizer=tokenizer)
        self.assertIsNone(obj.text)
        tokenizer.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
