import dataclasses
import unittest
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.utils.request_logger import (
    RequestLogger,
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclasses.dataclass
class _SamplingParams:
    temperature: float = 1.0
    ebnf: Optional[str] = None
    ebnf_full_assistant: bool = False


@dataclasses.dataclass
class _Req:
    rid: str = "r1"
    sampling_params: _SamplingParams = dataclasses.field(
        default_factory=_SamplingParams
    )


class TestSkipValueFieldsText(unittest.TestCase):
    def test_replaces_non_none_value_with_skipped(self):
        req = _Req(sampling_params=_SamplingParams(ebnf="root ::= " + "a" * 5000))
        out = _dataclass_to_string_truncated(
            req, skip_value_fields={"ebnf", "ebnf_full_assistant"}
        )
        self.assertIn("ebnf='<skipped>'", out)
        self.assertIn("ebnf_full_assistant='<skipped>'", out)
        self.assertNotIn("root ::=", out)
        # Other fields are preserved.
        self.assertIn("temperature=1.0", out)
        self.assertIn("rid='r1'", out)

    def test_preserves_none(self):
        req = _Req(sampling_params=_SamplingParams(ebnf=None))
        out = _dataclass_to_string_truncated(req, skip_value_fields={"ebnf"})
        self.assertIn("ebnf=None", out)

    def test_unset_skip_value_fields_keeps_value(self):
        req = _Req(sampling_params=_SamplingParams(ebnf="root ::= abc"))
        out = _dataclass_to_string_truncated(req)
        self.assertIn("root ::= abc", out)


class TestSkipValueFieldsJson(unittest.TestCase):
    def test_replaces_non_none_value_with_skipped(self):
        req = _Req(
            sampling_params=_SamplingParams(
                ebnf="root ::= " + "a" * 5000, ebnf_full_assistant=True
            )
        )
        out = _transform_data_for_logging(
            req, skip_value_fields={"ebnf", "ebnf_full_assistant"}
        )
        sp = out["sampling_params"]
        self.assertEqual(sp["ebnf"], "<skipped>")
        self.assertEqual(sp["ebnf_full_assistant"], "<skipped>")
        self.assertEqual(sp["temperature"], 1.0)

    def test_preserves_none(self):
        req = _Req(sampling_params=_SamplingParams(ebnf=None))
        out = _transform_data_for_logging(req, skip_value_fields={"ebnf"})
        self.assertIsNone(out["sampling_params"]["ebnf"])

    def test_dict_recursive(self):
        data = {
            "rid": "r3",
            "sampling_params": {"temperature": 1.0, "ebnf": "x" * 1000},
        }
        out = _transform_data_for_logging(data, skip_value_fields={"ebnf"})
        self.assertEqual(out["sampling_params"]["ebnf"], "<skipped>")
        self.assertEqual(out["sampling_params"]["temperature"], 1.0)


class TestRequestLoggerSkipFieldsEnv(unittest.TestCase):
    def _make_logger(self) -> RequestLogger:
        return RequestLogger(
            log_requests=True,
            log_requests_level=1,
            log_requests_format="text",
            log_requests_target=None,
        )

    def test_metadata_reads_env_var(self):
        with envs.SGLANG_LOG_REQUEST_SKIP_FIELDS.override("ebnf,ebnf_full_assistant"):
            logger = self._make_logger()
            _, _, _, skip_value_fields = logger.metadata
            self.assertEqual(skip_value_fields, {"ebnf", "ebnf_full_assistant"})

    def test_metadata_default_unset(self):
        with envs.SGLANG_LOG_REQUEST_SKIP_FIELDS.override(""):
            logger = self._make_logger()
            _, _, _, skip_value_fields = logger.metadata
            self.assertIsNone(skip_value_fields)


if __name__ == "__main__":
    unittest.main()
