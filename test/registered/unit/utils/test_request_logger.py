import dataclasses
import unittest
from unittest.mock import patch

from sglang.srt.utils import request_logger
from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclasses.dataclass
class _Req:
    rid: str
    input_ids: list
    sampling_params: dict


def _make_req() -> _Req:
    return _Req(
        rid="rid-0",
        input_ids=list(range(8)),
        sampling_params={"max_new_tokens": 16, "watermark": {"key": "secret"}},
    )


class TestRequestLoggerFormatting(CustomTestCase):
    def test_redacts_secrets_and_skips_fields(self):
        req = _make_req()

        self.assertEqual(
            _transform_data_for_logging(req, 2048, {"input_ids"}),
            {
                "rid": "rid-0",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "watermark": {"key": "<redacted>"},
                },
            },
        )
        self.assertEqual(
            _dataclass_to_string_truncated(req, 2048, {"input_ids"}),
            "_Req(rid='rid-0', sampling_params={'max_new_tokens': 16, "
            "'watermark': {'key': '<redacted>'}})",
        )

    def test_keeps_all_fields_without_skip_names(self):
        logged = _transform_data_for_logging(_make_req())

        self.assertEqual(logged["input_ids"], list(range(8)))
        self.assertEqual(logged["sampling_params"]["watermark"], {"key": "<redacted>"})

    def test_skipped_fields_are_not_redacted(self):
        for format_fn in (_transform_data_for_logging, _dataclass_to_string_truncated):
            with (
                self.subTest(format_fn=format_fn.__name__),
                patch.object(
                    request_logger,
                    "redact_watermark_secrets",
                    wraps=request_logger.redact_watermark_secrets,
                ) as redact,
            ):
                format_fn(_make_req(), 2048, {"input_ids"})

                self.assertEqual(
                    set(redact.call_args_list[0].args[0]), {"rid", "sampling_params"}
                )


if __name__ == "__main__":
    unittest.main()
