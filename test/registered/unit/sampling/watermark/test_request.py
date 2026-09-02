import unittest
from types import SimpleNamespace

from starlette.datastructures import Headers

from sglang.srt.entrypoints.request_headers import apply_watermark_request
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.watermark import (
    TextSealConfig,
    WatermarkRegistry,
    WatermarkRequestConfig,
    WatermarkRequestError,
    parse_watermark_header,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWatermarkRequest(CustomTestCase):
    def test_header_contract(self):
        self.assertIsNone(parse_watermark_header(None))
        self.assertIsNone(parse_watermark_header(" off "))
        self.assertEqual(
            parse_watermark_header("textseal"),
            WatermarkRequestConfig(provider="textseal"),
        )

    def test_rejects_unknown_provider_and_malformed_header(self):
        cases = [
            ("unknown", "watermark_provider_unknown"),
            ("textseal;key_a=741852963", "watermark_invalid_request"),
            ("textseal;profile=default", "watermark_invalid_request"),
        ]
        for value, code in cases:
            with self.subTest(value=value):
                with self.assertRaises(WatermarkRequestError) as context:
                    parse_watermark_header(value)
                self.assertEqual(context.exception.code, code)
                self.assertNotIn("741852963", str(context.exception))

    def test_http_request_uses_only_header(self):
        request = SimpleNamespace(
            watermark={"provider": "textseal", "key_a": "741852963"}
        )
        apply_watermark_request(request, Headers())
        self.assertIsNone(request.watermark)

        apply_watermark_request(request, Headers({"x-sglang-watermark": "textseal"}))
        self.assertEqual(request.watermark.provider, "textseal")

    def test_batch_normalization_and_splitting(self):
        request = GenerateReqInput(
            text=["first", "second"],
            watermark=[
                WatermarkRequestConfig(provider="textseal"),
                None,
            ],
        )
        request.normalize_batch_and_arguments()

        self.assertEqual(request[0].watermark.provider, "textseal")
        self.assertIsNone(request[1].watermark)

    def test_registry_resolves_request_errors_without_secrets(self):
        config = TextSealConfig(key_a=741852963, key_b=963852741)
        registry = WatermarkRegistry(textseal=config)
        request = WatermarkRequestConfig(provider="textseal")
        self.assertIs(registry.resolve_request(request), config)

        cases = [
            (
                WatermarkRegistry(),
                request,
                "watermark_disabled",
            ),
            (
                registry,
                WatermarkRequestConfig(provider="unknown"),
                "watermark_provider_unknown",
            ),
        ]
        for candidate_registry, candidate_request, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(WatermarkRequestError) as context:
                    candidate_registry.resolve_request(candidate_request)
                self.assertEqual(context.exception.code, code)
                self.assertNotIn("741852963", str(context.exception))


if __name__ == "__main__":
    unittest.main()
