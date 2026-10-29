import unittest

from sglang.api.v1 import api_types as api
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1


class TestIntegerRanges(CustomTestCase):
    def _assert_integer_range(self, decode, minimum, maximum):
        for value in (minimum, 0, maximum):
            with self.subTest(value=value):
                self.assertEqual(decode(value), value)
        for value in (
            minimum - 1,
            maximum + 1,
            -(1 << 100),
            1 << 100,
            True,
            False,
            1.0,
        ):
            with self.subTest(value=value):
                with self.assertRaises(api.JsonContractError):
                    decode(value)

    def test_integer_helper_bounds(self):
        for check, bits, minimum, maximum in (
            (api._expect_int, 32, -(1 << 31), (1 << 31) - 1),
            (api._expect_int, 64, I64_MIN, I64_MAX),
            (api._expect_uint, 32, 0, U32_MAX),
            (api._expect_uint, 64, 0, U64_MAX),
        ):
            with self.subTest(helper=check.__name__, bits=bits):
                self._assert_integer_range(
                    lambda value: check(
                        value, expected="a value or an array of values", bits=bits
                    ),
                    minimum,
                    maximum,
                )
                with self.assertRaises(api.JsonContractError):
                    check("1", expected="an integer", bits=bits)

    def test_integer_field_bounds(self):
        for cls, field, minimum, maximum in (
            (api.SamplingParams, "max_new_tokens", I64_MIN, I64_MAX),
            (api.SamplingParams, "top_k", I64_MIN, I64_MAX),
            (api.SamplingParams, "stream_interval", I64_MIN, I64_MAX),
            (api.GenerateRequest, "routed_experts_start_len", I64_MIN, I64_MAX),
            (api.GenerateMetaInfo, "prompt_tokens", 0, U32_MAX),
            (api.GenerateMetaInfo, "image_tokens", 0, U32_MAX),
            (api.GenerateMetaInfo, "completion_tokens", 0, U64_MAX),
            (api.GenerateMetaInfo, "output_token_logprobs_length", 0, U64_MAX),
        ):
            with self.subTest(type=cls.__name__, field=field):
                self._assert_integer_range(
                    lambda value: getattr(cls.from_json_value({field: value}), field),
                    minimum,
                    maximum,
                )

    def test_repeated_integer_bounds(self):
        self._assert_integer_range(
            lambda value: api.DetokenizeRequest.from_json_value(
                {"tokens": [0, value]}
            ).tokens[1],
            I64_MIN,
            I64_MAX,
        )

    def test_tuple_integer_bounds(self):
        self._assert_integer_range(
            lambda value: (
                api.LogprobEntry.from_json_value([None, value, None]).token_id
            ),
            I64_MIN,
            I64_MAX,
        )

    def test_transparent_integer_bounds(self):
        self._assert_integer_range(api.decode_OptionalInt64, I64_MIN, I64_MAX)

    def test_integer_union_bounds(self):
        for decode in (
            api.decode_Int64OrList,
            api.decode_OptionalInt64OrList,
            api.decode_Matched,
        ):
            with self.subTest(decoder=decode.__name__):
                self._assert_integer_range(decode, I64_MIN, I64_MAX)
                self._assert_integer_range(
                    lambda value: decode([0, value])[1], I64_MIN, I64_MAX
                )

    def test_matched_preserves_strings(self):
        self.assertEqual(api.decode_Matched("1"), "1")
        with self.assertRaises(api.JsonContractError):
            api.decode_Matched([0, "1"])

    def test_nested_integer_list_bounds(self):
        self._assert_integer_range(
            lambda value: api.decode_TokenIdsOrList([[0], [value]])[1][0],
            I64_MIN,
            I64_MAX,
        )

    def test_integer_defaults_and_nulls(self):
        self.assertEqual(api.SamplingParams.from_json_value({}).max_new_tokens, 128)
        params = api.SamplingParams.from_json_value(
            {"max_new_tokens": None, "top_k": None, "stream_interval": None}
        )
        self.assertIsNone(params.max_new_tokens)
        self.assertEqual(params.top_k, 1 << 30)
        self.assertIsNone(params.stream_interval)
        self.assertEqual(
            api.DetokenizeRequest.from_json_value({"tokens": None}).tokens, []
        )
        self.assertIsNone(api.decode_OptionalInt64(None))
        self.assertEqual(
            api.decode_OptionalInt64OrList([None, I64_MIN, I64_MAX]),
            [None, I64_MIN, I64_MAX],
        )

    def test_integer_range_errors(self):
        for cls, field, value, expected in (
            (api.SamplingParams, "max_new_tokens", I64_MAX + 1, "i64"),
            (api.SamplingParams, "max_new_tokens", I64_MIN - 1, "i64"),
            (api.GenerateMetaInfo, "prompt_tokens", -1, "u32"),
            (api.GenerateMetaInfo, "prompt_tokens", U32_MAX + 1, "u32"),
            (api.GenerateMetaInfo, "completion_tokens", U64_MAX + 1, "u64"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(api.JsonContractError) as ctx:
                    cls.from_json_value({field: value})
                self.assertEqual(
                    str(ctx.exception),
                    f"invalid value: integer `{value}`, expected {expected}",
                )

    def test_tagged_integer_overflow_passes_through(self):
        for raw in (
            {"type": "stop", "matched": I64_MIN - 1},
            {"type": "stop", "matched": I64_MAX + 1},
            {"type": "stop", "matched": [0, I64_MAX + 1]},
            {"type": "length", "length": -1},
            {"type": "length", "length": U64_MAX + 1},
            {"type": "abort", "status_code": -1},
            {"type": "abort", "status_code": U32_MAX + 1},
        ):
            with self.subTest(raw=raw):
                parsed = api.decode_FinishReason(raw)
                self.assertIsInstance(parsed, dict)
                self.assertEqual(api.encode_FinishReason(parsed), raw)


if __name__ == "__main__":
    unittest.main()
