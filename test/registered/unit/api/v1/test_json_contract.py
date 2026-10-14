"""JSON-contract tests for the generated Python API types (sglang.api.v1).

Mirrors rust/sglang-api-types/tests/json_contract.rs case for case (minus the
protobuf-arrival accessors, which have no Python counterpart), so the Rust and
Python emitters of sglang-api-codegen are held to one contract.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import unittest

from sglang.api.v1.api_types import (
    FinishAbort,
    FinishLength,
    FinishStop,
    GenerateMetaInfo,
    GenerateRequest,
    GenerateResponse,
    JsonContractError,
    LogprobEntry,
    MediaRef,
    SamplingParams,
    decode_FinishReason,
    decode_MediaInput,
    decode_OptionalInt64OrList,
    decode_OptionalStringOrList,
    decode_SamplingParamsOrList,
    decode_StringListOrList,
    decode_StringOrList,
    decode_TokenIdsOrList,
    encode_FinishReason,
    encode_MediaInput,
    encode_OptionalStringOrList,
    encode_StringOrList,
)
from sglang.test.test_utils import CustomTestCase


def compact(value) -> str:
    """serde_json::to_string parity: no whitespace."""
    return json.dumps(value, separators=(",", ":"))


def sp(text: str) -> SamplingParams:
    return SamplingParams.from_json_value(json.loads(text))


class TestSamplingParams(CustomTestCase):
    def test_defaults_match_hand_written(self):
        """Absent keys yield the schema defaults."""
        p = sp("{}")
        self.assertEqual(p.max_new_tokens, 128)
        self.assertEqual(p.temperature, 1.0)
        self.assertEqual(p.top_p, 1.0)
        self.assertEqual(p.top_k, 1 << 30)
        self.assertEqual(p.min_p, 0.0)
        self.assertEqual(p.repetition_penalty, 1.0)
        self.assertEqual(p.min_new_tokens, 0)
        self.assertEqual(p.n, 1)
        self.assertIs(p.ignore_eos, False)
        self.assertIs(p.skip_special_tokens, True)
        self.assertIs(p.spaces_between_special_tokens, True)
        self.assertIs(p.no_stop_trim, False)

    def test_constructor_defaults_match_absent_keys(self):
        """A dataclass built by hand carries the same schema defaults."""
        self.assertEqual(SamplingParams(), sp("{}"))

    def test_null_resets_default(self):
        """null_resets_default: an explicit null is the default, not an error."""
        p = sp('{"temperature": null, "top_k": null, "ignore_eos": null}')
        self.assertEqual(p.temperature, 1.0)
        self.assertEqual(p.top_k, 1 << 30)
        self.assertIs(p.ignore_eos, False)

    def test_max_new_tokens_absent_vs_null(self):
        """null_is_none: absent = 128, null = unbounded."""
        self.assertEqual(sp("{}").max_new_tokens, 128)
        self.assertIsNone(sp('{"max_new_tokens": null}').max_new_tokens)
        self.assertEqual(sp('{"max_new_tokens": 7}').max_new_tokens, 7)

    def test_denies_unknown_keys(self):
        """DENY policy with serde's own error text."""
        with self.assertRaises(JsonContractError) as ctx:
            sp('{"temperatur": 0.5}')  # codespell:ignore temperatur
        self.assertTrue(
            str(ctx.exception).startswith(
                "unknown field `temperatur`, expected one of"  # codespell:ignore temperatur
            ),
            str(ctx.exception),
        )

    def test_type_mismatch_uses_serde_text(self):
        with self.assertRaises(JsonContractError) as ctx:
            sp('{"temperature": "hot"}')
        self.assertEqual(str(ctx.exception), 'invalid type: string "hot", expected f64')

    def test_or_list_keeps_field_errors(self):
        """one_or_many over SamplingParams keeps field-level error texts."""
        with self.assertRaises(JsonContractError) as ctx:
            decode_SamplingParamsOrList(
                {"temperatur": 1}  # codespell:ignore temperatur
            )
        self.assertTrue(
            str(ctx.exception).startswith(
                "unknown field `temperatur`"  # codespell:ignore temperatur
            ),
            str(ctx.exception),
        )
        many = decode_SamplingParamsOrList([{"n": 2}, {}])
        self.assertEqual([p.n for p in many], [2, 1])

    def test_raw_json_passthrough(self):
        """raw_json: custom_params crosses untouched, integers included."""
        p = sp('{"custom_params": {"k": 3}}')
        self.assertEqual(p.custom_params, {"k": 3})
        self.assertEqual(p.to_json_value()["custom_params"], {"k": 3})

    def test_int_key_map(self):
        p = sp('{"logit_bias": {"42": -1}}')
        self.assertEqual(p.logit_bias, {"42": -1.0})


class TestGenerateRequest(CustomTestCase):
    def test_ignores_unknown_keys(self):
        """IGNORE policy: unported Python fields must not 400."""
        req = GenerateRequest.from_json_value(
            {
                "text": "hi",
                "priority": 3,
                "session_id": "s",
                "custom_logit_processor": "x",
            }
        )
        self.assertEqual(req.text, "hi")

    def test_media_input_shapes_dispatch(self):
        """Typed media: the value's shape picks the container form, nulls are
        kept per element, and a ref object decodes to a MediaRef."""
        req = GenerateRequest.from_json_value(
            {"image_data": [["data:image/png;base64,xx"], None, []]}
        )
        self.assertEqual(req.image_data, [["data:image/png;base64,xx"], None, []])
        flat = decode_MediaInput(["a", None, {"url": "b", "detail": "high"}])
        self.assertEqual(flat[:2], ["a", None])
        self.assertIsInstance(flat[2], MediaRef)
        self.assertEqual((flat[2].url, flat[2].detail), ("b", "high"))
        self.assertEqual(decode_MediaInput("u"), "u")
        self.assertEqual(decode_MediaInput([]), [])
        for value in (
            ["a", None, {"url": "b", "detail": "high"}],
            [["a", None], None, []],
            "u",
        ):
            self.assertEqual(encode_MediaInput(decode_MediaInput(value)), value)

    def test_media_ref_is_typed_and_strict(self):
        with self.assertRaises(JsonContractError) as ctx:
            decode_MediaInput(
                {"url": "u", "max_dynamic_patc": 6}
            )  # codespell:ignore patc
        self.assertIn("unknown field", str(ctx.exception))
        with self.assertRaises(JsonContractError):
            decode_MediaInput([{"format": "processor_output"}])
        with self.assertRaises(JsonContractError):
            decode_MediaInput(["a", ["b"]])

    def test_string_list_or_list_shape_dispatch(self):
        self.assertEqual(decode_StringListOrList(["a", "b"]), ["a", "b"])
        self.assertEqual(
            decode_StringListOrList([["a"], ["b", "c"]]), [["a"], ["b", "c"]]
        )
        self.assertEqual(decode_StringListOrList([]), [])

    def test_stream_null_resets_default(self):
        self.assertIs(GenerateRequest.from_json_value({"stream": None}).stream, False)
        self.assertIs(GenerateRequest.from_json_value({"stream": True}).stream, True)

    def test_round_trip_batch(self):
        body = {
            "text": ["a", "b"],
            "sampling_params": [{"n": 2}, {"temperature": 0.5}],
            "return_logprob": True,
            "bootstrap_host": [None, "h"],
        }
        req = GenerateRequest.from_json_value(body)
        back = req.to_json_value()
        self.assertEqual(back["text"], ["a", "b"])
        self.assertEqual(back["return_logprob"], True)
        self.assertEqual(back["bootstrap_host"], [None, "h"])
        self.assertEqual(back["sampling_params"][1]["temperature"], 0.5)


class TestCarriers(CustomTestCase):
    def test_one_or_many_flattens(self):
        self.assertEqual(decode_StringOrList("a"), "a")
        self.assertEqual(compact(encode_StringOrList("a")), '"a"')
        self.assertEqual(decode_StringOrList(["a", "b"]), ["a", "b"])
        self.assertEqual(compact(encode_StringOrList(["a", "b"])), '["a","b"]')
        with self.assertRaises(JsonContractError) as ctx:
            decode_StringOrList({"x": 1})
        self.assertEqual(
            str(ctx.exception),
            "invalid type: map, expected a value or an array of values for StringOrList",
        )

    def test_token_ids_or_list_shape_dispatch(self):
        """[1,2] is one id list, [[1],[2,3]] is a batch, [] is one empty list."""
        self.assertEqual(decode_TokenIdsOrList([1, 2]), [1, 2])
        self.assertEqual(decode_TokenIdsOrList([[1], [2, 3]]), [[1], [2, 3]])
        self.assertEqual(decode_TokenIdsOrList([]), [])

    def test_optional_carriers_keep_element_nulls(self):
        """PD bootstrap columns: [null, "h"] keeps per-item presence."""
        v = decode_OptionalStringOrList([None, "h"])
        self.assertEqual(v, [None, "h"])
        self.assertEqual(compact(encode_OptionalStringOrList(v)), '[null,"h"]')
        self.assertEqual(decode_OptionalInt64OrList(17000), 17000)


class TestFinishReason(CustomTestCase):
    def test_round_trips(self):
        """Tagged by "type"; matched keeps its wire shape (str / int / int list)."""
        for text in [
            '{"type":"stop","matched":"</s>"}',
            '{"type":"stop","matched":7}',
            '{"type":"stop","matched":[1,2]}',
            '{"type":"length","length":2}',
        ]:
            want = json.loads(text)
            parsed = decode_FinishReason(want)
            self.assertEqual(encode_FinishReason(parsed), want, text)

    def test_variant_classes(self):
        self.assertIsInstance(decode_FinishReason({"type": "stop"}), FinishStop)
        self.assertIsInstance(
            decode_FinishReason({"type": "length", "length": 2}), FinishLength
        )
        self.assertIsInstance(decode_FinishReason({"type": "abort"}), FinishAbort)

    def test_abort_emits_null_keys(self):
        """An abort carries all three keys, nulls included."""
        parsed = decode_FinishReason({"type": "abort", "message": "m"})
        back = encode_FinishReason(parsed)
        self.assertEqual(
            back,
            {"type": "abort", "message": "m", "status_code": None, "err_type": None},
        )

    def test_unknown_type_passes_through(self):
        """The forward-compat escape hatch: unknown tags round-trip unchanged."""
        raw = {"type": "paused", "step": 3}
        parsed = decode_FinishReason(raw)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(encode_FinishReason(parsed), raw)

    def test_malformed_known_type_passes_through(self):
        parsed = decode_FinishReason({"type": "stop", "matched": {"bad": "shape"}})
        self.assertIsInstance(parsed, dict)


class TestFrames(CustomTestCase):
    def test_logprob_tuple_shape(self):
        """[logprob, token_id, text|null]; the logprob slot is nullable."""
        e = LogprobEntry(logprob=-0.25, token_id=7, text=None)
        self.assertEqual(compact(e.to_json_value()), "[-0.25,7,null]")
        parsed = LogprobEntry.from_json_value([-0.5, 9, "x"])
        self.assertEqual(parsed.token_id, 9)
        self.assertEqual(parsed.text, "x")
        first_prefill = LogprobEntry.from_json_value([None, 10, None])
        self.assertIsNone(first_prefill.logprob)
        self.assertEqual(compact(first_prefill.to_json_value()), "[null,10,null]")
        with self.assertRaises(JsonContractError) as ctx:
            LogprobEntry.from_json_value([1.0, 2])
        self.assertEqual(
            str(ctx.exception), "invalid length 2, expected a 3-element tuple"
        )

    def test_generate_response_matches_frame_value_shape(self):
        """Key order (text, meta_info, output_ids), conditional keys, and
        finish_reason, cached_tokens_details, and dp_rank always present."""
        frame = GenerateResponse(
            text="ok",
            output_ids=[7, 8],
            meta_info=GenerateMetaInfo(
                id="client-rid",
                prompt_tokens=5,
                completion_tokens=2,
                finish_reason=None,
            ),
            index=None,
        )
        self.assertEqual(
            compact(frame.to_json_value()),
            '{"text":"ok","meta_info":{"id":"client-rid","prompt_tokens":5,"completion_tokens":2,"finish_reason":null,"cached_tokens_details":null,"dp_rank":null},"output_ids":[7,8]}',
        )

    def test_frame_round_trip_with_logprobs(self):
        wire = {
            "text": "ok",
            "meta_info": {
                "id": "r",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop", "matched": 2},
                "output_token_logprobs": [[-0.1, 5, "x"]],
                "output_top_logprobs": [None, [[-0.2, 6, None]]],
                "hidden_states": [[0.5, 0.25]],
                "e2e_latency": 0.01,
                "cached_tokens_details": None,
                "dp_rank": None,
            },
            "output_ids": [5],
            "index": 0,
        }
        frame = GenerateResponse.from_json_value(wire)
        self.assertIsInstance(frame.meta_info.finish_reason, FinishStop)
        self.assertEqual(frame.meta_info.output_top_logprobs[0], None)
        self.assertEqual(frame.meta_info.output_top_logprobs[1][0].token_id, 6)
        self.assertEqual(frame.to_json_value(), wire)


if __name__ == "__main__":
    unittest.main()
