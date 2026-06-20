"""Unit tests for the generated msgpack codec (srt/grpc/messages.py).

The proto (proto/sglang/runtime/v1/sglang.proto) is the IDL; this codec and the
Rust one (rust/sglang-grpc/src/msgpack.rs) are generated from it and must speak the
same wire. The golden vectors below are shared verbatim with the Rust test
(`golden_vector_*`) so the two codecs can never silently drift apart.
"""

import unittest

from sglang.srt.grpc import messages as M
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Canonical wire bytes, identical to the constants in rust/sglang-grpc/src/msgpack.rs.
GV_GENERATE = bytes.fromhex(
    "83a9696e7075745f69647393010203af73616d706c696e675f706172616d73"
    "83ae6d61785f6e65775f746f6b656e7310ae73746f705f746f6b656e5f6964"
    "739102a16e01a3726964a3616263"
)
GV_OPENAI = bytes.fromhex(
    "82a96a736f6e5f626f6479c4077b2261223a317dad74726163655f68656164" "65727381a178a179"
)


class TestMsgpackCodec(CustomTestCase):
    def test_golden_vector_generate_request(self):
        """Float-free message must be byte-identical to the Rust codec's output."""
        req = M.GenerateRequest(
            input_ids=[1, 2, 3],
            rid="abc",
            sampling_params=M.SamplingParams(
                n=1, max_new_tokens=16, stop_token_ids=[2]
            ),
        )
        self.assertEqual(M.encode(req), GV_GENERATE)
        self.assertEqual(M.decode(GV_GENERATE, M.GenerateRequest), req)

    def test_golden_vector_openai_request(self):
        """`bytes` field must be encoded as msgpack bin and round-trip."""
        req = M.OpenAIRequest(json_body=b'{"a":1}', trace_headers={"x": "y"})
        self.assertEqual(M.encode(req), GV_OPENAI)
        self.assertEqual(M.decode(GV_OPENAI, M.OpenAIRequest), req)
        # The bytes field is real msgpack bin (0xc4/0xc5/0xc6), not an int array.
        self.assertTrue(any(b in GV_OPENAI for b in (0xC4, 0xC5, 0xC6)))

    def test_round_trip_all_field_shapes(self):
        """Nested message + map + repeated + optional all survive a round-trip."""
        req = M.TextGenerateRequest(
            text="Hello, msgpack!",
            sampling_params=M.SamplingParams(
                temperature=0.7, top_p=0.95, max_new_tokens=128, stop=["</s>"]
            ),
            stream=True,
            rid="req-123",
            trace_headers={"traceparent": "00-abc-def-01"},
        )
        self.assertEqual(M.decode(M.encode(req), M.TextGenerateRequest), req)

    def test_unset_optionals_are_omitted(self):
        """omit_defaults: an all-default struct encodes to an empty msgpack map."""
        self.assertEqual(M.encode(M.SamplingParams()), b"\x80")

    def test_decode_is_tolerant(self):
        """Absent keys take defaults; unknown keys are ignored (proto-evolution safe)."""
        # Missing every field -> all defaults.
        empty = M.decode(b"\x80", M.GenerateRequest)
        self.assertEqual(empty.input_ids, [])
        self.assertIsNone(empty.rid)
        self.assertIsNone(empty.sampling_params)
        # An unknown field (from a newer proto) must not raise.
        import msgspec

        blob = msgspec.msgpack.encode({"rid": "x", "field_from_the_future": 7})
        self.assertEqual(M.decode(blob, M.GenerateRequest).rid, "x")


if __name__ == "__main__":
    unittest.main()
