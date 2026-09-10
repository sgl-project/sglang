import unittest

from sglang.srt.entrypoints.context import HarmonyContext, StreamingHarmonyContext
from sglang.srt.entrypoints.harmony_utils import get_encoding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# A tool-call header gpt-oss can sample with a duplicated "to=". It makes
# openai_harmony's StreamableParser raise HarmonyError at the <|message|>
# token and for every following token until the next message start.
MALFORMED_TOOL_CALL = (
    "<|channel|>commentary to=functions.get_weather"
    ' to=functions.get_weather<|message|>{"location": "SF"}<|call|>'
)
NEXT_FINAL = "<|start|>assistant<|channel|>final<|message|>All done.<|end|>"
FIRST_FINAL = "<|channel|>final<|message|>All done.<|end|>"


def _encode(text: str) -> list[int]:
    return get_encoding().encode(text, allowed_special="all")


def _engine_output(token_ids: list[int]) -> dict:
    return {
        "output_ids": token_ids,
        "meta_info": {
            "completion_tokens": len(token_ids),
            "finish_reason": {"type": "stop"},
        },
    }


class HarmonyParserErrorRecoveryTestCase(CustomTestCase):
    """append_output must survive malformed model output.

    Without the guard in HarmonyContext._process_token, the HarmonyError
    escapes append_output: the non-streaming Responses path reports it as a
    400 invalid_request_error and the streaming path dies mid-stream without
    an error event.
    """

    def test_non_streaming_append_output_recovers(self):
        ctx = HarmonyContext(messages=[], tool_sessions={})
        tokens = _encode(MALFORMED_TOOL_CALL + NEXT_FINAL)

        with self.assertLogs("sglang.srt.entrypoints.context", level="ERROR") as logs:
            ctx.append_output(_engine_output(tokens))

        # The malformed message is dropped and logged once; the parser
        # re-synchronizes so the following well-formed message survives.
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(len(ctx.messages), 1)
        self.assertEqual(ctx.messages[-1].channel, "final")
        self.assertEqual(ctx.messages[-1].content[0].text, "All done.")

    def test_streaming_append_output_recovers(self):
        ctx = StreamingHarmonyContext(messages=[], tool_sessions={})
        tokens = _encode(MALFORMED_TOOL_CALL + NEXT_FINAL)

        with self.assertLogs("sglang.srt.entrypoints.context", level="ERROR") as logs:
            # Feed one token per chunk, like the engine does when
            # --incremental-streaming-output is set.
            for token_id in tokens:
                ctx.append_output({"output_ids": [token_id], "meta_info": {}})

        self.assertEqual(len(logs.records), 1)
        self.assertEqual(len(ctx.messages), 1)
        self.assertEqual(ctx.messages[-1].channel, "final")
        self.assertEqual(ctx.messages[-1].content[0].text, "All done.")

    def test_well_formed_output_is_unaffected(self):
        ctx = HarmonyContext(messages=[], tool_sessions={})

        with self.assertNoLogs("sglang.srt.entrypoints.context", level="ERROR"):
            ctx.append_output(_engine_output(_encode(FIRST_FINAL)))

        self.assertEqual(len(ctx.messages), 1)
        self.assertEqual(ctx.messages[-1].channel, "final")
        self.assertEqual(ctx.messages[-1].content[0].text, "All done.")


if __name__ == "__main__":
    unittest.main()
