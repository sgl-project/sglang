import asyncio
import json
import unittest
from unittest.mock import patch

from openai_harmony import Message, Role

from sglang.srt.entrypoints.context import StreamingHarmonyContext
from sglang.srt.entrypoints.harmony_utils import get_encoding
from sglang.srt.entrypoints.openai import serving_responses
from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.responses_adapters import decode_reasoning_state
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_harmony_serving():
    # The generator only needs frontend configuration/storage. Its production
    # methods and the native parser run without starting a model worker.
    serving = serving_responses.OpenAIServingResponses.__new__(
        serving_responses.OpenAIServingResponses
    )
    serving.use_harmony = True
    serving.enable_prompt_tokens_details = False
    serving.enable_response_store = True
    serving.response_store = {}
    serving.response_store_lock = asyncio.Lock()
    serving.msg_store = {}
    return serving


async def collect_harmony_stream(serving, request, chunks, *, finished="stop"):
    context = StreamingHarmonyContext([], {})
    count = 0

    async def generate():
        nonlocal count
        for index, chunk in enumerate(chunks):
            count += len(chunk)
            context.append_output(
                {
                    "output_ids": chunk,
                    "meta_info": {
                        "completion_tokens": count,
                        "finish_reason": (
                            {"type": finished} if index == len(chunks) - 1 else None
                        ),
                    },
                }
            )
            yield context

    async for event in serving.responses_stream_generator(
        request,
        {},
        generate(),
        context,
        "gpt-oss",
        None,
        RequestResponseMetadata(request_id=request.request_id),
        created_time=1,
        require_reasoning=True,
    ):
        yield json.loads(event.split("data: ", 1)[1])


class TestHarmonyReasoningEncoding(unittest.TestCase):
    def test_encodes_replay_state_only_when_reasoning_item_closes(self):
        text = "Let us compare the two ordinary choices before answering."
        tokens = get_encoding().render(
            Message.from_role_and_content(Role.ASSISTANT, text).with_channel("analysis")
        )
        # End a normal reasoning prefix before the message's final marker.
        chunks = [tokens[:-2], tokens[-2:]]
        for cutoff in (False, True):
            for summary in (False, True):
                with self.subTest(cutoff=cutoff, summary=summary):
                    serving = make_harmony_serving()
                    request = ResponsesRequest(
                        model="gpt-oss",
                        input="Compare these options.",
                        stream=True,
                        store=True,
                        include=["reasoning.encrypted_content"],
                        reasoning={"summary": "auto"} if summary else None,
                    )
                    partial_chunks = [tokens[:-2], tokens[-2:-1]] if cutoff else chunks

                    async def run():
                        events = []
                        checked_partial = False
                        with patch.object(
                            serving_responses,
                            "encode_reasoning_state",
                            wraps=serving_responses.encode_reasoning_state,
                        ) as encoder:
                            async for event in collect_harmony_stream(
                                serving,
                                request,
                                partial_chunks,
                                finished="length" if cutoff else "stop",
                            ):
                                events.append(event)
                                if (
                                    event["type"] == "response.reasoning_text.delta"
                                    and not checked_partial
                                ):
                                    # The encoder delegates to actual zlib/base64;
                                    # partial deltas do not carry replay state.
                                    self.assertEqual(encoder.call_count, 0)
                                    checked_partial = True
                            self.assertTrue(checked_partial)
                            self.assertEqual(encoder.call_count, 1)
                        return events

                    events = asyncio.run(run())
                    final = events[-1]["response"]
                    self.assertEqual(
                        final["status"], "incomplete" if cutoff else "completed"
                    )
                    item = final["output"][0]
                    self.assertEqual(
                        decode_reasoning_state(item["encrypted_content"]), text
                    )
                    self.assertEqual(item["content"][0]["text"], text)
                    self.assertEqual(
                        item["summary"],
                        [{"type": "summary_text", "text": text}] if summary else [],
                    )
                    added = next(
                        event["item"]
                        for event in events
                        if event["type"] == "response.output_item.added"
                    )
                    done = next(
                        event["item"]
                        for event in events
                        if event["type"] == "response.output_item.done"
                    )
                    self.assertIsNone(added["encrypted_content"])
                    self.assertEqual(added["id"], item["id"])
                    self.assertEqual(done, item)
                    self.assertEqual(
                        serving.response_store[final["id"]].output[0].model_dump(), item
                    )


if __name__ == "__main__":
    unittest.main()
