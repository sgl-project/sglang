"""Unit tests for the DeepSeek-V4.1 chat encoder -- no server, no model loading.

WARNING: the golden fixtures under ``fixtures/dsv41/`` are vendored verbatim
from the DeepSeek-V4.1 reference encoder drop (see the README there). They
pin the port byte-for-byte during bring-up and MUST be replaced with
self-authored cases before any upstream PR.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import copy
import json
import unittest
from pathlib import Path

from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv41
from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    IMAGE_PLACEHOLDER,
    SYSTEM_SP_TOKEN,
    encode_messages,
    merge_tool_messages,
    parse_message_from_completion_text,
    render_message,
)
from sglang.test.test_utils import CustomTestCase

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "dsv41"

BOS = "<｜begin▁of▁sentence｜>"
EFFORT_PREFIX = (
    SYSTEM_SP_TOKEN + "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)

TOOL_CALL_COMPLETION = (
    "  reason  </think>summary\n\n"
    "<｜DSML｜ calls>\n"
    '<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls><｜end▁of▁sentence｜>"
)


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


def _tool_call_messages() -> list:
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value","limit":2}',
                    },
                }
            ],
        },
    ]


def _load_fixture_case(input_file: Path) -> dict:
    """Reference harness semantics: case-level tools attach to messages[0]."""
    data = json.loads(input_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"messages": data}
    messages = copy.deepcopy(data["messages"])
    if "tools" in data:
        messages[0]["tools"] = data["tools"]
    return {
        "messages": messages,
        "context": data.get("context"),
        "thinking_mode": data.get("thinking_mode") or "chat",
        "reasoning_effort": data.get("reasoning_effort"),
    }


class TestGoldenFixtures(CustomTestCase):
    def test_reference_goldens_encode_byte_exact(self):
        input_files = sorted(FIXTURES_DIR.glob("test_input_*.json"))
        self.assertEqual(len(input_files), 5)
        for input_file in input_files:
            case_id = input_file.stem.rsplit("_", 1)[1]
            with self.subTest(case=case_id):
                case = _load_fixture_case(input_file)
                prompt, media = encode_messages(
                    case["messages"],
                    thinking_mode=case["thinking_mode"],
                    context=case["context"],
                    reasoning_effort=case["reasoning_effort"],
                    return_multi_modal_data=True,
                )
                golden = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text(
                    encoding="utf-8"
                )
                self.assertEqual(prompt, golden)
                if case_id == "5":
                    self.assertEqual(
                        [img["url"] for img in media["images"]],
                        ["examples/images/carrots.jpeg", "examples/images/corn.jpeg"],
                    )
                else:
                    self.assertEqual(media, {"images": []})


class TestReasoningEffort(CustomTestCase):
    def test_tiers_and_budgets_map_to_1_100(self):
        for effort, budget in [
            (None, 50),
            ("low", 25),
            ("high", 50),
            ("xhigh", 75),
            ("max", 100),
            (1, 1),
            (42, 42),
            (100, 100),
        ]:
            with self.subTest(effort=effort):
                prompt = encode_messages(
                    [{"role": "user", "content": "question"}],
                    thinking_mode="thinking",
                    reasoning_effort=effort,
                )
                self.assertEqual(
                    prompt,
                    BOS
                    + EFFORT_PREFIX.format(budget=budget)
                    + "<｜User｜>question<｜Assistant｜><think>",
                )

    def test_effort_prompt_only_on_first_message_in_thinking_mode(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
        later = render_message(
            1, messages, thinking_mode="thinking", reasoning_effort=100
        )
        chat = render_message(0, messages, thinking_mode="chat", reasoning_effort=100)
        self.assertNotIn("Reasoning Effort:", later)
        self.assertNotIn("Reasoning Effort:", chat)

    def test_chat_mode_bare_user_has_no_effort_and_no_system_token(self):
        prompt = encode_messages(
            [{"role": "user", "content": "hello"}],
            thinking_mode="chat",
            reasoning_effort="max",
        )
        self.assertEqual(prompt, BOS + "<｜User｜>hello<｜Assistant｜></think>")

    def test_rejects_unsupported_efforts(self):
        # bool is not an int budget; floats are the serving layer's job to map.
        for effort in [-1, 0, 101, "medium", "none", True, False, 1.5]:
            with self.subTest(effort=effort):
                with self.assertRaises(ValueError):
                    encode_messages(
                        [{"role": "user", "content": "question"}],
                        thinking_mode="thinking",
                        reasoning_effort=effort,
                    )


class TestSystemToken(CustomTestCase):
    def test_leading_system_message_uses_system_token(self):
        prompt = encode_messages(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ],
            thinking_mode="chat",
        )
        self.assertEqual(
            prompt,
            BOS
            + SYSTEM_SP_TOKEN
            + "You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>",
        )

    def test_empty_system_message_still_emits_system_token(self):
        """Why the serving layer must not insert an implicit empty system for V4.1."""
        with_empty = encode_messages(
            [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
            thinking_mode="chat",
        )
        bare = encode_messages(
            [{"role": "user", "content": "hi"}], thinking_mode="chat"
        )
        self.assertEqual(with_empty, BOS + SYSTEM_SP_TOKEN + bare[len(BOS) :])

    def test_empty_system_in_thinking_mode_matches_bare_user(self):
        """The effort prompt already forces the system token, so an empty system
        message changes nothing there."""
        with_empty = encode_messages(
            [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
            thinking_mode="thinking",
        )
        bare = encode_messages(
            [{"role": "user", "content": "hi"}], thinking_mode="thinking"
        )
        self.assertEqual(with_empty, bare)

    def test_mid_conversation_system_message_triggers_assistant_header(self):
        prompt = encode_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                {"role": "system", "content": "mid sys"},
            ],
            thinking_mode="thinking",
            reasoning_effort=88,
        )
        self.assertEqual(
            prompt,
            BOS
            + EFFORT_PREFIX.format(budget=88)
            + "sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"
            + SYSTEM_SP_TOKEN
            + "mid sys<｜Assistant｜><think>",
        )


class TestDsmlTags(CustomTestCase):
    def test_tool_instructions_use_spaced_tags(self):
        prompt = encode_messages(
            [
                {"role": "system", "content": "system", "tools": [_tool()]},
                {"role": "user", "content": "question"},
            ],
            thinking_mode="chat",
        )
        self.assertIn(
            "<｜DSML｜ calls>\n"
            '<｜DSML｜ invoke name="$TOOL_NAME">\n'
            '<｜DSML｜ parameter name="$PARAMETER_NAME" '
            'string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n'
            "...\n"
            "</｜DSML｜ invoke>",
            prompt,
        )
        for unspaced in (
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke",
            "<｜DSML｜parameter",
        ):
            self.assertNotIn(unspaced, prompt)

    def test_assistant_tool_call_renders_spaced_tags(self):
        prompt = render_message(1, _tool_call_messages(), thinking_mode="thinking")
        self.assertEqual(prompt, TOOL_CALL_COMPLETION)

    def test_completion_parse_round_trips_through_encoder(self):
        messages = _tool_call_messages()
        parsed = parse_message_from_completion_text(
            TOOL_CALL_COMPLETION, thinking_mode="thinking"
        )
        self.assertEqual(parsed["reasoning_content"], "  reason  ")
        self.assertEqual(parsed["content"], "summary")
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(
            json.loads(parsed["tool_calls"][0]["function"]["arguments"]),
            {"query": "value", "limit": 2},
        )
        self.assertEqual(
            encode_messages([parsed], thinking_mode="thinking", context=messages[:1]),
            TOOL_CALL_COMPLETION,
        )

    def test_completion_parse_rejects_unspaced_v4_tags(self):
        v4_output = (
            TOOL_CALL_COMPLETION.replace("｜DSML｜ calls", "｜DSML｜tool_calls")
            .replace("｜DSML｜ invoke", "｜DSML｜invoke")
            .replace("｜DSML｜ parameter", "｜DSML｜parameter")
        )
        with self.assertRaises(AssertionError):
            parse_message_from_completion_text(v4_output, thinking_mode="thinking")

    def test_non_object_tool_call_arguments_rejected(self):
        """Kept strict like the V4 encoder: scalar arguments are a client error."""
        messages = _tool_call_messages()
        messages[1]["tool_calls"][0]["function"]["arguments"] = '"scalar"'
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            encode_messages(messages, thinking_mode="chat")


class TestMultiTurnAndPreprocessing(CustomTestCase):
    def test_drop_thinking_without_tools(self):
        prompt = encode_messages(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                {"role": "user", "content": "q2"},
            ],
            thinking_mode="thinking",
        )
        self.assertIn(
            "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>", prompt
        )
        self.assertNotIn("r1", prompt)
        self.assertTrue(prompt.endswith("<｜User｜>q2<｜Assistant｜><think>"))

    def test_merge_tool_messages_creates_tool_result_blocks(self):
        merged = merge_tool_messages(
            [
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "a", "content": "r1"},
                {"role": "tool", "tool_call_id": "b", "content": "r2"},
            ]
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["role"], "user")
        self.assertEqual(
            [b["type"] for b in merged[1]["content_blocks"]],
            ["tool_result", "tool_result"],
        )

    def test_merge_tool_messages_keeps_user_fields_and_content_blocks(self):
        """User messages keep every message-level field and pre-built blocks."""
        merged = merge_tool_messages(
            [
                {
                    "role": "user",
                    "content": "joined",
                    "content_blocks": [{"type": "text", "text": "a"}],
                    "mask": 1,
                    "custom": "x",
                }
            ]
        )
        self.assertEqual(merged[0]["content_blocks"], [{"type": "text", "text": "a"}])
        self.assertEqual(merged[0]["mask"], 1)
        self.assertEqual(merged[0]["custom"], "x")

    def test_task_sp_token(self):
        prompt = encode_messages(
            [{"role": "user", "content": "classify me", "task": "query"}],
            thinking_mode="chat",
        )
        self.assertTrue(prompt.endswith("classify me<｜query｜>"))
        self.assertNotIn("<｜Assistant｜>", prompt)

    def test_attach_task_targets_mid_conversation_system(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": "s"},
        ]
        encoding_dsv41.attach_task_to_last_user_message(messages, "domain")
        self.assertEqual(messages[2]["task"], "domain")


class TestImages(CustomTestCase):
    def test_image_blocks_render_placeholders_in_order(self):
        prompt, media = encode_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "/a.png"}},
                        {"type": "text", "text": "and"},
                        {"type": "image_url", "image_url": "/b.png"},
                    ],
                }
            ],
            thinking_mode="chat",
            return_multi_modal_data=True,
        )
        self.assertEqual(
            prompt,
            BOS
            + f"<｜User｜>inspect\n\n{IMAGE_PLACEHOLDER}\n\nand\n\n{IMAGE_PLACEHOLDER}"
            "<｜Assistant｜></think>",
        )
        self.assertEqual(
            media,
            {
                "images": [
                    {"type": "image", "url": "/a.png"},
                    {"type": "image", "url": "/b.png"},
                ]
            },
        )

    def test_text_only_default_returns_plain_string(self):
        prompt = encode_messages(
            [{"role": "user", "content": "hi"}], thinking_mode="chat"
        )
        self.assertIsInstance(prompt, str)

    def test_rejects_image_placeholder_in_text(self):
        with self.assertRaises(ValueError):
            encode_messages(
                [{"role": "user", "content": f"hi {IMAGE_PLACEHOLDER}"}],
                thinking_mode="chat",
            )


class TestParityWithV4(CustomTestCase):
    def test_chat_without_system_or_tools_matches_v4_encoder(self):
        """V4.1 only diverges from V4 via the system token, the effort prompt
        and the DSML tags; plain chat turns must stay identical."""
        for messages in (
            [{"role": "user", "content": "hi"}],
            [
                {"role": "user", "content": "1+1=?"},
                {"role": "assistant", "content": "2", "reasoning_content": "r"},
                {"role": "user", "content": "2+2=?"},
            ],
        ):
            with self.subTest(turns=len(messages)):
                self.assertEqual(
                    encoding_dsv41.encode_messages(messages, thinking_mode="chat"),
                    encoding_dsv4.encode_messages(messages, thinking_mode="chat"),
                )


if __name__ == "__main__":
    unittest.main()
