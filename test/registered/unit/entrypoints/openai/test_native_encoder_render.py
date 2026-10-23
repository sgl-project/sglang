"""Rendering a conversation must find the last user turn once, not per message.

``render_message`` in the dsv4 / dsv41 / dsv32 encoders needs the index of the
last user turn. It used to call ``find_last_user_index(messages)`` itself, and
that scan walks back over every trailing tool-result turn, so an agent loop
(one user turn followed by hundreds of tool calls and results) rendered in
O(n^2): 1,813 messages took 0.55 s, 97% of it in that scan. The encoders now
compute it once per conversation and pass it in.

These cases pin both halves of that change: the rendered prompt is identical to
the one rendered with the per-message lookup, and the lookup runs a bounded
number of times no matter how long the conversation is.
"""

import json
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv32, encoding_dsv41
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


def _agent_conversation(tool_turns, follow_up_user=False):
    """One user turn followed by ``tool_turns`` tool-call / tool-result pairs."""
    messages = [
        {"role": "system", "content": "You are a coding agent.", "tools": [_TOOL]},
        {"role": "user", "content": "Refactor the project."},
    ]
    for i in range(tool_turns):
        messages.append(
            {
                "role": "assistant",
                "content": f"Reading file {i}.",
                "reasoning_content": f"Need file {i}.",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": f"src/f{i}.py"}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"print({i})"}
        )
    if follow_up_user:
        messages.append({"role": "assistant", "content": "Done."})
        messages.append({"role": "user", "content": "Now add tests."})
    return messages


def _encode(module, messages, thinking_mode):
    return module.encode_messages(messages, thinking_mode=thinking_mode)


class TestNativeEncoderRender(CustomTestCase):
    MODULES = (encoding_dsv4, encoding_dsv41, encoding_dsv32)

    def test_prompt_matches_per_message_lookup(self):
        for module in self.MODULES:
            original = module.render_message

            def per_message_lookup(*args, last_user_idx=None, **kwargs):
                # The old behavior: every call finds the last user turn itself.
                return original(*args, **kwargs)

            for thinking_mode in ("chat", "thinking"):
                for follow_up_user in (False, True):
                    with self.subTest(
                        module=module.__name__,
                        thinking_mode=thinking_mode,
                        follow_up_user=follow_up_user,
                    ):
                        messages = _agent_conversation(12, follow_up_user)
                        with patch.object(module, "render_message", per_message_lookup):
                            expected = _encode(module, messages, thinking_mode)
                        self.assertEqual(
                            _encode(module, messages, thinking_mode), expected
                        )

    def test_last_user_lookup_is_not_per_message(self):
        for module in self.MODULES:
            with self.subTest(module=module.__name__):
                original = module.find_last_user_index
                with patch.object(
                    module, "find_last_user_index", side_effect=original
                ) as lookup:
                    _encode(module, _agent_conversation(300), "thinking")
                # One lookup for the render loop, plus at most the ones the
                # thinking-drop preprocessing does over the same list.
                self.assertLessEqual(lookup.call_count, 4)


if __name__ == "__main__":
    unittest.main()
