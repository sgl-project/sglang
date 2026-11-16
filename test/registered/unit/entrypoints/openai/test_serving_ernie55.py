from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_ernie55 import Ernie55ChatEncodingAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def render_chat(
    messages,
    tools=None,
    *,
    add_generation_prompt=True,
    continue_final_message=False,
    enable_thinking=None,
    reasoning_effort=None,
    options=None,
):
    # Same contract as the checkpoint's module-level render_chat: an explicit
    # signature, so any keyword it does not know raises TypeError.
    return [len(messages)]


class _ErnieLikeTokenizer:
    def __init__(self):
        self.template_kwargs: list[dict] = []
        self.rendered: list[list[dict]] = []

    def render_chat(self, conversation, tools=None, **kwargs):
        return render_chat(conversation, tools, **kwargs)

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        return_dict=False,
        **kwargs,
    ):
        self.template_kwargs.append(kwargs)
        self.rendered.append(
            dict(
                messages=conversation,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
        )
        return render_chat(
            conversation,
            tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **kwargs,
        )


class TestErnie55ChatEncodingAdapter(CustomTestCase):
    def _encode(
        self, *, thinking=False, chat_template_kwargs=None, messages=None, **req
    ):
        messages = messages or [{"role": "user", "content": "hi"}]
        tokenizer = _ErnieLikeTokenizer()
        hooks = Mock()
        hooks.split_assistant_prefix = lambda msgs, request: (
            OpenAIServingChat._handle_last_assistant_message(None, msgs, request)
        )
        adapter = Ernie55ChatEncodingAdapter(
            tokenizer=tokenizer,
            reasoning_parser="ernie55",
            tool_call_parser="ernie55",
            chat_hooks=hooks,
        )
        request = ChatCompletionRequest(
            model="ernie",
            messages=messages,
            chat_template_kwargs=chat_template_kwargs,
            **req,
        )
        adapter.encode_messages(
            messages=[dict(m) for m in messages],
            request=request,
            tools=None,
            thinking=thinking,
            image_data=[],
        )
        return tokenizer

    def test_template_kwargs_the_renderer_rejects_are_dropped(self):
        """A frontend's DeepSeek-style `thinking` key reached ERNIE's strict Python
        renderer, which raises on unknown keywords where a Jinja template ignores
        them; the client's enable_thinking must still get through."""
        forwarded = self._encode(
            thinking=False,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        ).template_kwargs[-1]
        self.assertEqual(forwarded, {"enable_thinking": False})

    def test_thinking_toggle_is_left_to_the_template_default(self):
        """The handler's `thinking` decision defaults off while the ernie55
        reasoning parser's enable_thinking defaults on; passing the former to the
        template rendered a thinking-off prompt under a force-reasoning parser,
        so plain answers landed in reasoning_content with empty content."""
        forwarded = self._encode(thinking=False).template_kwargs[-1]
        self.assertNotIn("enable_thinking", forwarded)

    def test_continue_final_message_continues_the_assistant_turn(self):
        """continue_final_message was ignored: the trailing assistant prefix was
        rendered as a finished turn and a new one opened after it. The renderer
        must continue that turn instead; without the flag a trailing assistant
        turn becomes a user turn, as on the Jinja path."""
        conversation = [
            {"role": "user", "content": "3+5=?"},
            {"role": "assistant", "content": "The answer is"},
        ]
        rendered = self._encode(
            messages=conversation, continue_final_message=True
        ).rendered[-1]
        self.assertEqual(rendered["messages"][-1]["role"], "assistant")
        self.assertTrue(rendered["continue_final_message"])
        self.assertFalse(rendered["add_generation_prompt"])

        rendered = self._encode(messages=conversation).rendered[-1]
        self.assertEqual(rendered["messages"][-1]["role"], "user")
        self.assertFalse(rendered["continue_final_message"])
        self.assertTrue(rendered["add_generation_prompt"])


if __name__ == "__main__":
    unittest.main()
