"""Chat-template probes for Mamba resume checkpoints."""

import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import jinja2

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GEN_THINK = "<|im_start|>assistant\n<think>\n"

# Strip reasoning from assistant history, as Qwen3-Thinking and Nemotron-Nano do.
STRIPS = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}{{ m.content.split('</think>')[-1].lstrip() }}"
    "{% else %}{{ m.content }}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}" + GEN_THINK.replace("\n", "\\n") + "{% endif %}"
)
# Keep reasoning so the next turn can append.
KEEPS = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n\n"
    "{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Rewriting the conversation start leaves no fixed resume depth.
REWRITES = "turns={{ messages | length }}\n" + STRIPS
# Drop empty assistant answers, including turns cut off during reasoning.
DROPS_EMPTY = STRIPS.replace(
    "{% for m in messages %}",
    "{% for m in messages if m.role != 'assistant' or m.content %}",
)


class CharTokenizer:
    def __init__(self, template):
        self._template = jinja2.Template(template.replace("\\n", "\n"))
        self.renders = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        self.renders += 1
        if kwargs.get("raise_in_template"):
            raise ValueError("template error")
        return self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )


def _server(template, parsed=True):
    server = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=CharTokenizer(template)),
        template_manager=SimpleNamespace(
            python_chat_template=None, reasoning_config=None
        ),
        reasoning_parser="nemotron_3" if parsed else None,
        _prompt_text_round_trip_is_lossy=False,
        _tokenizer_auto_adds_specials=False,
        _reasoning_detector=(
            SimpleNamespace(
                think_start_token="<think>",
                think_end_token="</think>",
                reasoning_default="enable_thinking",
            )
            if parsed
            else None
        ),
        _resume_depths=OrderedDict(),
        _resume_depths_lock=threading.Lock(),
    )
    for name in (
        "_render_and_encode_chat_template",
        "_probe_resume_depth",
        "_resume_depth",
        "_resume_checkpoint_pos",
        "_get_reasoning_from_request",
    ):
        setattr(server, name, getattr(OpenAIServingChat, name).__get__(server))
    return server


class TestResumeDepth(CustomTestCase):
    def test_stripping_template_diverges_where_the_generation_prompt_opens(self):
        for parsed in (True, False):
            with self.subTest(parsed=parsed):
                server = _server(STRIPS, parsed)
                self.assertEqual(server._resume_depth({}, {}), len("<think>\n"))

    def test_turn_cut_off_in_its_reasoning_sets_the_depth(self):
        # Dropping an empty answer moves divergence to the generation prompt start.
        self.assertEqual(
            _server(DROPS_EMPTY)._resume_depth({}, {}), len("assistant\n<think>\n")
        )

    def test_template_that_keeps_the_output_gives_none(self):
        for parsed in (True, False):
            with self.subTest(parsed=parsed):
                self.assertIsNone(_server(KEEPS, parsed)._resume_depth({}, {}))

    def test_template_that_rewrites_earlier_history_gives_none(self):
        self.assertIsNone(_server(REWRITES)._resume_depth({}, {}))

    def test_probe_runs_once_per_template_kwargs(self):
        server = _server(STRIPS)
        tokenizer = server.tokenizer_manager.tokenizer
        server._resume_depth({}, {})
        renders = tokenizer.renders
        server._resume_depth({}, {})
        self.assertEqual(tokenizer.renders, renders)
        server._resume_depth({"enable_thinking": True}, {})
        self.assertEqual(tokenizer.renders, 2 * renders)

    def test_concurrent_first_requests_probe_once(self):
        # Request conversion can probe the same template from multiple threads.
        server = _server(STRIPS)
        probe, calls, entered = server._probe_resume_depth, [], threading.Event()
        release = threading.Event()

        def slow_probe(*args):
            calls.append(1)
            entered.set()
            release.wait(5)
            return probe(*args)

        server._probe_resume_depth = slow_probe
        threads = [
            threading.Thread(target=server._resume_depth, args=({}, {}))
            for _ in range(2)
        ]
        threads[0].start()
        self.assertTrue(entered.wait(5))
        entered.clear()
        threads[1].start()
        # A second probe would set the event; the lock must keep it waiting.
        entered.wait(1)
        release.set()
        for thread in threads:
            thread.join()
        self.assertEqual(len(calls), 1)

    def test_probe_failure_gives_none_and_is_cached(self):
        server = _server(STRIPS)
        self.assertIsNone(server._resume_depth({"raise_in_template": True}, {}))
        renders = server.tokenizer_manager.tokenizer.renders
        self.assertIsNone(server._resume_depth({"raise_in_template": True}, {}))
        self.assertEqual(server.tokenizer_manager.tokenizer.renders, renders)

    def test_unserializable_kwargs_give_none(self):
        self.assertIsNone(_server(STRIPS)._resume_depth({"x": object()}, {}))


class TestResumeCheckpointPos(CustomTestCase):
    def _pos(self, server, **request_kwargs):
        request = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "Hi"}], **request_kwargs
        )
        return server._resume_checkpoint_pos(
            request=request, prompt_len=100, template_kwargs={}, encode_kwargs={}
        )

    def test_default_thinking_template_resumes_without_a_thinking_kwarg(self):
        with envs.SGLANG_OPT_MAMBA_CHAT_RESUME_CHECKPOINT.override(True):
            self.assertEqual(self._pos(_server(STRIPS)), 100 - len("<think>\n"))

    def test_no_resume_point_without_reasoning_or_the_env(self):
        with envs.SGLANG_OPT_MAMBA_CHAT_RESUME_CHECKPOINT.override(True):
            off = {"chat_template_kwargs": {"enable_thinking": False}}
            self.assertIsNone(self._pos(_server(STRIPS), **off))
            self.assertIsNone(self._pos(_server(STRIPS, parsed=False)))
        self.assertIsNone(self._pos(_server(STRIPS)))


if __name__ == "__main__":
    unittest.main()
