"""glm47 full-assistant EBNF: the grammar's thinking state must survive
chat templates that ignore ``enable_thinking``.

GLM-5.3-Flash's chat template ignores the ``enable_thinking`` kwarg and
always ends the rendered prompt with the think-start token. The glm47
grammar used to derive its thinking state from the raw kwarg: with
``enable_thinking=False`` it emitted ``thinking_block ::= ""`` and excluded
the think-end token everywhere else, so a model already inside a template-
opened thinking block could never close it and generated until
``max_tokens`` with empty content and no tool call (issue #48336). The
grammar now allows an optional, possibly empty thinking block, which is
correct for both template styles.
"""

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function
from sglang.srt.function_call.glm4_moe_detector import (
    GlmSpecialTokenConfig,
    generate_glm_grammar,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

T = GlmSpecialTokenConfig()

_FUNCS = [
    Function(
        name="shell",
        description="run a shell command",
        parameters={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    )
]

THINK_CLOSE_THEN_ANSWER = f"{T.end_of_thinking}391"
TOOL_CALL = (
    f"{T.begin_of_tool_call}shell"
    f"{T.begin_of_key}cmd{T.end_of_key}"
    f"{T.begin_of_value}ls{T.end_of_value}"
    f"{T.end_of_tool_call}"
)


def _grammar(enable_thinking, accommodate=True):
    ebnf = generate_glm_grammar(
        enable_thinking=enable_thinking,
        functions=_FUNCS,
        special_tokens=T,
        chat_template_version="glm47",
        accommodate_chat_template=accommodate,
        allow_multiple_assistant_turns=False,
    )
    return xgr.Grammar.from_ebnf(ebnf)


def _accepts(grammar, text):
    return _is_grammar_accept_string(grammar, text)


class TestGlm47GrammarThinkingState(CustomTestCase):
    def test_thinking_false_allows_closing_a_template_opened_think_block(self):
        """Regression for #48336: the template ignored enable_thinking and
        already opened the thinking block, so the very first generated token
        may be the think-end token. Banning it traps the model until
        max_tokens."""
        self.assertTrue(_accepts(_grammar(False), THINK_CLOSE_THEN_ANSWER))

    def test_thinking_false_still_allows_direct_answer(self):
        """Templates that honor enable_thinking render no think-start; the
        model goes straight to the answer."""
        self.assertTrue(_accepts(_grammar(False), "391"))

    def test_thinking_false_still_allows_direct_tool_call(self):
        self.assertTrue(_accepts(_grammar(False), TOOL_CALL))

    def test_thinking_false_allows_think_close_then_tool_call(self):
        self.assertTrue(_accepts(_grammar(False), f"{T.end_of_thinking}{TOOL_CALL}"))

    def test_thinking_true_still_requires_thinking_first(self):
        grammar = _grammar(True)
        self.assertFalse(_accepts(grammar, "391"))
        self.assertTrue(_accepts(grammar, THINK_CLOSE_THEN_ANSWER))

    def test_thinking_true_non_accommodate_accepts_full_block(self):
        grammar = _grammar(True, accommodate=False)
        self.assertTrue(
            _accepts(
                grammar,
                f"{T.begin_of_thinking}why{T.end_of_thinking}391",
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
