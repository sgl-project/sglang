import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from sglang.srt.entrypoints.openai.encoding_glm import (
    order_glm_tool_results,
    resolve_glm_tool_result_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# zai-org/GLM-5.3-Flash chat template at HF revision 04c4e9e9.
_TEMPLATE_SHA256 = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
_TEMPLATE = (Path(__file__).parent / "glm53_flash_chat_template.jinja").read_text()

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look one item up",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _resolve(template=_TEMPLATE, architecture="Glm5NextForConditionalGeneration"):
    return resolve_glm_tool_result_template(
        hf_config=SimpleNamespace(architectures=[architecture]),
        tokenizer=SimpleNamespace(chat_template=template),
    )


def _render(template, messages, tools=None):
    # Mirrors how transformers renders chat templates.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    env.filters["tojson"] = lambda value, ensure_ascii=False: json.dumps(
        value, ensure_ascii=ensure_ascii
    )
    return env.from_string(template).render(
        messages=messages, tools=tools, add_generation_prompt=True
    )


def _assistant(*call_ids):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": "lookup", "arguments": {"q": call_id}},
            }
            for call_id in call_ids
        ],
    }


def _result(call_id=None, content=None):
    message = {"role": "tool", "content": content or f"result-{call_id}"}
    if call_id is not None:
        message["tool_call_id"] = call_id
    return message


class TestGlmToolResultOrdering(CustomTestCase):
    def test_reversed_results_follow_declared_call_order(self):
        messages = [
            _assistant("a", "b", "c"),
            _result("c"),
            _result("b"),
            _result("a"),
        ]

        ordered = order_glm_tool_results(messages)

        self.assertEqual(
            [message["tool_call_id"] for message in ordered[1:]],
            ["a", "b", "c"],
        )

    def test_missing_result_still_follows_declared_call_order(self):
        messages = [
            _assistant("a", "b", "c"),
            _result("c"),
            _result("a"),
        ]

        ordered = order_glm_tool_results(messages)

        self.assertEqual(
            [message["tool_call_id"] for message in ordered[1:]], ["a", "c"]
        )

    def test_list_of_outputs_is_flattened_in_declared_order(self):
        messages = [
            _assistant("a", "b", "c"),
            {
                "role": "tool",
                "content": [
                    {"tool_call_id": "c", "output": "result-c"},
                    {"tool_call_id": "a", "output": "result-a"},
                    {"tool_call_id": "b", "output": "result-b"},
                ],
            },
        ]

        ordered = order_glm_tool_results(messages)

        self.assertEqual(
            [message["content"][0]["tool_call_id"] for message in ordered[1:]],
            ["a", "b", "c"],
        )
        self.assertEqual(
            [message["content"][0]["output"] for message in ordered[1:]],
            ["result-a", "result-b", "result-c"],
        )

    def test_invalid_blocks_preserve_received_order(self):
        cases = {
            "missing id": [_result("b"), _result(None, "unidentified")],
            "duplicate result id": [_result("a", "first"), _result("a", "second")],
            "unknown result id": [_result("unknown")],
            "malformed list entry": [
                {
                    "role": "tool",
                    "content": [
                        {"tool_call_id": "a", "output": "result-a"},
                        "malformed",
                    ],
                }
            ],
        }
        for name, results in cases.items():
            with self.subTest(name=name):
                messages = [_assistant("a", "b"), *results]
                self.assertEqual(order_glm_tool_results(messages), messages)

    def test_duplicate_or_missing_call_ids_preserve_received_order(self):
        for name, call_ids in {
            "duplicate": ("a", "a"),
            "missing": ("a", None),
        }.items():
            with self.subTest(name=name):
                messages = [_assistant(*call_ids), _result("a"), _result("b")]
                self.assertEqual(order_glm_tool_results(messages), messages)

    def test_each_contiguous_result_block_uses_its_own_calls(self):
        messages = [
            _assistant("a", "b"),
            _result("b"),
            _result("a"),
            {"role": "user", "content": "continue"},
            _assistant("c", "d"),
            _result("d"),
            _result("c"),
        ]

        ordered = order_glm_tool_results(messages)

        self.assertEqual(
            [ordered[1]["tool_call_id"], ordered[2]["tool_call_id"]],
            ["a", "b"],
        )
        self.assertEqual(
            [ordered[5]["tool_call_id"], ordered[6]["tool_call_id"]],
            ["c", "d"],
        )


class TestGlmToolResultTemplate(CustomTestCase):
    def test_fixture_is_the_pinned_hf_template(self):
        self.assertEqual(
            hashlib.sha256(_TEMPLATE.encode("utf-8")).hexdigest(), _TEMPLATE_SHA256
        )

    def test_quadratic_glm_template_is_replaced(self):
        resolved = _resolve()

        self.assertIsNotNone(resolved)
        self.assertNotIn("namespace(tool_calls=none)", resolved)
        self.assertNotIn("has_dup_tool_result_id(block_start", resolved)
        self.assertIn("render_tool_response(messages[k])", resolved)

    def test_legacy_glm_architecture_is_supported(self):
        self.assertIsNotNone(_resolve(architecture="GlmMoeDsaForCausalLM"))

    def test_other_architecture_keeps_its_template(self):
        self.assertIsNone(_resolve(architecture="LlamaForCausalLM"))

    def test_unrecognized_template_is_not_rewritten(self):
        self.assertIsNone(_resolve(template="already linear"))
        self.assertIsNone(
            _resolve(template="has_dup_tool_result_id without known block")
        )

    def test_edited_sort_region_is_not_rewritten(self):
        """An upstream edit inside the excised region must disable the patch,
        not splice the old semantics over the new ones."""
        tampered = _TEMPLATE.replace(
            "ns_chk.can_sort = false", "ns_chk.can_sort = true"
        )
        self.assertIsNone(_resolve(template=tampered))


class TestGlmToolResultGoldenEquivalence(CustomTestCase):
    """Python reorder + patched template must render byte-identically to the
    stock quadratic template; split-off list-of-outputs entries used to take a
    different render_tool_response branch and change the output."""

    @classmethod
    def setUpClass(cls):
        cls.patched = _resolve()

    def _assert_equivalent(self, messages, tools=None):
        self.assertEqual(
            _render(_TEMPLATE, messages, tools=tools),
            _render(self.patched, order_glm_tool_results(messages), tools=tools),
        )

    def test_plain_result_blocks(self):
        cases = {
            "reversed": [_result("c"), _result("b"), _result("a")],
            "partial": [_result("c"), _result("a")],
            "duplicate id": [_result("a", "first"), _result("a", "second")],
            "unknown id": [_result("unknown")],
            "missing id": [_result("b"), _result(None, "unidentified")],
        }
        for name, results in cases.items():
            with self.subTest(name=name):
                self._assert_equivalent([_assistant("a", "b", "c"), *results])

    def test_list_of_outputs_blocks(self):
        cases = {
            "reversed": [
                {"tool_call_id": "b", "output": "result-b"},
                {"tool_call_id": "a", "output": "result-a"},
            ],
            "entry without output": [
                {"tool_call_id": "b", "output": "result-b"},
                {"tool_call_id": "a", "type": "text", "text": "SURPRISE"},
            ],
            "tool_reference-typed entry": [
                {"tool_call_id": "b", "output": "result-b"},
                {"tool_call_id": "a", "type": "tool_reference", "output": "result-a"},
            ],
            "none output": [
                {"tool_call_id": "a", "output": None},
                {"tool_call_id": "b", "output": "result-b"},
            ],
            "tool_reference output": [
                {
                    "tool_call_id": "a",
                    "output": [{"type": "tool_reference", "name": "lookup"}],
                },
                {"tool_call_id": "b", "output": "result-b"},
            ],
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                self._assert_equivalent(
                    [_assistant("a", "b"), {"role": "tool", "content": content}],
                    tools=_TOOLS,
                )

    def test_tool_reference_message_content(self):
        messages = [
            _assistant("a"),
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": [{"type": "tool_reference", "name": "lookup"}],
            },
        ]
        self._assert_equivalent(messages, tools=_TOOLS)

    def test_mixed_valid_and_invalid_blocks(self):
        messages = [
            _assistant("a", "b"),
            _result("b"),
            _result("a"),
            {"role": "user", "content": "continue"},
            _assistant("c"),
            _result("unknown"),
        ]
        self._assert_equivalent(messages)


if __name__ == "__main__":
    import unittest

    unittest.main()
