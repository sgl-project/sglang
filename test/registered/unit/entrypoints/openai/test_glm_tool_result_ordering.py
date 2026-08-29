from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.chat_encoding import (
    order_glm_tool_results,
    resolve_glm_tool_result_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
    _VULNERABLE_TEMPLATE = """has_dup_tool_result_id
prefix
    {%- set ns_a = namespace(tool_calls=none) -%}
    quadratic association
    {%- endif -%}
{% endif -%}
{%- elif m.role == 'system' -%}
suffix"""

    def _resolve(self, template=None, architecture="Glm5NextForConditionalGeneration"):
        return resolve_glm_tool_result_template(
            hf_config=SimpleNamespace(architectures=[architecture]),
            tokenizer=SimpleNamespace(
                chat_template=template or self._VULNERABLE_TEMPLATE
            ),
        )

    def test_quadratic_glm_template_is_replaced(self):
        resolved = self._resolve()

        self.assertIsNotNone(resolved)
        self.assertNotIn("namespace(tool_calls=none)", resolved)
        self.assertNotIn("quadratic association", resolved)
        self.assertIn("render_tool_response(messages[k])", resolved)
        self.assertIn("{%- elif m.role == 'system' -%}", resolved)

    def test_legacy_glm_architecture_is_supported(self):
        self.assertIsNotNone(self._resolve(architecture="GlmMoeDsaForCausalLM"))

    def test_other_architecture_keeps_its_template(self):
        self.assertIsNone(self._resolve(architecture="LlamaForCausalLM"))

    def test_updated_or_unrecognized_glm_template_is_not_rewritten(self):
        self.assertIsNone(self._resolve(template="already linear"))
        self.assertIsNone(
            self._resolve(template="has_dup_tool_result_id without known block")
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
