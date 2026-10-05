"""
BDD Unit Tests for XGrammarGrammarBackend._sanitize_structural_format.

Tests correspond to Issue #42144: Recursively replacing null json_schema across all
structural tag container types (optional, star, plus, repeat, dispatch,
token_dispatch, token_triggered_tags).
"""

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestStructuralFormatSanitizerBDD(CustomTestCase):
    """BDD Specification Tests for XGrammar structural format sanitization."""

    # --- Scenario 1: Content-bearing container sanitization ---
    def test_scenario_1_content_containers(self):
        """Scenario 1: optional, star, plus, and repeat containers normalize content schema."""
        for c_type in ("optional", "star", "plus", "repeat"):
            with self.subTest(container_type=c_type):
                # Given
                null_schema = {"type": "json_schema", "json_schema": None}
                fmt = {"type": c_type, "content": null_schema}
                if c_type == "repeat":
                    fmt["min"], fmt["max"] = 1, 2

                # When
                XGrammarGrammarBackend._sanitize_structural_format(fmt)

                # Then
                self.assertEqual(
                    fmt["content"]["json_schema"],
                    {},
                    f"Scenario 1 failed for container type: {c_type}",
                )

    # --- Scenario 2: Rule-based dispatch container sanitization ---
    def test_scenario_2_dispatch_containers(self):
        """Scenario 2: dispatch and token_dispatch rule pairs normalize sub_format schema."""
        for d_type in ("dispatch", "token_dispatch"):
            with self.subTest(dispatch_type=d_type):
                # Given
                null_schema = {"type": "json_schema", "json_schema": None}
                trigger = "<func>" if d_type == "dispatch" else 42
                fmt = {"type": d_type, "rules": [[trigger, null_schema]]}

                # When
                XGrammarGrammarBackend._sanitize_structural_format(fmt)

                # Then
                self.assertEqual(
                    fmt["rules"][0][1]["json_schema"],
                    {},
                    f"Scenario 2 failed for dispatch type: {d_type}",
                )

    # --- Scenario 3: Tag list container sanitization ---
    def test_scenario_3_token_triggered_tags(self):
        """Scenario 3: token_triggered_tags container normalizes tags list schemas."""
        # Given
        null_schema = {"type": "json_schema", "json_schema": None}
        fmt = {"type": "token_triggered_tags", "tags": [null_schema]}

        # When
        XGrammarGrammarBackend._sanitize_structural_format(fmt)

        # Then
        self.assertEqual(
            fmt["tags"][0]["json_schema"],
            {},
            "Scenario 3 failed for token_triggered_tags",
        )

    # --- Scenario 4: Arbitrary recursive nesting ---
    def test_scenario_4_deeply_nested_composition(self):
        """Scenario 4: Deep nesting of optional -> repeat -> dispatch -> sequence -> leaf."""
        # Given
        leaf = {"type": "json_schema", "json_schema": None}
        nested = {
            "type": "optional",
            "content": {
                "type": "repeat",
                "min": 1,
                "max": 3,
                "content": {
                    "type": "dispatch",
                    "rules": [
                        [
                            "<tool_call>",
                            {"type": "sequence", "elements": [leaf]},
                        ]
                    ],
                },
            },
        }

        # When
        XGrammarGrammarBackend._sanitize_structural_format(nested)

        # Then
        sanitized_leaf_schema = nested["content"]["content"]["rules"][0][1]["elements"][
            0
        ]["json_schema"]
        self.assertEqual(
            sanitized_leaf_schema,
            {},
            "Scenario 4 failed: leaf schema was not sanitized in deep hierarchy",
        )

    # --- Scenario 5: Robustness against malformed or missing attributes ---
    def test_scenario_5_resilience_to_malformed_inputs(self):
        """Scenario 5: Sanitizer gracefully handles null, non-dict, or missing sub-keys."""
        cases = [
            ("string_payload", "not_a_dict"),
            ("null_root", None),
            ("optional_null_content", {"type": "optional", "content": None}),
            ("repeat_null_content", {"type": "repeat", "content": None}),
            ("dispatch_empty_rules", {"type": "dispatch", "rules": []}),
            ("dispatch_null_rules", {"type": "dispatch", "rules": None}),
            (
                "dispatch_incomplete_rule",
                {"type": "dispatch", "rules": [["<trigger>"]]},
            ),
            (
                "token_triggered_tags_null_tags",
                {"type": "token_triggered_tags", "tags": None},
            ),
            ("sequence_null_elements", {"type": "sequence", "elements": None}),
        ]
        for name, payload in cases:
            with self.subTest(case_name=name):
                # When & Then: must not raise exception
                XGrammarGrammarBackend._sanitize_structural_format(payload)

    # --- Regression check for existing containers ---
    def test_existing_containers_regression(self):
        """Ensure tag, sequence, or, and triggered_tags still work properly."""
        for c_type, key in [
            ("tag", "content"),
            ("sequence", "elements"),
            ("or", "elements"),
            ("triggered_tags", "tags"),
            ("tags_with_separator", "tags"),
        ]:
            with self.subTest(container_type=c_type):
                null_schema = {"type": "json_schema", "json_schema": None}
                if key == "content":
                    fmt = {"type": c_type, "content": null_schema}
                    target = fmt["content"]
                else:
                    fmt = {"type": c_type, key: [null_schema]}
                    target = fmt[key][0]

                XGrammarGrammarBackend._sanitize_structural_format(fmt)
                self.assertEqual(
                    target["json_schema"],
                    {},
                    f"Regression: existing container {c_type} broken",
                )
