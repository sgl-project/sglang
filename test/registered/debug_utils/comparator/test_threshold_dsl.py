import sys

import pytest

from sglang.srt.debug_utils.comparator.threshold_dsl import (
    DiffThresholdRule,
    evaluate_predicate,
    parse_diff_threshold_rules,
    parse_predicate,
    resolve_predicate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu", nightly=True)


def _ev(
    expr: str, *, rel: float = 0.0, max_abs: float = 0.0, mean_abs: float = 0.0
) -> bool:
    return evaluate_predicate(
        parse_predicate(expr), rel=rel, max_abs=max_abs, mean_abs=mean_abs
    )


class TestParsePredicate:
    @pytest.mark.parametrize(
        "expr",
        [
            "rel <= 0.0085",
            "rel < 1",
            "max_abs > 0",
            "mean_abs >= 1e-5",
            "rel <= 0.01 or max_abs <= 1e-4",
            "rel <= 0.01 and max_abs <= 1e-4",
            "(rel <= 0.01 and max_abs <= 1e-4) or mean_abs <= 1e-5",
            "0 <= rel < 1",
            "rel <= -0.0",
            "rel <= 0",
        ],
    )
    def test_valid_predicates_parse(self, expr: str) -> None:
        """All supported forms parse without error."""
        parse_predicate(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "abs(rel) < 1",
            "rel.x < 1",
            "foo < 1",
            "rel < 'x'",
            "",
            "rel <",
        ],
    )
    def test_invalid_predicates_raise(self, expr: str) -> None:
        """Unknown names, attribute access, bad types, and syntax errors raise ValueError."""
        with pytest.raises(ValueError):
            parse_predicate(expr)

    def test_unknown_name_message_lists_allowed(self) -> None:
        """The error for an unknown variable names the allowed variables."""
        with pytest.raises(ValueError, match="rel.*max_abs.*mean_abs"):
            parse_predicate("foo < 1")


class TestEvaluatePredicate:
    def test_rel_only_true_and_false(self) -> None:
        """A pure rel predicate uses only rel."""
        assert _ev("rel <= 0.01", rel=0.005) is True
        assert _ev("rel <= 0.01", rel=0.02) is False

    def test_le_boundary_inclusive(self) -> None:
        """<= includes the boundary; < excludes it."""
        assert _ev("rel <= 0.01", rel=0.01) is True
        assert _ev("rel < 0.01", rel=0.01) is False

    def test_or_short_circuit_semantics(self) -> None:
        """or passes if either side holds (near-zero rescue pattern)."""
        assert _ev("rel <= 0.0085 or max_abs <= 1e-3", rel=2.0, max_abs=2e-5) is True
        assert _ev("rel <= 0.0085 or max_abs <= 1e-3", rel=2.0, max_abs=0.5) is False

    def test_and_requires_both(self) -> None:
        """and passes only if both sides hold."""
        assert _ev("rel <= 0.01 and max_abs <= 1e-3", rel=0.005, max_abs=1e-4) is True
        assert _ev("rel <= 0.01 and max_abs <= 1e-3", rel=0.005, max_abs=0.5) is False

    def test_mean_abs_variable(self) -> None:
        """mean_abs is a usable variable."""
        assert _ev("mean_abs <= 1e-5", mean_abs=1e-6) is True
        assert _ev("mean_abs <= 1e-5", mean_abs=1e-4) is False

    def test_parentheses_grouping(self) -> None:
        """Parentheses override and/or precedence."""
        assert (
            _ev(
                "(rel <= 0.01 and max_abs <= 1e-4) or mean_abs <= 1e-5",
                rel=2.0,
                max_abs=2.0,
                mean_abs=1e-6,
            )
            is True
        )

    def test_chained_comparison(self) -> None:
        """Chained comparison follows Python all-must-hold semantics."""
        assert _ev("0 <= rel < 1", rel=0.5) is True
        assert _ev("0 <= rel < 1", rel=1.5) is False

    def test_bitwise_zero_predicate(self) -> None:
        """rel <= 0 passes only for an exactly-zero rel (bitwise)."""
        assert _ev("rel <= 0", rel=0.0) is True
        assert _ev("rel <= 0", rel=1e-12) is False


class TestDiffThresholdRule:
    def test_is_frozen_value_object(self) -> None:
        """DiffThresholdRule is a frozen, value-equal dataclass."""
        assert DiffThresholdRule(".*", "rel <= 1e-3") == DiffThresholdRule(
            ".*", "rel <= 1e-3"
        )
        with pytest.raises(Exception):
            DiffThresholdRule(".*", "rel <= 1e-3").pattern = "x"


class TestParseDiffThresholdRules:
    def test_none_and_empty_return_default(self) -> None:
        """Missing flag (None) and bare flag (empty list) both yield the caller's default."""
        assert parse_diff_threshold_rules(None, default_predicate="rel <= 0.001") == [
            DiffThresholdRule(".*", "rel <= 0.001")
        ]
        assert parse_diff_threshold_rules([], default_predicate="rel <= 0.001") == [
            DiffThresholdRule(".*", "rel <= 0.001")
        ]

    def test_single_float_shorthand(self) -> None:
        """A single float token expands to a global 'rel <= X' rule."""
        assert parse_diff_threshold_rules(
            ["0.0085"], default_predicate="rel <= 0.001"
        ) == [DiffThresholdRule(".*", "rel <= 0.0085")]

    def test_single_non_float_raises(self) -> None:
        """A single non-float token is a usage error (not a valid shorthand)."""
        with pytest.raises(ValueError):
            parse_diff_threshold_rules([".*expert.*"], default_predicate="rel <= 0.001")

    def test_pairs_parsed_in_order(self) -> None:
        """Flat regex/predicate tokens parse into ordered rules."""
        assert parse_diff_threshold_rules(
            [".*apple.*", "rel <= 0.01 or max_abs <= 1e-4", ".*", "rel <= 0.0085"],
            default_predicate="rel <= 0.001",
        ) == [
            DiffThresholdRule(".*apple.*", "rel <= 0.01 or max_abs <= 1e-4"),
            DiffThresholdRule(".*", "rel <= 0.0085"),
        ]

    def test_odd_number_of_tokens_raises(self) -> None:
        """An unpaired token is a usage error."""
        with pytest.raises(ValueError):
            parse_diff_threshold_rules(
                [".*apple.*", "rel <= 0.01", ".*orange.*"],
                default_predicate="rel <= 0.001",
            )

    def test_bad_predicate_raises(self) -> None:
        """A malformed predicate fails fast at parse time."""
        with pytest.raises(ValueError):
            parse_diff_threshold_rules(
                [".*", "rel <= "], default_predicate="rel <= 0.001"
            )


class TestResolvePredicate:
    def test_none_or_empty_returns_explicit_default(self) -> None:
        """No rules → the supplied default predicate."""
        assert (
            resolve_predicate("x", None, default_predicate="rel <= 1e-3")
            == "rel <= 1e-3"
        )
        assert (
            resolve_predicate("x", [], default_predicate="rel <= 1e-3") == "rel <= 1e-3"
        )

    def test_first_matching_pattern_wins(self) -> None:
        """Rules are tried in order; the first fullmatch wins (specific before general)."""
        rules = [DiffThresholdRule(".*expert.*", "P1"), DiffThresholdRule(".*", "P2")]
        assert (
            resolve_predicate("layer.expert.weight", rules, default_predicate="D")
            == "P1"
        )
        assert (
            resolve_predicate("layer.attn.weight", rules, default_predicate="D") == "P2"
        )

    def test_unmatched_name_raises(self) -> None:
        """A name matching no pattern is a fail-closed error."""
        with pytest.raises(ValueError, match="matched no --diff-threshold pattern"):
            resolve_predicate(
                "k_layernorm",
                [DiffThresholdRule(".*expert.*", "P")],
                default_predicate="D",
            )

    def test_fullmatch_semantics(self) -> None:
        """Matching is fullmatch: a partial pattern does not match a longer name (→ raises)."""
        assert (
            resolve_predicate(
                "expert", [DiffThresholdRule("expert", "P")], default_predicate="D"
            )
            == "P"
        )
        with pytest.raises(ValueError):
            resolve_predicate(
                "layer.expert.weight",
                [DiffThresholdRule("expert", "P")],
                default_predicate="D",
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
