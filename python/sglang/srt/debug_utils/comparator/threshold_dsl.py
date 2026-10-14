import re
from dataclasses import dataclass
from functools import lru_cache
from types import CodeType
from typing import Optional

ALLOWED_NAMES: tuple[str, ...] = ("rel", "max_abs", "mean_abs")

_EVAL_GLOBALS: dict = {"__builtins__": {}}
_DUMMY_ENV: dict[str, float] = {name: 1.0 for name in ALLOWED_NAMES}


@dataclass(frozen=True)
class DiffThresholdRule:
    pattern: str
    predicate: str


def parse_diff_threshold_rules(
    raw: Optional[list[str]], *, default_predicate: str
) -> list[DiffThresholdRule]:
    if not raw:
        return [DiffThresholdRule(".*", default_predicate)]
    if len(raw) == 1:
        try:
            value = float(raw[0])
        except ValueError as e:
            raise ValueError(
                f"--diff-threshold with a single argument must be a float shorthand "
                f"(e.g. 0.0085); got {raw[0]!r}. For per-regex predicates pass "
                f"(regex predicate) pairs."
            ) from e
        return [DiffThresholdRule(".*", f"rel <= {value}")]
    if len(raw) % 2 != 0:
        raise ValueError(
            f"--diff-threshold expects a single float shorthand or (regex predicate) "
            f"pairs; got an odd number of arguments: {raw}"
        )
    rules = [DiffThresholdRule(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    for rule in rules:
        parse_predicate(rule.predicate)
    return rules


def resolve_predicate(
    name: str,
    diff_threshold_rules: Optional[list[DiffThresholdRule]],
    *,
    default_predicate: str,
) -> str:
    if not diff_threshold_rules:
        return default_predicate
    for rule in diff_threshold_rules:
        if re.fullmatch(rule.pattern, name):
            return rule.predicate
    raise ValueError(
        f"tensor {name!r} matched no --diff-threshold pattern "
        f"({[rule.pattern for rule in diff_threshold_rules]}); add a catch-all '.*' rule or a matching pattern."
    )


@lru_cache(maxsize=None)
def parse_predicate(expr: str) -> CodeType:
    try:
        code = compile(expr, "<predicate>", "eval")
    except SyntaxError as e:
        raise ValueError(f"invalid predicate {expr!r}: {e}") from e
    try:
        eval(code, _EVAL_GLOBALS, dict(_DUMMY_ENV))
    except Exception as e:
        raise ValueError(
            f"invalid predicate {expr!r}: {e}; allowed names are {ALLOWED_NAMES}."
        ) from e
    return code


def evaluate_predicate(
    code: CodeType, *, rel: float, max_abs: float, mean_abs: float
) -> bool:
    return bool(
        eval(
            code, _EVAL_GLOBALS, {"rel": rel, "max_abs": max_abs, "mean_abs": mean_abs}
        )
    )
