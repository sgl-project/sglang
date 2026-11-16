"""Admission checks for stop patterns matched by CPython on the decode path."""

try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python < 3.11

# Keep the cost policy aligned with rust/sglang-server/src/utils/regex.rs.
_MAX_REPEAT_PRODUCT = 512
_MAX_AMBIGUITY_DEGREE = 1
_MAX_NESTING = 64
_REPEAT_OPS = (
    sre_parse.MAX_REPEAT,
    sre_parse.MIN_REPEAT,
    getattr(sre_parse, "POSSESSIVE_REPEAT", None),
)
_ATOMIC_GROUP = getattr(sre_parse, "ATOMIC_GROUP", None)
_SIMPLE_OPS = frozenset(
    (
        sre_parse.LITERAL,
        sre_parse.NOT_LITERAL,
        sre_parse.IN,
        sre_parse.ANY,
        sre_parse.AT,
    )
)
_UNSUPPORTED_OPS = {
    sre_parse.ASSERT: "look-around",
    sre_parse.ASSERT_NOT: "look-around",
    sre_parse.GROUPREF: "backreferences",
    sre_parse.GROUPREF_EXISTS: "conditional groups",
}


def validate_stop_regex(pattern: str) -> sre_parse.SubPattern:
    """Parse a stop pattern and reject compounding backtracking costs.

    This is a conservative admission policy, not a wall-clock runtime guarantee.
    Python-only constructs we cannot cost-check are rejected, as in the Rust
    frontend. The parsed tree is returned so normalization can derive the
    existing scan bound without parsing the pattern a second time.
    """
    try:
        parsed = sre_parse.parse(pattern)
    except OverflowError as error:
        raise ValueError(
            f"stop_regex {pattern!r} has a repeat count too large"
        ) from error
    try:
        _check_cost(parsed, repeat_product=1, depth=0, repeated=False)
    except ValueError as error:
        raise ValueError(f"stop_regex {pattern!r} {error}") from error
    return parsed


def _check_cost(
    subpattern: sre_parse.SubPattern,
    *,
    repeat_product: int,
    depth: int,
    repeated: bool,
) -> int:
    if depth > _MAX_NESTING:
        raise ValueError(f"nests too deeply (limit {_MAX_NESTING})")

    degree = 0
    for op, value in subpattern:
        if op in _SIMPLE_OPS:
            continue
        if op == sre_parse.SUBPATTERN or op == _ATOMIC_GROUP:
            body = value[-1] if op == sre_parse.SUBPATTERN else value
            degree += _check_cost(
                body,
                repeat_product=repeat_product,
                depth=depth + 1,
                repeated=repeated,
            )
        elif op == sre_parse.BRANCH:
            # CPython may fold a literal alternation into a character class.
            # IN is safe; a surviving BRANCH still offers multiple parses.
            if repeated:
                raise ValueError("alternates inside a repetition")
            degree += max(
                _check_cost(
                    branch,
                    repeat_product=repeat_product,
                    depth=depth + 1,
                    repeated=repeated,
                )
                for branch in value[1]
            )
        elif op in _REPEAT_OPS:
            minimum, maximum, body = value
            unbounded = maximum == sre_parse.MAXREPEAT
            factor = minimum if unbounded else maximum
            product = repeat_product * max(1, factor)
            if product >= _MAX_REPEAT_PRODUCT:
                raise ValueError(f"has a repeat-count product >= {_MAX_REPEAT_PRODUCT}")
            body_degree = _check_cost(
                body, repeat_product=product, depth=depth + 1, repeated=True
            )
            if unbounded:
                # A single repeat of a fixed-width body is one length choice.
                # Repeating a variable-width body compounds those choices.
                low_width, high_width = body.getwidth()
                if body_degree or low_width != high_width:
                    raise ValueError(
                        "repeats a variable-length expression without a bound"
                    )
                degree += 1
            else:
                degree += maximum * body_degree + (minimum != maximum)
        else:
            construct = _UNSUPPORTED_OPS.get(op, f"an unsupported construct ({op})")
            raise ValueError(f"uses {construct}, which cannot be cost-checked")

        # Independent choices also compound when spelled flat: a*a*a*b.
        if degree > _MAX_AMBIGUITY_DEGREE:
            raise ValueError(
                f"has too many independent repetition choices "
                f"(limit {_MAX_AMBIGUITY_DEGREE})"
            )
    return degree
