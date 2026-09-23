"""Detect JSON Schema constraints that grammar backends silently ignore."""

import concurrent.futures
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse


class JSONSchemaDepthExceeded(ValueError):
    """Raised when JSON schema nesting depth exceeds the maximum allowed limit."""

    pass


class JSONSchemaStateExplosion(ValueError):
    """Raised when JSON schema would cause excessive DFA/NFA state explosion."""

    pass


class JSONSchemaCircularRef(ValueError):
    """Raised when JSON schema contains a circular $ref."""

    pass


# Maximum allowed nesting depth for JSON schema (increased from 16 to support valid deeply nested tool definitions)
MAX_SCHEMA_DEPTH = 64
# Maximum allowed total nodes in schema traversal (prevents pathological cases)
MAX_TOTAL_NODES = 50000
# Maximum allowed estimated DFA states (to prevent state explosion)
MAX_DFA_STATES = 10000
# Maximum regex AST nodes before rejecting
MAX_REGEX_AST_NODES = 500
# Maximum nested quantifier depth
MAX_NESTED_QUANTIFIER_DEPTH = 5
# Maximum DFA states after compilation
MAX_COMPILED_DFA_STATES = 10000
# Maximum time for FSM compilation (seconds)
MAX_FSM_COMPILE_TIME = 0.5


def check_regex_ast_complexity(
    pattern: str,
    max_ast_nodes: int = MAX_REGEX_AST_NODES,
    max_nested_quantifiers: int = MAX_NESTED_QUANTIFIER_DEPTH,
) -> None:
    """
    Analyze regex AST for structural complexity that causes DFA state explosion.

    Uses Python's built-in sre_parse to detect:
    - Excessive AST node count
    - Deeply nested quantifiers (e.g., ((a+)+)+)
    - Large bounded repetitions in nested contexts

    Args:
        pattern: The regex pattern string to analyze
        max_ast_nodes: Maximum allowed AST nodes
        max_nested_quantifiers: Maximum allowed quantifiers in a single nesting path

    Raises:
        JSONSchemaStateExplosion: If pattern exceeds complexity budget
    """
    try:
        parsed = sre_parse.parse(pattern)
    except Exception:
        # Invalid regex syntax will be handled downstream by the parser
        return

    if len(parsed) > max_ast_nodes:
        raise JSONSchemaStateExplosion(
            f"Regex AST node count ({len(parsed)}) exceeds budget ({max_ast_nodes})"
        )

    def inspect_subpattern(subpattern, quantifier_depth=0):
        # Track quantifier nesting depth along the current path
        for op, arg in subpattern:
            if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_rep, max_rep, nested = arg
                new_quantifier_depth = quantifier_depth + 1

                if new_quantifier_depth > max_nested_quantifiers:
                    raise JSONSchemaStateExplosion(
                        f"Excessive nested regex quantifiers (depth {new_quantifier_depth} > {max_nested_quantifiers}) in pattern: {pattern[:40]}"
                    )

                # Check for large bounded repetition in nested quantifier context
                # max_rep != MAXREPEAT means it's a bounded quantifier {N} or {N,M}
                # Check if the upper bound is large (> 100) while nested inside another quantifier
                if (
                    max_rep != sre_parse.MAXREPEAT
                    and max_rep > 100
                    and quantifier_depth > 0
                ):
                    raise JSONSchemaStateExplosion(
                        f"Explosive bounded repetition in nested regex: {pattern[:40]}"
                    )

                inspect_subpattern(nested, new_quantifier_depth)
            elif op == sre_parse.SUBPATTERN:
                # SUBPATTERN (grouping) - continue with current quantifier depth
                # arg is (group_id, group_name, flags, subpattern)
                inspect_subpattern(arg[3], quantifier_depth)

    inspect_subpattern(parsed)


def build_fsm_with_budget(
    pattern: str,
    max_states: int = MAX_COMPILED_DFA_STATES,
    timeout_sec: float = MAX_FSM_COMPILE_TIME,
) -> Any:
    """
    Compile regex to FSM with hard timeout and state budget enforcement.

    Tier 1: Fast AST pre-filter (check_regex_ast_complexity)
    Tier 2: Bounded execution with timeout and state count verification

    Args:
        pattern: The regex pattern string to compile
        max_states: Maximum allowed DFA states after compilation
        timeout_sec: Maximum time allowed for compilation

    Returns:
        The compiled FSM object

    Raises:
        JSONSchemaStateExplosion: If timeout exceeded or state budget exceeded
        interegular.patterns.InvalidSyntax: If pattern syntax is invalid
    """
    # Tier 1: Fast AST pre-filter
    check_regex_ast_complexity(pattern)

    # Tier 2: Bounded execution
    import interegular

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: interegular.parse_pattern(pattern).to_fsm())
        try:
            fsm = future.result(timeout=timeout_sec)
            if len(fsm.states) > max_states:
                raise JSONSchemaStateExplosion(
                    f"Compiled DFA states ({len(fsm.states)}) exceeds max limit ({max_states})"
                )
            return fsm
        except concurrent.futures.TimeoutError:
            raise JSONSchemaStateExplosion(
                f"Regex DFA compilation timed out (> {timeout_sec}s), aborting to prevent DoS: {pattern[:40]}"
            )


MAX_SCHEMA_DEPTH = 64
# Maximum allowed total nodes in schema traversal (prevents pathological cases)
MAX_TOTAL_NODES = 50000
# Maximum allowed estimated DFA states (to prevent state explosion)
MAX_DFA_STATES = 10000


@dataclass
class TraversalContext:
    """Thread-local traversal context passed explicitly down the call stack.

    This ensures thread safety by avoiding global or thread-local counters.
    """

    depth: int = 0
    total_nodes: int = 0
    active_ref_stack: list[str] = field(default_factory=list)
    max_depth: int = MAX_SCHEMA_DEPTH
    max_total_nodes: int = MAX_TOTAL_NODES

    def enter(self) -> "TraversalContext":
        """Create a new context for a child node with incremented depth."""
        return TraversalContext(
            depth=self.depth + 1,
            total_nodes=self.total_nodes + 1,
            active_ref_stack=self.active_ref_stack,
            max_depth=self.max_depth,
            max_total_nodes=self.max_total_nodes,
        )

    def check_bounds(self) -> None:
        """Check if traversal bounds are exceeded."""
        if self.depth > self.max_depth:
            raise JSONSchemaDepthExceeded(
                f"JSON schema nesting depth exceeds allowable limit of {self.max_depth}"
            )
        if self.total_nodes > self.max_total_nodes:
            raise JSONSchemaStateExplosion(
                f"JSON schema total nodes ({self.total_nodes}) exceeds maximum allowed ({self.max_total_nodes})"
            )

    def push_ref(self, ref: str) -> None:
        """Push a $ref onto the active reference stack for circular detection."""
        if ref in self.active_ref_stack:
            raise JSONSchemaCircularRef(
                f"Circular $ref detected: {' -> '.join(self.active_ref_stack + [ref])}"
            )
        self.active_ref_stack.append(ref)

    def pop_ref(self) -> None:
        """Pop a $ref from the active reference stack."""
        if self.active_ref_stack:
            self.active_ref_stack.pop()


def validate_schema_depth(
    schema: Any,
    max_depth: int = MAX_SCHEMA_DEPTH,
    max_total_nodes: int = MAX_TOTAL_NODES,
) -> None:
    """
    Validate that JSON schema nesting depth does not exceed the maximum allowed limit.

    Only traverses schema-bearing keywords to avoid false positives from instance data.

    Args:
        schema: The JSON schema to validate
        max_depth: Maximum allowed nesting depth
        max_total_nodes: Maximum allowed total nodes in traversal

    Raises:
        JSONSchemaDepthExceeded: If nesting depth exceeds max_depth
        JSONSchemaStateExplosion: If total nodes exceeds max_total_nodes
        JSONSchemaCircularRef: If a circular $ref is detected
    """
    ctx = TraversalContext(max_depth=max_depth, max_total_nodes=max_total_nodes)
    _validate_schema_depth_recursive(schema, ctx)


def _validate_schema_depth_recursive(schema: Any, ctx: TraversalContext) -> None:
    """Recursive implementation using explicit TraversalContext."""
    ctx.check_bounds()

    if not isinstance(schema, dict):
        return

    # Handle $ref for circular reference detection
    ref = schema.get("$ref")
    if isinstance(ref, str):
        ctx.push_ref(ref)
        try:
            # Note: We don't resolve the ref here, just detect cycles in the ref chain
            # Actual ref resolution happens at compile time
            pass
        finally:
            ctx.pop_ref()

    # Single subschema keywords
    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            _validate_schema_depth_recursive(child, ctx.enter())
        elif keyword == "items" and isinstance(child, list):
            for item in child:
                _validate_schema_depth_recursive(item, ctx.enter())

    # Array subschema keywords
    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                _validate_schema_depth_recursive(child, ctx.enter())

    # Map subschema keywords
    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                _validate_schema_depth_recursive(child, ctx.enter())

    # Dependencies
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for child in dependencies.values():
            if isinstance(child, (bool, dict)):
                _validate_schema_depth_recursive(child, ctx.enter())


def _estimate_dfa_states(schema: Any, depth: int = 0) -> int:
    """
    Estimate the number of DFA states that would be generated from a JSON schema.
    This is a heuristic to catch schemas that would cause state explosion.

    Only traverses schema-bearing keywords to match validation behavior.

    Args:
        schema: The JSON schema to estimate
        depth: Current nesting depth

    Returns:
        Estimated number of DFA states
    """
    if not isinstance(schema, dict):
        return 1

    state_count = 1  # Base state for this schema level

    # Each property adds states
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        state_count += len(properties) * 2  # Property name + value states

    # Handle string patterns - regex patterns can cause exponential state explosion
    # We estimate based on pattern complexity (rough heuristic)
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        # Estimate regex complexity: count of quantifiers, alternations, groups
        # A pattern like "[ab]*a[ab]{13}" can generate ~16K states
        import re

        # Match quantifiers properly: *, +, ?, {N}, {N,}, {N,M}
        # Use raw strings for proper regex escaping
        quantifier_count = len(re.findall(r"[*+?]|\{\d+,?\d*\}", pattern))
        alt_count = pattern.count("|")
        group_count = pattern.count("(")
        char_class_count = pattern.count("[")
        # Look for {N} quantifiers with large N
        brace_quantifiers = re.findall(r"\{(\d+)(?:,\d*)?\}", pattern)
        large_quantifier_sum = sum(int(n) for n in brace_quantifiers if int(n) > 10)
        # Heuristic: each quantifier/alternation/group roughly multiplies states
        # For patterns like [ab]*a[ab]{13}, the combination of * and {N} is problematic
        # Use exponential estimation for bounded quantifiers combined with unbounded
        has_unbounded = "*" in pattern or "+" in pattern
        has_bounded = bool(brace_quantifiers)
        if has_unbounded and has_bounded:
            # This combination can cause exponential blowup
            # Multiply by the product of bounded quantifiers
            bounded_product = 1
            for n in brace_quantifiers:
                bounded_product *= max(1, int(n))
            # Cap at reasonable value
            bounded_product = min(bounded_product, 10000)
            pattern_complexity = max(
                1,
                (quantifier_count + alt_count + group_count + char_class_count) * 100
                + bounded_product * 100,
            )
        else:
            pattern_complexity = max(
                1,
                (quantifier_count + alt_count + group_count + char_class_count) * 100
                + large_quantifier_sum * 500,
            )
        state_count += pattern_complexity

    # Single subschema keywords
    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            state_count += _estimate_dfa_states(child, depth + 1)
        elif keyword == "items" and isinstance(child, list):
            for item in child:
                state_count += _estimate_dfa_states(item, depth + 1)

    # Array subschema keywords
    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                state_count += _estimate_dfa_states(child, depth + 1)

    # Map subschema keywords
    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                state_count += _estimate_dfa_states(child, depth + 1)

    # Dependencies
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for child in dependencies.values():
            if isinstance(child, (bool, dict)):
                state_count += _estimate_dfa_states(child, depth + 1)

    return state_count


def validate_schema_bounds(
    schema: Any,
    max_depth: int = MAX_SCHEMA_DEPTH,
    max_states: int = MAX_DFA_STATES,
    max_total_nodes: int = MAX_TOTAL_NODES,
) -> None:
    """
    Validate JSON schema bounds to prevent DFA state explosion and CPU thread hanging.

    This should be called before compiling a JSON schema to a grammar automaton.

    Args:
        schema: The JSON schema to validate
        max_depth: Maximum allowed nesting depth (default: 64)
        max_states: Maximum allowed estimated DFA states (default: 10000)
        max_total_nodes: Maximum allowed total nodes in traversal (default: 50000)

    Raises:
        JSONSchemaDepthExceeded: If nesting depth exceeds max_depth
        JSONSchemaStateExplosion: If estimated DFA states exceeds max_states or total nodes exceeds max_total_nodes
        JSONSchemaCircularRef: If a circular $ref is detected
    """
    # First validate depth and total nodes (also detects circular refs)
    validate_schema_depth(schema, max_depth=max_depth, max_total_nodes=max_total_nodes)

    # Then estimate DFA states
    estimated_states = _estimate_dfa_states(schema)
    if estimated_states > max_states:
        raise JSONSchemaStateExplosion(
            f"JSON schema estimated DFA states ({estimated_states}) exceeds "
            f"maximum allowed ({max_states}). Schema may cause DoS via state explosion."
        )


class UnsupportedJSONSchemaFeature(ValueError):
    """A schema uses constraints that the selected backend cannot preserve."""


_SINGLE_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)

_XGRAMMAR_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "maxContains",
        "minContains",
        "multipleOf",
        "not",
        "then",
        "uniqueItems",
    }
)
_XGRAMMAR_STRING_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "json-pointer",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
        "uri-template",
        "uuid",
    }
)

# Outlines Core 0.1.x accepts schemas containing these assertion keywords but
# does not encode them in the generated regex.
_OUTLINES_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "allOf",
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "if",
        "maxContains",
        "maximum",
        "minContains",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "patternProperties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
)


def _escape_json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _iter_subschemas(
    schema: Any, pointer: str = "#"
) -> Iterator[tuple[dict[str, Any], str]]:
    """Walk schema-bearing keywords without inspecting instance-valued data."""
    if not isinstance(schema, dict):
        return

    yield schema, pointer

    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            yield from _iter_subschemas(child, f"{pointer}/{keyword}")
        elif keyword == "items" and isinstance(child, list):
            for index, item in enumerate(child):
                yield from _iter_subschemas(item, f"{pointer}/{keyword}/{index}")

    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                yield from _iter_subschemas(child, f"{pointer}/{keyword}/{index}")

    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for name, child in children.items():
                escaped_name = _escape_json_pointer(name)
                yield from _iter_subschemas(
                    child, f"{pointer}/{keyword}/{escaped_name}"
                )

    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for name, child in dependencies.items():
            if isinstance(child, (bool, dict)):
                escaped_name = _escape_json_pointer(name)
                yield from _iter_subschemas(
                    child, f"{pointer}/dependencies/{escaped_name}"
                )


def _raise_unsupported(backend: str, pointer: str, reason: str) -> None:
    raise UnsupportedJSONSchemaFeature(
        f"JSON Schema at {pointer} is not supported by {backend}: {reason}"
    )


def _string_constraint_groups(schema: dict[str, Any]) -> list[str]:
    groups = []
    if "format" in schema:
        groups.append("format")
    if "pattern" in schema:
        groups.append("pattern")
    if "minLength" in schema or "maxLength" in schema:
        groups.append("minLength/maxLength")
    return groups


def _can_describe_string(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    return (
        schema_type is None
        or schema_type == "string"
        or (isinstance(schema_type, list) and "string" in schema_type)
    )


def validate_xgrammar_json_schema(schema: Any) -> None:
    """Reject constraints XGrammar 0.2.x accepts without fully enforcing."""
    for subschema, pointer in _iter_subschemas(schema):
        unsupported = sorted(_XGRAMMAR_UNSUPPORTED_KEYWORDS.intersection(subschema))
        if unsupported:
            _raise_unsupported(
                "xgrammar",
                pointer,
                f"keyword(s) {', '.join(unsupported)} would be ignored",
            )

        if _can_describe_string(subschema) and "format" in subschema:
            format_name = subschema["format"]
            if (
                not isinstance(format_name, str)
                or format_name not in _XGRAMMAR_STRING_FORMATS
            ):
                _raise_unsupported(
                    "xgrammar",
                    pointer,
                    f"string format {format_name!r} is not implemented",
                )

        groups = (
            _string_constraint_groups(subschema)
            if _can_describe_string(subschema)
            else []
        )
        if len(groups) > 1:
            _raise_unsupported(
                "xgrammar",
                pointer,
                f"constraints {', '.join(groups)} cannot be enforced together",
            )


def validate_outlines_json_schema(schema: Any) -> None:
    """Reject constraints Outlines Core 0.1.x accepts without preserving."""
    for subschema, pointer in _iter_subschemas(schema):
        unsupported = sorted(_OUTLINES_UNSUPPORTED_KEYWORDS.intersection(subschema))
        if unsupported:
            _raise_unsupported(
                "outlines",
                pointer,
                f"keyword(s) {', '.join(unsupported)} would be ignored or weakened",
            )

        groups = (
            _string_constraint_groups(subschema)
            if _can_describe_string(subschema)
            else []
        )
        if len(groups) > 1:
            _raise_unsupported(
                "outlines",
                pointer,
                f"constraints {', '.join(groups)} cannot be enforced together",
            )

        if "properties" in subschema:
            ignored_object_constraints = {
                "additionalProperties",
                "maxProperties",
                "minProperties",
            }.intersection(subschema)
            if subschema.get("additionalProperties") is False:
                # The properties regex already excludes undeclared keys.
                ignored_object_constraints.discard("additionalProperties")
            if ignored_object_constraints:
                _raise_unsupported(
                    "outlines",
                    pointer,
                    "properties cannot be combined with "
                    + ", ".join(sorted(ignored_object_constraints)),
                )

        if "prefixItems" in subschema:
            ignored_array_constraints = sorted(
                {"maxItems", "minItems"}.intersection(subschema)
            )
            if ignored_array_constraints:
                _raise_unsupported(
                    "outlines",
                    pointer,
                    "prefixItems cannot be combined with "
                    + ", ".join(ignored_array_constraints),
                )

        if "required" in subschema and "properties" not in subschema:
            _raise_unsupported(
                "outlines",
                pointer,
                "required is only enforced when properties is present",
            )
