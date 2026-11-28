"""Detect JSON Schema constraints that grammar backends silently ignore."""

import contextvars
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

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

_FSM_STATE_BUDGET: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "sglang_fsm_state_budget", default=None
)
_INTEREGULAR_CRAWL_INSTALL_LOCK = threading.Lock()


def _install_interegular_state_budget(interegular_fsm: Any) -> None:
    """Add a context-local state cap to interegular's determinization walk."""
    if getattr(interegular_fsm.crawl, "_sglang_budgeted", False):
        return

    with _INTEREGULAR_CRAWL_INSTALL_LOCK:
        if getattr(interegular_fsm.crawl, "_sglang_budgeted", False):
            return

        original_crawl = interegular_fsm.crawl

        def budgeted_crawl(alphabet, initial, final, follow):
            max_states = _FSM_STATE_BUDGET.get()
            if max_states is None:
                return original_crawl(alphabet, initial, final, follow)

            if max_states < 1:
                raise JSONSchemaStateExplosion(
                    f"FSM construction exceeded state budget ({max_states})"
                )

            states = [initial]
            state_indices = {_fsm_state_key(initial): 0}
            finals = set()
            transitions = {}
            index = 0

            while index < len(states):
                state = states[index]
                if final(state):
                    finals.add(index)

                state_transitions = {}
                for transition in alphabet.by_transition:
                    try:
                        next_state = follow(state, transition)
                    except interegular_fsm.OblivionError:
                        continue

                    key = _fsm_state_key(next_state)
                    next_index = state_indices.get(key)
                    if next_index is None:
                        if len(states) >= max_states:
                            raise JSONSchemaStateExplosion(
                                "FSM construction exceeded state budget "
                                f"({max_states}) before determinization completed"
                            )
                        next_index = len(states)
                        state_indices[key] = next_index
                        states.append(next_state)

                    state_transitions[transition] = next_index

                transitions[index] = state_transitions
                index += 1

            return interegular_fsm.FSM(
                alphabet=alphabet,
                states=range(len(states)),
                initial=0,
                finals=finals,
                map=transitions,
                __no_validation__=True,
            )

        budgeted_crawl._sglang_budgeted = True
        interegular_fsm.crawl = budgeted_crawl


def _fsm_state_key(state: Any) -> Any:
    """Make interegular's composite crawl states usable as dictionary keys."""
    if isinstance(state, dict):
        return (
            dict,
            frozenset((_fsm_state_key(k), _fsm_state_key(v)) for k, v in state.items()),
        )
    if isinstance(state, (set, frozenset)):
        return (set, frozenset(_fsm_state_key(value) for value in state))
    if isinstance(state, list):
        return (list, tuple(_fsm_state_key(value) for value in state))
    if isinstance(state, tuple):
        return (tuple, tuple(_fsm_state_key(value) for value in state))
    return (type(state), state)


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

    # Let known linear patterns reach the bounded FSM builder, even when they
    # contain more AST nodes than this structural pre-filter allows.
    if len(parsed) > max_ast_nodes and _estimate_fixed_regex_states(pattern) is None:
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
    Compile regex to FSM with state budget enforcement.

    Tier 1: Fast AST pre-filter (check_regex_ast_complexity)
    Tier 2: Determinization with a state cap and final state-count verification

    Args:
        pattern: The regex pattern string to compile
        max_states: Maximum allowed DFA states after compilation
        timeout_sec: Maximum time allowed for compilation (reserved for future use)

    Returns:
        The compiled FSM object

    Raises:
        JSONSchemaStateExplosion: If state budget exceeded
        ImportError: If interegular is not installed
    """
    # Tier 1: Fast AST pre-filter
    check_regex_ast_complexity(pattern)

    # Interegular's determinization walk is capped so explosive patterns stop
    # before allocating the full DFA. Keep the final check as a defensive guard.
    try:
        import interegular
        import interegular.fsm as interegular_fsm
    except ImportError as e:
        raise ImportError(
            "interegular is required for Outlines regex compilation but not installed. "
            "Install it with: pip install interegular"
        ) from e

    _install_interegular_state_budget(interegular_fsm)
    budget_token = _FSM_STATE_BUDGET.set(max_states)
    try:
        fsm = interegular.parse_pattern(pattern).to_fsm()
    finally:
        _FSM_STATE_BUDGET.reset(budget_token)

    if len(fsm.states) > max_states:
        raise JSONSchemaStateExplosion(
            f"Compiled DFA states ({len(fsm.states)}) exceeds max limit ({max_states})"
        )
    return fsm


MAX_SCHEMA_DEPTH = 64
# Maximum allowed total nodes in schema traversal (prevents pathological cases)
MAX_TOTAL_NODES = 50000
# Maximum allowed estimated DFA states (to prevent state explosion)
MAX_DFA_STATES = 10000


@dataclass
class TraversalContext:
    """Bound schema traversal depth and total expanded nodes."""

    depth: int = 0
    total_nodes: int = 1
    active_ref_stack: list[str] = field(default_factory=list)
    max_depth: int = MAX_SCHEMA_DEPTH
    max_total_nodes: int = MAX_TOTAL_NODES
    _node_count: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._node_count:
            self._node_count.append(self.total_nodes)

    def enter(self) -> "TraversalContext":
        """Create a child context, sharing the traversal-wide node budget."""
        self._node_count[0] += 1
        return TraversalContext(
            depth=self.depth + 1,
            total_nodes=self._node_count[0],
            active_ref_stack=self.active_ref_stack,
            max_depth=self.max_depth,
            max_total_nodes=self.max_total_nodes,
            _node_count=self._node_count,
        )

    def check_bounds(self) -> None:
        """Check if traversal bounds are exceeded."""
        if self.depth > self.max_depth:
            raise JSONSchemaDepthExceeded(
                f"JSON schema nesting depth exceeds allowable limit of {self.max_depth}"
            )
        if self._node_count[0] > self.max_total_nodes:
            raise JSONSchemaStateExplosion(
                f"JSON schema total nodes ({self._node_count[0]}) exceeds maximum allowed ({self.max_total_nodes})"
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


def _resolve_schema_ref(root_schema: Any, ref: str) -> tuple[Any, str]:
    """Resolve a local JSON Pointer reference and return its canonical pointer."""
    if not ref.startswith("#"):
        raise ValueError(f"Only local JSON Schema references are supported: {ref!r}")

    fragment = unquote(ref[1:])
    if not fragment:
        return root_schema, "#"
    if not fragment.startswith("/"):
        raise ValueError(f"Only JSON Pointer references are supported: {ref!r}")

    tokens = []
    for encoded_token in fragment[1:].split("/"):
        token_chars = []
        index = 0
        while index < len(encoded_token):
            char = encoded_token[index]
            if char != "~":
                token_chars.append(char)
                index += 1
                continue
            if index + 1 >= len(encoded_token) or encoded_token[index + 1] not in "01":
                raise ValueError(f"Invalid JSON Pointer escape in reference: {ref!r}")
            token_chars.append("~" if encoded_token[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(token_chars))

    target = root_schema
    for token in tokens:
        if isinstance(target, dict) and token in target:
            target = target[token]
        elif isinstance(target, list) and token.isdigit() and str(int(token)) == token:
            array_index = int(token)
            if array_index >= len(target):
                raise ValueError(f"Unresolved JSON Schema reference: {ref!r}")
            target = target[array_index]
        else:
            raise ValueError(f"Unresolved JSON Schema reference: {ref!r}")

    canonical_ref = "#/" + "/".join(_escape_json_pointer(token) for token in tokens)
    return target, canonical_ref


def validate_schema_depth(
    schema: Any,
    max_depth: int = MAX_SCHEMA_DEPTH,
    max_total_nodes: int = MAX_TOTAL_NODES,
) -> None:
    """Validate depth and total nodes in the schema after resolving local refs."""
    ctx = TraversalContext(max_depth=max_depth, max_total_nodes=max_total_nodes)
    _validate_schema_depth_recursive(schema, ctx, schema)


def _validate_schema_depth_recursive(
    schema: Any, ctx: TraversalContext, root_schema: Any
) -> None:
    """Recursively validate embedded schemas and their referenced targets."""
    ctx.check_bounds()

    if not isinstance(schema, dict):
        return

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target, canonical_ref = _resolve_schema_ref(root_schema, ref)
        ctx.push_ref(canonical_ref)
        try:
            _validate_schema_depth_recursive(target, ctx.enter(), root_schema)
        finally:
            ctx.pop_ref()

    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            _validate_schema_depth_recursive(child, ctx.enter(), root_schema)
        elif keyword == "items" and isinstance(child, list):
            for item in child:
                _validate_schema_depth_recursive(item, ctx.enter(), root_schema)

    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                _validate_schema_depth_recursive(child, ctx.enter(), root_schema)

    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                _validate_schema_depth_recursive(child, ctx.enter(), root_schema)

    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for child in dependencies.values():
            if isinstance(child, (bool, dict)):
                _validate_schema_depth_recursive(child, ctx.enter(), root_schema)


def _estimate_fixed_regex_states(pattern: str) -> int | None:
    """Estimate linear states for fixed-width patterns and bounded character atoms."""
    try:
        parsed = sre_parse.parse(pattern)
    except Exception:
        # Leave unsupported syntax to the backend compiler.
        return None

    if not parsed or parsed.state.flags & sre_parse.SRE_FLAG_MULTILINE:
        return None

    character_atom_ops = (
        sre_parse.LITERAL,
        sre_parse.NOT_LITERAL,
        sre_parse.IN,
        sre_parse.ANY,
        sre_parse.CATEGORY,
    )

    def is_single_character_atom(subpattern):
        return len(subpattern) == 1 and subpattern[0][0] in character_atom_ops

    is_start_anchored = parsed[0] in (
        (sre_parse.AT, sre_parse.AT_BEGINNING),
        (sre_parse.AT, sre_parse.AT_BEGINNING_STRING),
    )
    # A standalone bounded repetition of one character atom needs only states
    # for the current run length, even without anchors. Other unanchored
    # expressions can need states for many candidate match positions.
    body = list(parsed)
    if is_start_anchored:
        body.pop(0)
    if body and body[-1] in (
        (sre_parse.AT, sre_parse.AT_END),
        (sre_parse.AT, sre_parse.AT_END_STRING),
    ):
        body.pop()
    if (
        not is_start_anchored
        and body
        and all(op == sre_parse.LITERAL for op, _ in body)
    ):
        return len(body) + 1

    if len(body) == 1 and body[0][0] in (
        sre_parse.MAX_REPEAT,
        sre_parse.MIN_REPEAT,
    ):
        _, max_rep, child = body[0][1]
        if max_rep != sre_parse.MAXREPEAT and is_single_character_atom(child):
            return max_rep + 1

    if not is_start_anchored:
        return None

    def fixed_width(subpattern):
        width = 0
        for op, arg in subpattern:
            if op in character_atom_ops:
                width += 1
            elif op == sre_parse.AT:
                continue
            elif op == sre_parse.SUBPATTERN:
                child_width = fixed_width(arg[3])
                if child_width is None:
                    return None
                width += child_width
            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_rep, max_rep, child = arg
                if min_rep != max_rep or max_rep == sre_parse.MAXREPEAT:
                    return None
                child_width = fixed_width(child)
                if child_width is None:
                    return None
                width += max_rep * child_width
            else:
                return None
        return width

    def char_set(op, arg):
        if op == sre_parse.LITERAL:
            return ((arg, arg),)
        if op != sre_parse.IN:
            return None
        intervals = []
        for class_op, class_arg in arg:
            if class_op == sre_parse.LITERAL:
                intervals.append((class_arg, class_arg))
            elif class_op == sre_parse.RANGE:
                intervals.append(class_arg)
            else:
                return None
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return tuple(merged)

    def sets_intersect(left, right):
        if left is None or right is None:
            return True
        return any(
            left_start <= right_end and right_start <= left_end
            for left_start, left_end in left
            for right_start, right_end in right
        )

    def fixed_description(subpattern):
        width = 0
        first_set = last_set = None
        for op, arg in subpattern:
            if op in character_atom_ops:
                part_width = 1
                part_first = part_last = char_set(op, arg)
            elif op == sre_parse.AT:
                continue
            elif op == sre_parse.SUBPATTERN:
                description = fixed_description(arg[3])
                if description is None:
                    return None
                part_width, part_first, part_last = description
            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_rep, max_rep, child = arg
                if min_rep != max_rep or max_rep == sre_parse.MAXREPEAT:
                    return None
                description = fixed_description(child)
                if description is None:
                    return None
                child_width, child_first, child_last = description
                part_width = max_rep * child_width
                part_first = child_first if part_width else None
                part_last = child_last if part_width else None
            else:
                return None

            if part_width:
                if width == 0:
                    first_set = part_first
                last_set = part_last
                width += part_width
        return width, first_set, last_set

    def bounded_composition_segments(subpattern):
        segments = []
        for op, arg in subpattern:
            if op in character_atom_ops:
                charset = char_set(op, arg)
                segments.append(("fixed", 1, charset, charset))
            elif op == sre_parse.AT:
                continue
            elif op == sre_parse.SUBPATTERN:
                nested = bounded_composition_segments(arg[3])
                if nested is None:
                    return None
                segments.extend(nested)
            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_rep, max_rep, child = arg
                if min_rep == max_rep and max_rep != sre_parse.MAXREPEAT:
                    description = fixed_description(child)
                    if description is None:
                        return None
                    child_width, child_first, child_last = description
                    width = max_rep * child_width
                    if width:
                        segments.append(("fixed", width, child_first, child_last))
                elif (
                    max_rep != sre_parse.MAXREPEAT
                    and len(child) == 1
                    and child[0][0] in character_atom_ops
                ):
                    repeated_set = char_set(child[0][0], child[0][1])
                    if repeated_set is None:
                        return None
                    if max_rep:
                        segments.append(
                            ("variable", max_rep, repeated_set, repeated_set)
                        )
                else:
                    return None
            else:
                return None

        merged = []
        for segment in segments:
            if segment[0] == "fixed" and merged and merged[-1][0] == "fixed":
                previous = merged[-1]
                merged[-1] = (
                    "fixed",
                    previous[1] + segment[1],
                    previous[2],
                    segment[3],
                )
            else:
                merged.append(segment)
        return merged

    def bounded_composition_states(subpattern):
        segments = bounded_composition_segments(subpattern)
        if segments is None:
            return None

        for index, segment in enumerate(segments):
            if segment[0] != "variable":
                continue
            if index and sets_intersect(segment[2], segments[index - 1][3]):
                return None
            if index + 1 < len(segments) and sets_intersect(
                segment[2], segments[index + 1][2]
            ):
                return None

        return 1 + sum(segment[1] for segment in segments)

    if body and body[-1][0] in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
        min_rep, max_rep, child = body[-1][1]
        if (
            min_rep != max_rep
            and max_rep != sre_parse.MAXREPEAT
            and is_single_character_atom(child)
        ):
            prefix_width = fixed_width(body[:-1])
            if prefix_width is not None:
                return prefix_width + max_rep + 1

    composition_states = bounded_composition_states(body)
    if composition_states is not None:
        return composition_states

    width = fixed_width(parsed)
    return None if width is None else width + 1


def _estimate_dfa_states(
    schema: Any,
    depth: int = 0,
    *,
    root_schema: Any = None,
    max_states: int = MAX_DFA_STATES,
    max_depth: int = MAX_SCHEMA_DEPTH,
    max_total_nodes: int = MAX_TOTAL_NODES,
    ctx: TraversalContext | None = None,
) -> int:
    """Estimate DFA states while bounding the expanded schema, including refs."""
    if root_schema is None:
        root_schema = schema
    if ctx is None:
        ctx = TraversalContext(
            depth=depth, max_depth=max_depth, max_total_nodes=max_total_nodes
        )
    ctx.check_bounds()

    state_count = 1

    def add_states(additional_states: int) -> None:
        nonlocal state_count
        state_count += additional_states
        if state_count > max_states:
            raise JSONSchemaStateExplosion(
                f"JSON schema estimated DFA states ({state_count}) exceeds "
                f"maximum allowed ({max_states}). Schema may cause DoS via state explosion."
            )

    if not isinstance(schema, dict):
        return state_count

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target, canonical_ref = _resolve_schema_ref(root_schema, ref)
        ctx.push_ref(canonical_ref)
        try:
            add_states(
                _estimate_dfa_states(
                    target,
                    depth + 1,
                    root_schema=root_schema,
                    max_states=max_states,
                    max_depth=max_depth,
                    max_total_nodes=max_total_nodes,
                    ctx=ctx.enter(),
                )
            )
        finally:
            ctx.pop_ref()

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        add_states(len(properties) * 2)

    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        import re

        quantifier_count = len(re.findall(r"[*+?]|\{\d+,?\d*\}", pattern))
        alt_count = pattern.count("|")
        group_count = pattern.count("(")
        char_class_count = pattern.count("[")
        brace_quantifiers = re.findall(r"\{(\d+)(?:,\d*)?\}", pattern)
        large_quantifier_sum = sum(int(n) for n in brace_quantifiers if int(n) > 10)
        has_unbounded = "*" in pattern or "+" in pattern
        has_bounded = bool(brace_quantifiers)
        nested_quantifier_pattern = r"\([^)]*[\+\*\?\{][^)]*\)[\+\*\?\{]"
        has_nested_quantifiers = bool(re.search(nested_quantifier_pattern, pattern))
        linear_states = _estimate_fixed_regex_states(pattern)
        if linear_states is not None:
            pattern_complexity = linear_states
        elif has_unbounded and has_bounded and has_nested_quantifiers:
            bounded_product = 1
            for n in brace_quantifiers:
                bounded_product *= max(1, int(n))
            bounded_product = min(bounded_product, 10000)
            pattern_complexity = max(
                1,
                (quantifier_count + alt_count + group_count + char_class_count) * 100
                + bounded_product * 100,
            )
        elif has_unbounded and has_bounded:
            bounded_sum = sum(max(1, int(n)) for n in brace_quantifiers)
            pattern_complexity = max(
                1,
                (quantifier_count + alt_count + group_count + char_class_count) * 100
                + bounded_sum * 500,
            )
        else:
            pattern_complexity = max(
                1,
                (quantifier_count + alt_count + group_count + char_class_count) * 100
                + large_quantifier_sum * 500,
            )
        add_states(pattern_complexity)

    format_name = schema.get("format")
    if isinstance(format_name, str) and format_name in _XGRAMMAR_STRING_FORMATS:
        add_states(3)

    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            add_states(
                _estimate_dfa_states(
                    child,
                    depth + 1,
                    root_schema=root_schema,
                    max_states=max_states,
                    max_depth=max_depth,
                    max_total_nodes=max_total_nodes,
                    ctx=ctx.enter(),
                )
            )
        elif keyword == "items" and isinstance(child, list):
            for item in child:
                add_states(
                    _estimate_dfa_states(
                        item,
                        depth + 1,
                        root_schema=root_schema,
                        max_states=max_states,
                        max_depth=max_depth,
                        max_total_nodes=max_total_nodes,
                        ctx=ctx.enter(),
                    )
                )

    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                add_states(
                    _estimate_dfa_states(
                        child,
                        depth + 1,
                        root_schema=root_schema,
                        max_states=max_states,
                        max_depth=max_depth,
                        max_total_nodes=max_total_nodes,
                        ctx=ctx.enter(),
                    )
                )

    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                add_states(
                    _estimate_dfa_states(
                        child,
                        depth + 1,
                        root_schema=root_schema,
                        max_states=max_states,
                        max_depth=max_depth,
                        max_total_nodes=max_total_nodes,
                        ctx=ctx.enter(),
                    )
                )

    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for child in dependencies.values():
            if isinstance(child, (bool, dict)):
                add_states(
                    _estimate_dfa_states(
                        child,
                        depth + 1,
                        root_schema=root_schema,
                        max_states=max_states,
                        max_depth=max_depth,
                        max_total_nodes=max_total_nodes,
                        ctx=ctx.enter(),
                    )
                )

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
    estimated_states = _estimate_dfa_states(
        schema,
        max_states=max_states,
        max_depth=max_depth,
        max_total_nodes=max_total_nodes,
    )
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
    schema: Any,
    pointer: str = "#",
    *,
    root_schema: Any = None,
    active_refs: list[str] | None = None,
    resolve_refs: bool = False,
) -> Iterator[tuple[dict[str, Any], str]]:
    """Walk schema-bearing keywords, optionally following local references."""
    if root_schema is None:
        root_schema = schema
    if active_refs is None:
        active_refs = []
    if not isinstance(schema, dict):
        return

    yield schema, pointer

    ref = schema.get("$ref")
    if resolve_refs and isinstance(ref, str):
        target, canonical_ref = _resolve_schema_ref(root_schema, ref)
        if canonical_ref in active_refs:
            raise JSONSchemaCircularRef(
                f"Circular $ref detected: {' -> '.join(active_refs + [canonical_ref])}"
            )
        yield from _iter_subschemas(
            target,
            f"{pointer}/$ref",
            root_schema=root_schema,
            active_refs=active_refs + [canonical_ref],
            resolve_refs=resolve_refs,
        )

    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, (bool, dict)):
            yield from _iter_subschemas(
                child,
                f"{pointer}/{keyword}",
                root_schema=root_schema,
                active_refs=active_refs,
                resolve_refs=resolve_refs,
            )
        elif keyword == "items" and isinstance(child, list):
            for index, item in enumerate(child):
                yield from _iter_subschemas(
                    item,
                    f"{pointer}/{keyword}/{index}",
                    root_schema=root_schema,
                    active_refs=active_refs,
                    resolve_refs=resolve_refs,
                )

    for keyword in _SUBSCHEMA_ARRAY_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for index, child in enumerate(children):
                yield from _iter_subschemas(
                    child,
                    f"{pointer}/{keyword}/{index}",
                    root_schema=root_schema,
                    active_refs=active_refs,
                    resolve_refs=resolve_refs,
                )

    for keyword in _SUBSCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            for name, child in children.items():
                escaped_name = _escape_json_pointer(name)
                yield from _iter_subschemas(
                    child,
                    f"{pointer}/{keyword}/{escaped_name}",
                    root_schema=root_schema,
                    active_refs=active_refs,
                    resolve_refs=resolve_refs,
                )

    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for name, child in dependencies.items():
            if isinstance(child, (bool, dict)):
                escaped_name = _escape_json_pointer(name)
                yield from _iter_subschemas(
                    child,
                    f"{pointer}/dependencies/{escaped_name}",
                    root_schema=root_schema,
                    active_refs=active_refs,
                    resolve_refs=resolve_refs,
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
    """Reject unsupported constraints and over-budget expanded Outlines schemas."""
    validate_schema_bounds(schema)
    for subschema, pointer in _iter_subschemas(
        schema, root_schema=schema, resolve_refs=True
    ):
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
