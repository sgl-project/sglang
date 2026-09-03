"""Detect JSON Schema constraints that grammar backends silently ignore."""

from collections.abc import Iterator
from typing import Any


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

        if "format" in subschema:
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

        groups = _string_constraint_groups(subschema)
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

        groups = _string_constraint_groups(subschema)
        if len(groups) > 1:
            _raise_unsupported(
                "outlines",
                pointer,
                f"constraints {', '.join(groups)} cannot be enforced together",
            )

        if "properties" in subschema:
            ignored_object_constraints = sorted(
                {"additionalProperties", "maxProperties", "minProperties"}.intersection(
                    subschema
                )
            )
            if ignored_object_constraints:
                _raise_unsupported(
                    "outlines",
                    pointer,
                    "properties cannot be combined with "
                    + ", ".join(ignored_object_constraints),
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
