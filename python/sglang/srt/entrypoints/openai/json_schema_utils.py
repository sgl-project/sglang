import copy
from typing import Any, Dict


def _enforce_no_additional_properties(schema: Dict[str, Any]) -> None:
    """Recursively disallow additional properties for object schemas."""
    schema_type = schema.get("type")

    if schema_type == "object":
        schema.setdefault("additionalProperties", False)
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for subschema in properties.values():
                if isinstance(subschema, dict):
                    _enforce_no_additional_properties(subschema)
    elif schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            _enforce_no_additional_properties(items)
        elif isinstance(items, list):
            for subschema in items:
                if isinstance(subschema, dict):
                    _enforce_no_additional_properties(subschema)

    for composite_key in ("anyOf", "allOf", "oneOf"):
        composite = schema.get(composite_key)
        if isinstance(composite, list):
            for subschema in composite:
                if isinstance(subschema, dict):
                    _enforce_no_additional_properties(subschema)

    defs = schema.get("$defs") or schema.get("definitions")
    if isinstance(defs, dict):
        for subschema in defs.values():
            if isinstance(subschema, dict):
                _enforce_no_additional_properties(subschema)


def normalize_json_schema(schema: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    """Return a normalized copy of the schema with stricter defaults when requested."""
    normalized = copy.deepcopy(schema)
    if strict:
        _enforce_no_additional_properties(normalized)
    return normalized
