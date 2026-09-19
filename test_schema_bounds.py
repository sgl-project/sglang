"""Test script to validate JSON schema depth and state bounds checking."""

import json

from sglang.srt.constrained.json_schema_validation import (
    MAX_DFA_STATES,
    MAX_SCHEMA_DEPTH,
    JSONSchemaDepthExceeded,
    JSONSchemaStateExplosion,
    validate_schema_bounds,
    validate_schema_depth,
)


def test_valid_schema():
    """Test that a normal schema passes validation."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    validate_schema_bounds(schema)
    print("✓ Valid schema passed validation")


def test_excessive_depth():
    """Test that deeply nested schema raises JSONSchemaDepthExceeded."""
    # Create a schema nested deeper than MAX_SCHEMA_DEPTH (16)
    schema = {"type": "object"}
    current = schema
    for i in range(MAX_SCHEMA_DEPTH + 2):
        current["properties"] = {"nested": {"type": "object"}}
        current = current["properties"]["nested"]

    try:
        validate_schema_bounds(schema)
        print("✗ FAIL: Should have raised JSONSchemaDepthExceeded")
    except JSONSchemaDepthExceeded as e:
        print(f"✓ Correctly caught depth exceeded: {e}")
    except Exception as e:
        print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")


def test_excessive_states():
    """Test that schema with many properties raises JSONSchemaStateExplosion."""
    # Create a schema with many properties to exceed MAX_DFA_STATES
    properties = {}
    for i in range(6000):  # Each property adds ~2 states, so 6000 > 10000
        properties[f"prop_{i}"] = {"type": "string"}

    schema = {
        "type": "object",
        "properties": properties,
    }

    try:
        validate_schema_bounds(schema)
        print("✗ FAIL: Should have raised JSONSchemaStateExplosion")
    except JSONSchemaStateExplosion as e:
        print(f"✓ Correctly caught state explosion: {e}")
    except Exception as e:
        print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")


def test_allOf_explosion():
    """Test that many allOf clauses cause state explosion."""
    all_of = []
    for i in range(3000):  # Each allOf adds states
        all_of.append({"type": "object", "properties": {f"x{i}": {"type": "string"}}})

    schema = {
        "type": "object",
        "allOf": all_of,
    }

    try:
        validate_schema_bounds(schema)
        print("✗ FAIL: Should have raised JSONSchemaStateExplosion")
    except JSONSchemaStateExplosion as e:
        print(f"✓ Correctly caught allOf state explosion: {e}")
    except Exception as e:
        print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")


def test_depth_16_boundary():
    """Test that exactly MAX_SCHEMA_DEPTH passes validation."""
    schema = {"type": "object"}
    current = schema
    for i in range(MAX_SCHEMA_DEPTH):
        current["properties"] = {"nested": {"type": "object"}}
        current = current["properties"]["nested"]

    try:
        validate_schema_bounds(schema)
        print("✓ Depth exactly at MAX_SCHEMA_DEPTH passes")
    except Exception as e:
        print(f"✗ FAIL: Should pass at boundary: {type(e).__name__}: {e}")


def test_depth_17_exceeds():
    """Test that MAX_SCHEMA_DEPTH + 1 fails validation."""
    schema = {"type": "object"}
    current = schema
    for i in range(MAX_SCHEMA_DEPTH + 1):
        current["properties"] = {"nested": {"type": "object"}}
        current = current["properties"]["nested"]

    try:
        validate_schema_bounds(schema)
        print("✗ FAIL: Should have raised JSONSchemaDepthExceeded at depth 17")
    except JSONSchemaDepthExceeded:
        print("✓ Depth 17 correctly rejected")
    except Exception as e:
        print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")


def test_validate_schema_depth_directly():
    """Test the validate_schema_depth function directly."""
    # Schema with 2 levels of nesting: root -> properties -> nested -> properties -> nested
    schema = {"type": "object"}
    current = schema
    for i in range(2):
        current["properties"] = {"nested": {"type": "object"}}
        current = current["properties"]["nested"]

    validate_schema_depth(schema, max_depth=2)
    print("✓ validate_schema_depth works at depth 2")

    try:
        validate_schema_depth(schema, max_depth=1)
        print("✗ FAIL: Should have raised at depth 1")
    except JSONSchemaDepthExceeded:
        print("✓ validate_schema_depth correctly rejects depth 1")


if __name__ == "__main__":
    print("Running JSON schema bounds validation tests...\n")
    test_valid_schema()
    test_excessive_depth()
    test_excessive_states()
    test_allOf_explosion()
    test_depth_16_boundary()
    test_depth_17_exceeds()
    test_validate_schema_depth_directly()
    print("\nAll tests completed!")
