import json
import sys

import pytest
from pydantic import BaseModel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.utils import convert_json_schema_to_str

register_cpu_ci(1.0, "base-a-test-cpu")


class Schema(BaseModel):
    name: str


@pytest.mark.parametrize("value", [None, 42, [], True, Schema(name="test"), int])
def test_invalid_values_raise_value_error(value):
    with pytest.raises(ValueError, match="Cannot parse schema"):
        convert_json_schema_to_str(value)


def test_supported_inputs():
    schema = Schema.model_json_schema()
    assert json.loads(convert_json_schema_to_str(Schema)) == schema
    assert json.loads(convert_json_schema_to_str(schema)) == schema
    schema_str = json.dumps(schema)
    assert convert_json_schema_to_str(schema_str) == schema_str


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
