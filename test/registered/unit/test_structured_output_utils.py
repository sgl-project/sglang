import json
import unittest

from sglang.srt.structured_output_utils import (
    apply_response_format_to_sampling_params,
    response_format_to_json_schema,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestStructuredOutputUtils(unittest.TestCase):
    def test_json_object_response_format(self):
        self.assertEqual(
            response_format_to_json_schema({"type": "json_object"}),
            '{"type": "object"}',
        )

    def test_json_schema_response_format(self):
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        converted = response_format_to_json_schema(
            {
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": schema},
            }
        )
        self.assertEqual(json.loads(converted), schema)

    def test_apply_response_format_does_not_override_existing_constraint(self):
        params = {"json_schema": '{"type":"array"}'}
        self.assertEqual(
            apply_response_format_to_sampling_params(params, {"type": "json_object"}),
            params,
        )

    def test_apply_response_format_adds_json_schema(self):
        params = apply_response_format_to_sampling_params(
            {"temperature": 0},
            {"type": "json_object"},
        )
        self.assertEqual(params["temperature"], 0)
        self.assertEqual(params["json_schema"], '{"type": "object"}')

    def test_apply_response_format_handles_list_of_dicts(self):
        params = apply_response_format_to_sampling_params(
            [{"temperature": 0}, {"temperature": 0.5}],
            {"type": "json_object"},
        )

        self.assertEqual(params[0]["temperature"], 0)
        self.assertEqual(params[0]["json_schema"], '{"type": "object"}')
        self.assertEqual(params[1]["temperature"], 0.5)
        self.assertEqual(params[1]["json_schema"], '{"type": "object"}')

    def test_apply_response_format_list_preserves_existing_constraint(self):
        params = apply_response_format_to_sampling_params(
            [{"temperature": 0}, {"regex": "[a-z]+"}],
            {"type": "json_object"},
        )

        self.assertEqual(params[0]["json_schema"], '{"type": "object"}')
        self.assertEqual(params[1], {"regex": "[a-z]+"})


if __name__ == "__main__":
    unittest.main()
