import unittest

import numpy as np
import orjson

from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,
    dumps_json,
    orjson_response,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestJSONResponseUtils(unittest.TestCase):
    def test_dumps_json_maps_non_finite_values_to_null(self):
        payload = {
            "neg_inf": float("-inf"),
            "pos_inf": float("inf"),
            "nan": float("nan"),
        }
        parsed = orjson.loads(dumps_json(payload))

        self.assertIsNone(parsed["neg_inf"])
        self.assertIsNone(parsed["pos_inf"])
        self.assertIsNone(parsed["nan"])

    def test_dumps_json_supports_numpy_and_non_string_keys(self):
        payload = {
            1: np.array([1, 2, 3], dtype=np.int64),
            "scalar": np.float32(1.5),
        }
        parsed = orjson.loads(dumps_json(payload))

        self.assertEqual(parsed["1"], [1, 2, 3])
        self.assertAlmostEqual(parsed["scalar"], 1.5)

    def test_orjson_response_uses_expected_media_type(self):
        response = orjson_response({"value": float("-inf")}, status_code=201)
        parsed = orjson.loads(response.body)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.media_type, "application/json")
        self.assertIsNone(parsed["value"])

    def test_sglang_orjson_response_serializes_with_shared_options(self):
        response = SGLangORJSONResponse(content={"value": float("-inf")})
        parsed = orjson.loads(response.body)

        self.assertIsNone(parsed["value"])


if __name__ == "__main__":
    unittest.main()
