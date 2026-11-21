"""Unit tests for RuntimeEndpoint._handle_dtype_to_regex."""

import unittest

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.ir import REGEX_BOOL, REGEX_FLOAT, REGEX_INT, REGEX_STR, SglSamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRuntimeEndpointDtypeToRegex(CustomTestCase):
    def setUp(self):
        self.endpoint = object.__new__(RuntimeEndpoint)

    def test_int_dtype_with_str_stop(self):
        params = SglSamplingParams(dtype=int, stop="\n")
        self.endpoint._handle_dtype_to_regex(params)
        self.assertEqual(params.regex, REGEX_INT)
        self.assertEqual(params.stop, ["\n", " ", "\n"])

    def test_float_dtype_with_str_stop(self):
        params = SglSamplingParams(dtype=float, stop=";")
        self.endpoint._handle_dtype_to_regex(params)
        self.assertEqual(params.regex, REGEX_FLOAT)
        self.assertEqual(params.stop, [";", " ", "\n"])

    def test_int_dtype_with_tuple_stop(self):
        params = SglSamplingParams(dtype="int", stop=("\n", ","))
        self.endpoint._handle_dtype_to_regex(params)
        self.assertEqual(params.regex, REGEX_INT)
        self.assertEqual(params.stop, ["\n", ",", " ", "\n"])

    def test_float_dtype_does_not_mutate_caller_stop_list(self):
        user_stop = ["END"]
        params = SglSamplingParams(dtype="float", stop=user_stop)
        self.endpoint._handle_dtype_to_regex(params)
        self.assertEqual(user_stop, ["END"])
        self.assertEqual(params.stop, ["END", " ", "\n"])

    def test_default_empty_tuple_stop(self):
        params = SglSamplingParams(dtype=int)
        self.endpoint._handle_dtype_to_regex(params)
        self.assertEqual(params.regex, REGEX_INT)
        self.assertEqual(params.stop, [" ", "\n"])

    def test_str_and_bool_dtypes(self):
        str_params = SglSamplingParams(dtype=str, stop="\n")
        self.endpoint._handle_dtype_to_regex(str_params)
        self.assertEqual(str_params.regex, REGEX_STR)
        self.assertEqual(str_params.stop, ["\n"])

        bool_params = SglSamplingParams(dtype=bool)
        self.endpoint._handle_dtype_to_regex(bool_params)
        self.assertEqual(bool_params.regex, REGEX_BOOL)
        self.assertEqual(bool_params.stop, [])


if __name__ == "__main__":
    unittest.main()
