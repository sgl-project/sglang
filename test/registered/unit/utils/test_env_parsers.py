import os
import unittest
import uuid
from unittest.mock import patch

from sglang.srt.environ import EnvBool, EnvInt
from sglang.srt.utils import common as common_utils
from sglang.srt.utils.common import get_bool_env_var, get_int_env_var
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGetBoolEnvVar(CustomTestCase):
    def setUp(self):
        common_utils._warned_bool_env_var_keys.clear()

    def test_strips_whitespace_for_truthy_values(self):
        key = f"SGLANG_TEST_BOOL_{uuid.uuid4().hex}"
        with patch.dict(os.environ, {key: "  true  "}):
            self.assertTrue(get_bool_env_var(key))

    def test_strips_whitespace_for_falsy_values(self):
        key = f"SGLANG_TEST_BOOL_{uuid.uuid4().hex}"
        with patch.dict(os.environ, {key: "\tfalse\n"}):
            self.assertFalse(get_bool_env_var(key))

    def test_whitespace_only_value_is_treated_as_false(self):
        key = f"SGLANG_TEST_BOOL_{uuid.uuid4().hex}"
        with patch.dict(os.environ, {key: "   "}):
            self.assertFalse(get_bool_env_var(key))


class TestGetIntEnvVar(CustomTestCase):
    def test_strips_whitespace_before_parsing(self):
        key = f"SGLANG_TEST_INT_{uuid.uuid4().hex}"
        with patch.dict(os.environ, {key: "  42  "}):
            self.assertEqual(get_int_env_var(key, default=0), 42)

    def test_whitespace_only_value_returns_default(self):
        key = f"SGLANG_TEST_INT_{uuid.uuid4().hex}"
        with patch.dict(os.environ, {key: "   "}):
            self.assertEqual(get_int_env_var(key, default=7), 7)


class TestEnvFieldParsers(CustomTestCase):
    def test_env_bool_strips_whitespace(self):
        field = EnvBool(False)
        field.name = "SGLANG_TEST_ENV_BOOL"
        self.assertTrue(field.parse("  yes  "))
        self.assertFalse(field.parse("\t0\n"))

    def test_env_int_strips_whitespace(self):
        field = EnvInt(0)
        field.name = "SGLANG_TEST_ENV_INT"
        self.assertEqual(field.parse("  128  "), 128)


if __name__ == "__main__":
    unittest.main()
