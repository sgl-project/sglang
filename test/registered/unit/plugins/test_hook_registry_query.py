"""Public hook readiness queries reflect successful application."""

import sys
import types
import unittest
import uuid

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHookRegistryQuery(CustomTestCase):
    def setUp(self):
        HookRegistry.reset()
        self.module_name = f"_hook_query_test_{uuid.uuid4().hex}"
        self.module = types.ModuleType(self.module_name)
        self.module.operation = lambda value: value * 2
        sys.modules[self.module_name] = self.module
        self.target = f"{self.module_name}.operation"

    def tearDown(self):
        HookRegistry.reset()
        sys.modules.pop(self.module_name, None)

    def test_query_requires_successful_application_and_exact_callback_type(self):
        def add_one(result, value):
            return result + 1

        self.assertFalse(HookRegistry.is_hook_applied(self.target, add_one))
        HookRegistry.register(self.target, add_one, HookType.AFTER)
        self.assertFalse(HookRegistry.is_hook_applied(self.target, add_one))

        HookRegistry.apply_hooks()

        self.assertEqual(self.module.operation(3), 7)
        self.assertTrue(HookRegistry.is_hook_applied(self.target, add_one))
        self.assertFalse(
            HookRegistry.is_hook_applied(self.target, add_one, HookType.BEFORE)
        )
        self.assertFalse(
            HookRegistry.is_hook_applied(self.target, lambda result, value: result + 1)
        )
        self.assertFalse(HookRegistry.is_hook_applied(self.target + "_other", add_one))

    def test_late_registration_is_not_applied_when_target_was_already_patched(self):
        def first(result, value):
            return result + 1

        def late(result, value):
            return result + 100

        HookRegistry.register(self.target, first)
        HookRegistry.apply_hooks()
        HookRegistry.register(self.target, late)
        HookRegistry.apply_hooks()

        self.assertEqual(self.module.operation(3), 7)
        self.assertTrue(HookRegistry.is_hook_applied(self.target, first))
        self.assertFalse(HookRegistry.is_hook_applied(self.target, late))

        HookRegistry.reset()
        self.assertFalse(HookRegistry.is_hook_applied(self.target, first))

    def test_failed_target_reports_false_until_successful_retry(self):
        target = f"{self.module_name}.missing"

        def add_one(result, value):
            return result + 1

        HookRegistry.register(target, add_one)
        with self.assertLogs("sglang.srt.plugins.hook_registry", level="ERROR"):
            HookRegistry.apply_hooks()
        self.assertFalse(HookRegistry.is_hook_applied(target, add_one))

        self.module.missing = lambda value: value * 2
        HookRegistry.apply_hooks()

        self.assertEqual(self.module.missing(3), 7)
        self.assertTrue(HookRegistry.is_hook_applied(target, add_one))


if __name__ == "__main__":
    unittest.main()
