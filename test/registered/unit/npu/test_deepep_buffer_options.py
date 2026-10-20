"""NPU DeepEP selection uses public buffer options without global patches."""

import unittest

from sglang.srt.hardware_backend.npu.deepep_utils import alltoall_buffer_kwargs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Buffer:
    def __init__(self, group, *, normal_strategy="default"):
        self.normal_strategy = normal_strategy


class _LegacyBuffer:
    def __init__(self, group):
        pass


class TestDeepEPBufferOptions(CustomTestCase):
    def options(self, buffer_cls=_Buffer, **overrides):
        values = dict(
            enabled=True, normal_enabled=True, use_zbal=False, deep_use_mode=None
        )
        values.update(overrides)
        return alltoall_buffer_kwargs(buffer_cls, **values)

    def test_opt_in_selects_normal_strategy_for_only_the_new_buffer(self):
        before = _Buffer(None)
        selected = _Buffer(None, **self.options())
        after = _Buffer(None)
        self.assertEqual(selected.normal_strategy, "alltoall")
        self.assertEqual(before.normal_strategy, "default")
        self.assertEqual(after.normal_strategy, "default")

    def test_default_does_not_require_private_vendor_interfaces(self):
        self.assertEqual(self.options(_LegacyBuffer, enabled=False), {})

    def test_zbal_and_low_latency_keep_their_own_strategy(self):
        self.assertEqual(self.options(_LegacyBuffer, use_zbal=True), {})
        self.assertEqual(self.options(_LegacyBuffer, normal_enabled=False), {})

    def test_explicit_opt_in_rejects_an_unsupported_buffer_api(self):
        with self.assertRaisesRegex(ValueError, "normal_strategy"):
            self.options(_LegacyBuffer)

    def test_vendor_environment_cannot_silently_override_the_opt_in(self):
        for deep_mode in ("default", "ops", "alltoall", ""):
            with self.subTest(deep_mode=deep_mode):
                with self.assertRaisesRegex(ValueError, "DEEP_USE_MODE"):
                    self.options(deep_use_mode=deep_mode)


if __name__ == "__main__":
    unittest.main()
