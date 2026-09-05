"""A platform fact has one address, and one place to change it.

The value lives in one object, so `override_platform(...)` reaches every reader
at once, and the facts answer before anything is published.
"""

import unittest

from sglang.srt.runtime_context import (
    _PLATFORM_PROBES,
    _PLATFORM_VALUES,
    get_platform,
    override_platform,
)
from sglang.srt.utils import common as _common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlatformContext(CustomTestCase):
    def test_every_name_maps_to_a_real_probe(self):
        """A typo in either table would answer with an AttributeError at runtime."""
        for name, probe in {**_PLATFORM_PROBES, **_PLATFORM_VALUES}.items():
            self.assertTrue(
                callable(getattr(_common, probe, None)),
                f"{name} maps to {probe!r}, which is not callable in utils.common",
            )

    def test_it_answers_before_anything_is_published(self):
        """Unlike a config bag: a launcher asks these before it publishes."""
        platform = get_platform()
        for name in _PLATFORM_PROBES:
            self.assertIsInstance(getattr(platform, name), bool)

    def test_the_probe_is_what_it_answers_with(self):
        platform = get_platform()
        for name, probe in {**_PLATFORM_PROBES, **_PLATFORM_VALUES}.items():
            self.assertEqual(getattr(platform, name), getattr(_common, probe)())

    def test_an_override_is_scoped_and_restores(self):
        platform = get_platform()
        before = platform.is_sm100
        with override_platform(is_sm100=not before):
            self.assertEqual(platform.is_sm100, not before)
        self.assertEqual(platform.is_sm100, before)

    def test_overrides_nest(self):
        platform = get_platform()
        with override_platform(is_sm100=True):
            self.assertTrue(platform.is_sm100)
            with override_platform(is_sm100=False):
                self.assertFalse(platform.is_sm100)
            self.assertTrue(platform.is_sm100)

    def test_an_unknown_fact_is_refused_on_both_paths(self):
        with self.assertRaises(AttributeError):
            get_platform().is_sm999
        with self.assertRaises(ValueError):
            override_platform(is_sm999=True).install()

    def test_a_fact_is_not_assigned(self):
        """Assignment would move it for one reader, which is the defect."""
        with self.assertRaises(AttributeError):
            get_platform().is_sm100 = True

    def test_one_statement_reaches_every_reader(self):
        """The point of the address.

        Two modules that both ask the context see the same answer from one
        override -- which is what the round-4 failure needed and could not say.
        """

        def module_a():
            return get_platform().is_sm100

        def module_b():
            return get_platform().is_sm100

        with override_platform(is_sm100=True):
            self.assertTrue(module_a())
            self.assertTrue(module_b())
        with override_platform(is_sm100=False):
            self.assertFalse(module_a())
            self.assertFalse(module_b())


if __name__ == "__main__":
    unittest.main()
