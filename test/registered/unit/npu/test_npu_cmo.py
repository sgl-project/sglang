"""
Unit tests for sglang.srt.hardware_backend.npu.cmo.
"""

import unittest

import sglang.srt.hardware_backend.npu.cmo as cmo_mod
from sglang.srt.hardware_backend.npu.cmo import (
    get_cmo_stream,
    get_share_stream,
    set_cmo_stream,
    set_share_stream,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestGetSetCmoStream(unittest.TestCase):
    def setUp(self):
        self._saved = cmo_mod.cmo_stream
        cmo_mod.cmo_stream = None

    def tearDown(self):
        cmo_mod.cmo_stream = self._saved

    def test_get_returns_none_initially(self):
        self.assertIsNone(get_cmo_stream())

    def test_set_then_get_returns_value(self):
        sentinel = object()
        set_cmo_stream(sentinel)
        self.assertIs(get_cmo_stream(), sentinel)

    def test_set_to_none_resets(self):
        set_cmo_stream(object())
        set_cmo_stream(None)
        self.assertIsNone(get_cmo_stream())


class TestGetSetShareStream(unittest.TestCase):
    def setUp(self):
        self._saved = cmo_mod.share_stream
        cmo_mod.share_stream = None

    def tearDown(self):
        cmo_mod.share_stream = self._saved

    def test_get_returns_none_initially(self):
        self.assertIsNone(get_share_stream())

    def test_set_then_get_returns_value(self):
        sentinel = object()
        set_share_stream(sentinel)
        self.assertIs(get_share_stream(), sentinel)

    def test_set_to_none_resets(self):
        set_share_stream(object())
        set_share_stream(None)
        self.assertIsNone(get_share_stream())


class TestGetCmoStreamDefaultNone(unittest.TestCase):
    def setUp(self):
        self._saved = cmo_mod.cmo_stream
        cmo_mod.cmo_stream = None

    def tearDown(self):
        cmo_mod.cmo_stream = self._saved

    def test_module_level_default_is_none(self):
        self.assertIsNone(cmo_mod.cmo_stream)
        self.assertIsNone(get_cmo_stream())


class TestGetShareStreamDefaultNone(unittest.TestCase):
    def setUp(self):
        self._saved = cmo_mod.share_stream
        cmo_mod.share_stream = None

    def tearDown(self):
        cmo_mod.share_stream = self._saved

    def test_module_level_default_is_none(self):
        self.assertIsNone(cmo_mod.share_stream)
        self.assertIsNone(get_share_stream())


if __name__ == "__main__":
    unittest.main()
