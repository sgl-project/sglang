"""Unit tests for srt/multimodal/customized_mm_processor_utils.py — no server, no model weights."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

import sglang.srt.multimodal.customized_mm_processor_utils as cmmpu
from sglang.test.test_utils import CustomTestCase
from transformers import PretrainedConfig, ProcessorMixin


class _DummyProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class TestRegisterCustomizedProcessor(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._orig = dict(cmmpu._CUSTOMIZED_MM_PROCESSOR)
        cmmpu._CUSTOMIZED_MM_PROCESSOR.clear()

    def tearDown(self):
        cmmpu._CUSTOMIZED_MM_PROCESSOR.clear()
        cmmpu._CUSTOMIZED_MM_PROCESSOR.update(self._orig)

    def test_missing_model_type_raises(self):
        class NoModelTypeConfig:
            pass

        decorator = cmmpu.register_customized_processor(_DummyProcessor)
        with self.assertRaises(ValueError):
            decorator(NoModelTypeConfig)

    def test_registers_processor_by_model_type(self):
        class MyConfig(PretrainedConfig):
            model_type = "my_model"

        decorator = cmmpu.register_customized_processor(_DummyProcessor)
        out_cls = decorator(MyConfig)

        self.assertIs(out_cls, MyConfig)
        self.assertIn("my_model", cmmpu._CUSTOMIZED_MM_PROCESSOR)
        self.assertIs(cmmpu._CUSTOMIZED_MM_PROCESSOR["my_model"], _DummyProcessor)

    def test_subsequent_registration_overwrites(self):
        class MyConfig(PretrainedConfig):
            model_type = "my_model"

        class OtherProcessor(_DummyProcessor):
            pass

        cmmpu.register_customized_processor(_DummyProcessor)(MyConfig)
        cmmpu.register_customized_processor(OtherProcessor)(MyConfig)

        self.assertIs(cmmpu._CUSTOMIZED_MM_PROCESSOR["my_model"], OtherProcessor)


if __name__ == "__main__":
    import unittest

    unittest.main()
