"""Unit tests for srt/multimodal/customized_mm_processor_utils.py — no server, no model weights."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

# Load the target module from source with a lightweight `transformers` stub.
_old_transformers = sys.modules.get("transformers")
if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class PretrainedConfig:
        model_type = None

    class ProcessorMixin:
        pass

    transformers_stub.PretrainedConfig = PretrainedConfig
    transformers_stub.ProcessorMixin = ProcessorMixin
    sys.modules["transformers"] = transformers_stub
else:
    from transformers import PretrainedConfig, ProcessorMixin

if "PretrainedConfig" not in globals():
    from transformers import PretrainedConfig, ProcessorMixin

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_TARGET = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "multimodal"
    / "customized_mm_processor_utils.py"
)
spec = importlib.util.spec_from_file_location(
    "customized_mm_processor_utils_test", str(_TARGET)
)
cmmpu = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cmmpu)

if _old_transformers is None:
    sys.modules.pop("transformers", None)
else:
    sys.modules["transformers"] = _old_transformers


class _DummyProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class TestRegisterCustomizedProcessor(unittest.TestCase):
    def setUp(self):
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
    unittest.main()
