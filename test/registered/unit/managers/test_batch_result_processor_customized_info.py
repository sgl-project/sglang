import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_NON_NATIVE_FACTORIES = {
    "array": lambda: array("i", [1]),
    "ndarray": lambda: np.array([1], dtype=np.int32),
    "tensor": lambda: torch.tensor([1], dtype=torch.int32),
}


class TestBatchResultProcessorCustomizedInfo(unittest.TestCase):
    def setUp(self):
        self.processor = object.__new__(SchedulerBatchResultProcessor)

    def test_msgpack_rejects_non_native_values(self):
        for name, factory in _NON_NATIVE_FACTORIES.items():
            with self.subTest(value=name):
                req = SimpleNamespace(customized_info=None)
                logits_output = SimpleNamespace(customized_info={"probe": [factory()]})

                with patch.object(batch_result_processor, "_USE_PICKLE_IPC", False):
                    with self.assertRaisesRegex(
                        TypeError,
                        r"customized_info\['probe'\]\[0\]",
                    ):
                        self.processor._maybe_collect_customized_info(
                            0,
                            req,
                            logits_output,
                        )

    def test_msgpack_accepts_native_values(self):
        values = (None, True, 1, 0.5, "tag", [1, "x"], {"ok": True})
        req = SimpleNamespace(customized_info=None)
        logits_output = SimpleNamespace(
            customized_info={
                f"value_{index}": [value] for index, value in enumerate(values)
            }
        )

        with patch.object(batch_result_processor, "_USE_PICKLE_IPC", False):
            self.processor._maybe_collect_customized_info(0, req, logits_output)

        actual = tuple(
            req.customized_info[f"value_{index}"][0] for index in range(len(values))
        )
        self.assertEqual(actual, values)
        self.assertEqual(
            tuple(type(value) for value in actual),
            (type(None), bool, int, float, str, list, dict),
        )

    def test_msgpack_rejects_non_native_nested_values(self):
        values = {
            "list_leaf": [torch.tensor([1])],
            "map_leaf": {"value": np.array([1])},
            "map_key": {1: "value"},
        }
        for name, value in values.items():
            with self.subTest(value=name):
                req = SimpleNamespace(customized_info=None)
                logits_output = SimpleNamespace(customized_info={"probe": [value]})

                with patch.object(batch_result_processor, "_USE_PICKLE_IPC", False):
                    with self.assertRaises(TypeError):
                        self.processor._maybe_collect_customized_info(
                            0,
                            req,
                            logits_output,
                        )

    def test_pickle_accepts_non_native_values(self):
        for name, factory in _NON_NATIVE_FACTORIES.items():
            with self.subTest(value=name):
                original = factory()
                req = SimpleNamespace(customized_info=None)
                logits_output = SimpleNamespace(customized_info={"probe": [original]})

                with patch.object(batch_result_processor, "_USE_PICKLE_IPC", True):
                    self.processor._maybe_collect_customized_info(
                        0,
                        req,
                        logits_output,
                    )

                self.assertIs(type(req.customized_info["probe"][0]), type(original))


if __name__ == "__main__":
    unittest.main()
