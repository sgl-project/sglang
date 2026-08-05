import inspect
import unittest

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool


class TestNPUMLATokenToKVPool(unittest.TestCase):
    def test_index_head_dim_is_optional_for_hybrid_mla(self):
        parameter = inspect.signature(NPUMLATokenToKVPool).parameters[
            "index_head_dim"
        ]
        self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
