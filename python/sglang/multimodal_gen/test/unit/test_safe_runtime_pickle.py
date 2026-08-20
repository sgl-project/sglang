import pickle
import unittest

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.utils.common import safe_runtime_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSafeRuntimePickle(unittest.TestCase):
    def test_blocks_builtin_import_gadget_chain(self):
        payload = (
            b"cbuiltins\n__import__\n(S'os'\ntRp0\n"
            b"cbuiltins\ngetattr\n(g0\nS'system'\ntR"
            b"(S'touch /tmp/sglang-safe-runtime-pickle-test-marker'\ntR."
        )

        with self.assertRaisesRegex(RuntimeError, "Blocked unsafe class loading"):
            safe_runtime_pickle_loads(payload)

    def test_preserves_runtime_control_requests(self):
        req = UpdateWeightFromDiskReqInput(
            model_path="demo/model",
            flush_cache=False,
            target_modules=["unet"],
        )

        restored = safe_runtime_pickle_loads(
            pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        )

        self.assertEqual(restored, req)


if __name__ == "__main__":
    unittest.main()
