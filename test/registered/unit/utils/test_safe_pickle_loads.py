import pickle
import unittest

import torch

from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSafePickleLoads(unittest.TestCase):
    def test_blocks_builtin_import_gadget_chain(self):
        payload = (
            b"cbuiltins\n__import__\n(S'os'\ntRp0\n"
            b"cbuiltins\ngetattr\n(g0\nS'system'\ntR"
            b"(S'touch /tmp/sglang-safe-unpickler-test-marker'\ntR."
        )

        with self.assertRaisesRegex(RuntimeError, "Blocked unsafe class loading"):
            safe_pickle_loads(payload)

    def test_preserves_tensor_round_trip(self):
        payload = {
            "weights": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "bias": torch.tensor([1.0, 2.0]),
        }

        restored = safe_pickle_loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

        torch.testing.assert_close(restored["weights"], payload["weights"])
        torch.testing.assert_close(restored["bias"], payload["bias"])


if __name__ == "__main__":
    unittest.main()
