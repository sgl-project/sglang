import pickle
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.mm_utils import ShmPointerMMData

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestShmPointerMMData(CustomTestCase):
    def test_repickle_creates_independent_shared_memory_segments(self):
        tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        wrapper = ShmPointerMMData(tensor)

        first_payload = pickle.dumps(wrapper)
        second_payload = pickle.dumps(wrapper)

        first_receiver = pickle.loads(first_payload)
        second_receiver = None
        try:
            self.assertTrue(torch.equal(first_receiver.materialize(), tensor))

            second_receiver = pickle.loads(second_payload)
            self.assertTrue(torch.equal(second_receiver.materialize(), tensor))
        finally:
            if (
                second_receiver is not None
                and getattr(second_receiver, "_shm_handle", None) is not None
            ):
                second_receiver.materialize()


if __name__ == "__main__":
    unittest.main()
