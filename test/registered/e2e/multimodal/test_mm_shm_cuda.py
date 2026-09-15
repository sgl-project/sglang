import gc
import pickle
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestShmCudaReadLifetime(CustomTestCase):
    def setUp(self):
        override = envs.SGLANG_ENABLE_MM_SHM_ZERO_COPY.override(True)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def test_nonblocking_copy_survives_request_release(self):
        """A queued H2D read must remain valid after the CPU request is dropped."""
        stream = torch.cuda.Stream()
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = torch.arange(1024, dtype=dtype).repeat(4096)
                sender = ShmPointerMMData(expected)
                self.addCleanup(sender.close_and_unlink)
                receiver = pickle.loads(pickle.dumps(sender))
                feature = receiver.materialize()[17:-13]
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(10_000_000)
                    device_feature = feature.to("cuda", non_blocking=True)
                del receiver, feature
                gc.collect()
                stream.synchronize()
                self.assertTrue(torch.equal(device_feature.cpu(), expected[17:-13]))


if __name__ == "__main__":
    unittest.main()
