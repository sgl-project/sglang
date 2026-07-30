import pickle

import torch

from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_shm_pointer_round_trips_non_page_aligned_tensor():
    tensor = torch.arange(1025, dtype=torch.float32).reshape(5, 205)
    pointer = ShmPointerMMData(tensor)
    restored_pointer = pickle.loads(pickle.dumps(pointer))

    restored = restored_pointer.materialize()
    torch.testing.assert_close(restored, tensor)
