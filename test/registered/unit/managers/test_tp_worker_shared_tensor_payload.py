"""A tensor weight update reads its own rank's payload, or one payload shared by every rank."""

import io
import pickle
import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers.tp_worker import BaseTpWorker
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _payload(value: float) -> bytes:
    # HTTP callers ship a plain pickle of CPU tensors (no shared-memory or IPC handles).
    buffer = io.BytesIO()
    pickle.dump([("weight", torch.full((2, 2), value))], buffer)
    return buffer.getvalue()


class TestDeserializeOwnRank(unittest.TestCase):
    def _read(self, serialized, tp_rank: int) -> torch.Tensor:
        # The torch-reduction monkey patch is process-wide; a plain pickle of CPU tensors
        # does not need it, so keep it out of other tests in the same process.
        with (
            patch("sglang.srt.managers.tp_worker.monkey_patch_torch_reductions"),
            get_parallel().override(tp_rank=tp_rank),
        ):
            ((name, tensor),) = BaseTpWorker.deserialize_own_rank(None, serialized)
        self.assertEqual(name, "weight")
        return tensor

    def test_per_rank_list_reads_own_entry(self):
        serialized = [_payload(float(rank)) for rank in range(4)]
        for rank in range(4):
            self.assertTrue(
                torch.equal(
                    self._read(serialized, rank), torch.full((2, 2), float(rank))
                )
            )

    def test_single_entry_is_shared_by_every_rank(self):
        serialized = [_payload(7.0)]
        for rank in range(4):
            self.assertTrue(
                torch.equal(self._read(serialized, rank), torch.full((2, 2), 7.0))
            )


if __name__ == "__main__":
    unittest.main()
