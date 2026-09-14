"""Cross-stream lifetime of the overlap snapshot, on the real caching allocator.

The CPU tests pin ownership and ordering with fake events. This one runs the
hazard the ownership exists for: a tensor produced on the schedule stream, read
by a kernel still queued on the forward stream, and freed by the scheduler's
release. If the release runs before that batch's completion event, the caching
allocator can hand the block to the next schedule-stream allocation and the
queued read sees the overwrite.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

# Long enough that the schedule stream finishes its poison write first, short
# enough to keep the test quick. Same units as torch.cuda._sleep elsewhere.
_FORWARD_STREAM_DELAY_CYCLES = 400_000_000
_ELEMS = 4096
_ORIGINAL = 7
_POISON = -1


def _run_once(release_before_event: bool):
    """Play one launch, release, and reallocation.

    Returns (block_was_reused, read_saw_original).
    """
    schedule = torch.cuda.Stream()
    forward = torch.cuda.Stream()

    with torch.cuda.stream(schedule):
        src = torch.full((_ELEMS,), _ORIGINAL, dtype=torch.int32, device="cuda")
    block = src.data_ptr()

    with torch.cuda.stream(forward):
        forward.wait_stream(schedule)
        # Stand in for the forward: the read of src happens well after the
        # scheduler has moved on.
        torch.cuda._sleep(_FORWARD_STREAM_DELAY_CYCLES)
        observed = src.clone()
        completion = torch.cuda.Event()
        completion.record(forward)

    # The scheduler's release point. release_keep_alive_refs waits on the
    # completion event before dropping the reference; the mutant does not.
    if not release_before_event:
        completion.synchronize()
    del src

    # The next schedule-stream allocation of the same size.
    with torch.cuda.stream(schedule):
        reused = torch.full((_ELEMS,), _POISON, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()

    return reused.data_ptr() == block, bool((observed == _ORIGINAL).all().item())


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestOverlapSnapshotSurvivesAllocatorReuse(CustomTestCase):
    def test_releasing_before_the_event_loses_the_tensor(self):
        """The hazard: the block is recycled under a queued read."""
        reused, saw_original = _run_once(release_before_event=True)

        self.assertTrue(reused, "the allocator did not reuse the freed block")
        self.assertFalse(
            saw_original,
            "expected the queued read to see the reallocation's contents",
        )

    def test_releasing_after_the_event_keeps_the_tensor(self):
        """The fix: the read has already happened, so recycling is harmless."""
        _, saw_original = _run_once(release_before_event=False)

        self.assertTrue(saw_original, "the queued read lost the original values")


if __name__ == "__main__":
    unittest.main()
