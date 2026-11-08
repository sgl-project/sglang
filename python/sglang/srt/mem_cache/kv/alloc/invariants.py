"""Invariants the chain allocator checks, registered once per process: a
module that is executed again (a test fixture re-importing the allocator)
must not register them twice."""

from sglang.srt.utils.invariants import Bucket, Invariant, IsTrue

# A window's segment frees trust the caller's split between rows still bound
# to the window and rows the window already released.
WINDOW_SEGMENT_BOUND = Invariant(
    "chain.window_segment_bound", Bucket.FATAL_UNCONTAINABLE, IsTrue()
)
WINDOW_SEGMENT_RELEASED = Invariant(
    "chain.window_segment_released", Bucket.FATAL_UNCONTAINABLE, IsTrue()
)
