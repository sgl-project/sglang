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
# An owner free never names a page a host load reserved: a cancelled load
# unfixes its pages before it frees them.
FREE_OUTSIDE_RESERVATION = Invariant(
    "band.free_outside_reservation", Bucket.FATAL_UNCONTAINABLE, IsTrue()
)
# Layout-changing geometry decides on holes and live pages only: the band holds
# no pending page and, inside an urgent recovery, its free list's host copy is
# current.
GEOMETRY_READY = Invariant("band.geometry_ready", Bucket.GUARD, IsTrue())
# An allocator operation reads device state on the host only before its first
# stream wait.
HOST_READ_BEFORE_WAIT = Invariant("band.host_read_before_wait", Bucket.GUARD, IsTrue())
# Under pipeline parallelism every stage makes the same capacity decisions and
# holds the same units.
STAGE_CAPACITY_AGREEMENT = Invariant(
    "chain.stage_capacity_agreement", Bucket.FATAL_UNCONTAINABLE, IsTrue()
)
