# SPDX-License-Identifier: Apache-2.0
"""Names shared by the Nomas realtime adapter (HTTP process) and pipeline (worker)."""

# condition_inputs key: this chunk only flushes withheld tail frames.
NOMAS_FLUSH_CONDITION = "nomas_flush"
# condition_inputs key: the event segment carried by this chunk.
NOMAS_SEGMENT_CONDITION = "nomas_segment"
# Placeholder caption encoded for a flush chunk; never conditions a segment.
NOMAS_FLUSH_CAPTION = "flush"
