"""Per-forward batches a speculative worker builds from the step it is handed.

A worker takes the scheduled batch (the step) as read-only. It builds its own
forward batch from it, and from that each phase's batch (draft, verify, draft
extend), with `derive_batch`; a phase rebinds fields only on the batch it was
built for.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def derive_batch(batch: ScheduleBatch, **fields) -> ScheduleBatch:
    """A new batch with every field of `batch` and `fields` rebound.

    `batch` is left as it is. Tensors, the reqs list and the sub-objects not
    named in `fields` are shared, so they are never written in place.
    """
    unknown = fields.keys() - batch.__dict__.keys()
    if unknown:
        raise AttributeError(f"ScheduleBatch has no fields {sorted(unknown)}")
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    derived = ScheduleBatch.__new__(ScheduleBatch)
    derived.__dict__.update(batch.__dict__)
    derived.__dict__.update(fields)
    return derived


def forward_sampling_info(
    sampling_info: Optional[SamplingBatchInfo],
) -> Optional[SamplingBatchInfo]:
    """The sampling info one forward reads and fills (the grammar mask, the
    token oracle's ids): a copy of the step's, which stays as built."""
    if sampling_info is None:
        return None
    return dataclasses.replace(sampling_info)


def worker_forward_batch(step: ScheduleBatch, **fields) -> ScheduleBatch:
    """The batch a worker forwards for `step`: a copy with a forward-only
    sampling info and a copy of the step's draft input (both may be filled
    in by the forward), plus `fields`."""
    fields.setdefault("sampling_info", forward_sampling_info(step.sampling_info))
    if "spec_info" not in fields and step.spec_info is not None:
        fields["spec_info"] = copy.copy(step.spec_info)
    return derive_batch(step, **fields)
