"""Ragged feature storage: keep the per-item parts instead of concatenating them.

Many multimodal processors build a list of per-image (or per-video) tensors,
``torch.cat`` it into one request-wide buffer, and hand that to the scheduler --
which splits it straight back apart in ``get_new_expanded_mm_items`` so each
item can be hashed and cached on its own. ``kimi_k25`` says so outright at its
call site: "Use SGL-standard key so get_new_expanded_mm_items() can split".

``SegmentedFeatures`` carries the list with its boundaries recorded, so neither
half of that round trip has to run. The parts are already allocated; the
concatenation is a second request-sized allocation on top of them, live at the
same time as the first, and it is the peak that decides whether a large image
request fits.

This is deliberately not an array-like object. HF ``BatchFeature`` must carry
the parts through without coercing them back into one tensor, so the class
exposes no ``__len__``/``__getitem__``/``__array__``. Scheduler and transport
consumers never see it: ``expand_segmented_item`` resolves it to ordinary
per-item tensors, each owning its own rows, before anything hashes or
transports them, and any consumer that cannot split an item falls back to
:meth:`dense`, which is exactly the ``torch.cat`` that would have run anyway.
Adopting this can therefore not change what a model computes.

One requirement on a producer: its output must reach the scheduler through
``process_and_combine_mm_data``, which is what calls
``get_new_expanded_mm_items`` and so what resolves the ragged storage. A
processor that instead builds ``MultimodalDataItem``s itself and returns a
``MultimodalProcessorOutput`` directly -- InternVL does -- never passes through
that call, and must hand over ordinary per-item tensors. It has no reason to
do otherwise: holding the parts is the whole point, and it already has them.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


def _storage_key(tensor: torch.Tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


@dataclass(frozen=True)
class SegmentedFeatures:
    """Per-item feature rows held in one or more owning buffers.

    ``slices`` is in item order and holds ``(buffer index, first row, last row
    exclusive)``. Several items may share one buffer -- a processor that
    batches same-shape images through one kernel produces exactly that -- and
    items sharing a buffer need not be adjacent in item order.
    """

    buffers: tuple[torch.Tensor, ...]
    slices: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        if not self.buffers or not self.slices:
            raise ValueError("segmented features must have buffers and slices")
        first = self.buffers[0]
        if first.ndim < 2:
            raise ValueError("feature buffers must have a leading row dimension")
        for buffer in self.buffers:
            if (
                buffer.shape[1:] != first.shape[1:]
                or buffer.dtype != first.dtype
                or buffer.device != first.device
            ):
                raise ValueError(
                    "feature buffers must agree on trailing shape, dtype and device"
                )
        coverage = [[] for _ in self.buffers]
        for index, start, end in self.slices:
            if not 0 <= index < len(self.buffers):
                raise ValueError("slice names a buffer that does not exist")
            if not 0 <= start < end <= self.buffers[index].shape[0]:
                raise ValueError("slice rows fall outside their buffer")
            coverage[index].append((start, end))
        for buffer, spans in zip(self.buffers, coverage):
            offset = 0
            # Sorted, not item order: items sharing a buffer may be reordered.
            for start, end in sorted(spans):
                if start != offset:
                    raise ValueError(
                        "slices must tile each buffer without gaps or overlap"
                    )
                offset = end
            if offset != buffer.shape[0]:
                raise ValueError("unreferenced feature rows")

    # -- shape and identity ------------------------------------------------

    @property
    def num_parts(self) -> int:
        return len(self.slices)

    @property
    def shape(self) -> tuple[int, ...]:
        rows = sum(end - start for _, start, end in self.slices)
        return (rows, *self.buffers[0].shape[1:])

    @property
    def dtype(self) -> torch.dtype:
        return self.buffers[0].dtype

    @property
    def device(self) -> torch.device:
        return self.buffers[0].device

    @property
    def storage_bytes(self) -> int:
        """Bytes actually retained, counting a shared storage once."""
        return sum(
            {
                _storage_key(b): b.untyped_storage().nbytes() for b in self.buffers
            }.values()
        )

    def __repr__(self) -> str:
        # The generated dataclass repr would print every buffer's contents.
        return (
            f"SegmentedFeatures(parts={self.num_parts}, "
            f"buffers={len(self.buffers)}, shape={self.shape}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    # -- access ------------------------------------------------------------

    def part(self, index: int) -> torch.Tensor:
        buffer, start, end = self.slices[index]
        return self.buffers[buffer][start:end]

    def parts(self) -> list[torch.Tensor]:
        return [self.part(i) for i in range(self.num_parts)]

    def to(self, device) -> SegmentedFeatures:
        """Move every owning buffer once, keeping per-item views and order."""
        return type(self)(
            tuple(buffer.to(device) for buffer in self.buffers), self.slices
        )

    def dense(self) -> torch.Tensor:
        """The request-wide tensor this storage exists to avoid.

        The fallback for any consumer that cannot take the parts apart. It
        allocates exactly what ``torch.cat`` over the original list would --
        including when one buffer already holds every row in order, where
        returning that buffer would alias the producer's memory and make an
        in-place write by the consumer reach back into it.
        """
        return torch.cat(self.parts(), dim=0)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_parts(cls, parts: Sequence[torch.Tensor]) -> SegmentedFeatures:
        """Drop-in for ``torch.cat(parts, dim=0)`` that keeps the parts.

        Runs of consecutive parts that are already contiguous slices of one
        storage -- what a processor that batches same-shape items through one
        kernel produces -- are recorded as a single buffer, so a later device
        transfer moves them in one copy rather than one per item.

        One deliberate difference from ``torch.cat``: a zero-row part is
        rejected rather than absorbed. Each part has to stand for exactly one
        placeholder, so a part with no rows is an item with no feature, which
        expansion could not produce a usable item for. Failing at the producer
        names the problem; ``torch.cat`` would swallow it and mis-split later.
        """
        if not parts:
            raise ValueError("cannot build segmented features from no parts")
        for part in parts:
            if part.ndim < 2:
                raise ValueError("feature parts must have a leading row dimension")
            if part.shape[0] == 0:
                raise ValueError("feature parts must be non-empty")

        buffers: list[torch.Tensor] = []
        slices: list[tuple[int, int, int]] = []
        run_start = 0
        while run_start < len(parts):
            run_end = run_start + 1
            head = parts[run_start]
            if head.is_contiguous():
                key = _storage_key(head)
                offset = head.storage_offset() + head.numel()
                while run_end < len(parts):
                    nxt = parts[run_end]
                    if (
                        not nxt.is_contiguous()
                        or _storage_key(nxt) != key
                        or nxt.storage_offset() != offset
                        or nxt.shape[1:] != head.shape[1:]
                        or nxt.dtype != head.dtype
                    ):
                        break
                    offset += nxt.numel()
                    run_end += 1
            run = parts[run_start:run_end]
            if len(run) == 1:
                buffer = run[0]
            else:
                rows = sum(part.shape[0] for part in run)
                buffer = head.as_strided((rows, *head.shape[1:]), head.stride())
            index = len(buffers)
            buffers.append(buffer)
            row = 0
            for part in run:
                slices.append((index, row, row + part.shape[0]))
                row += part.shape[0]
            run_start = run_end
        return cls(tuple(buffers), tuple(slices))


def _narrowed(part: torch.Tensor) -> torch.Tensor:
    """A tensor holding only its own rows.

    A part cut from a shared buffer is a view, and pickle serialises a view's
    whole underlying storage, so an item that leaves here still holding one
    would carry every other item's bytes to the scheduler. The copy is paid
    once, here, and only when the view is not already the whole buffer.

    The type check is exact on purpose: a tensor subclass may keep state that
    ``clone`` does not reproduce, and it owns its own transport anyway.
    """
    if type(part) is not torch.Tensor:
        return part
    if part.untyped_storage().nbytes() > part.numel() * part.element_size():
        return part.clone()
    return part


def expand_segmented_item(
    item,
    slice_model_data: Callable,
    offsets_per_part: Sequence[int] | None = None,
) -> list | None:
    """Resolve ragged ownership into per-item tensor views before hashing.

    ``offsets_per_part`` says how many consecutive placeholder spans each part
    owns; the default is one each. More than one is not exotic -- Step3 wraps
    every crop of an image in its own boundary tokens, so one image's feature
    covers ``num_patches + 1`` spans of the placeholder token, and taking one
    span per part would cut the request in the wrong places.

    Returns ``None`` when this item cannot be split one part per image -- a
    video whose offsets count frames rather than videos, or a grouping that
    does not account for every span. The caller then falls back to
    :meth:`SegmentedFeatures.dense`, so a processor can always emit segmented
    features without having to know whether every consumer downstream can use
    them.
    """
    features = item.feature
    if item.precomputed_embeddings is not None:
        return None
    if item.offsets is None:
        return None

    if offsets_per_part is None:
        offsets_per_part = [1] * features.num_parts
    if len(offsets_per_part) != features.num_parts:
        return None
    if sum(offsets_per_part) != len(item.offsets):
        return None

    expanded_items = []
    offset = 0
    first_span = 0
    for index in range(features.num_parts):
        part = features.part(index)
        end = offset + part.shape[0]
        last_span = first_span + offsets_per_part[index]
        expanded = copy.copy(item)
        expanded.feature = _narrowed(part)
        expanded.offsets = list(item.offsets[first_span:last_span])
        expanded.model_specific_data = slice_model_data(
            item.model_specific_data,
            index=index,
            start=offset,
            end=end,
            num_items=features.num_parts,
            total_feature_len=features.shape[0],
        )
        # Both are derived from the feature, which has just changed.
        expanded.hash = None
        expanded.pad_value = None
        expanded_items.append(expanded)
        offset = end
        first_span = last_span
    return expanded_items
