"""FB-shared slot registry for CUDA Graph (and eager) forward paths.

This module provides ``CudaGraphBufferRegistry`` — the unified FB →
graph-resident buffer mirror used by eager / capture / replay. It replaces
the scattered ``DecodeInputBuffers`` / ``PrefillInputBuffers`` god dataclasses
and their ad-hoc ``populate_from_forward_batch`` methods with a single
``GraphSlot``-driven registry.

Scope:

  * ``GraphSlot``   — single FB-field spec (shape / dtype / padding policy /
                      optional post-fill hook).
  * ``PaddingPolicy`` — enum capturing the four padding behaviors the old
                      populate path encoded inline (``KEEP_PAD`` /
                      ``FILL_SENTINEL`` / ``ZERO`` / ``FOREACH_COPY``).
  * ``CudaGraphBufferRegistry`` — registers slots, allocates their physical
                      buffers (sharing via ``ForwardInputBuffers`` pool),
                      and exposes ``fill_from(fb, padded_bs, ...)`` +
                      ``extract_buffer(template) -> ForwardBatch`` for the
                      callers.

Backend-private buffers (kernel workspaces, derived page tables, etc.) stay
on ``AttentionBackend.cuda_graph_*`` — the registry only owns FB-shared
slots (FB attribute name maps 1:1 to slot name).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.model_executor.input_buffers import share_input_buffer

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_has_foreach_copy = hasattr(torch, "_foreach_copy_")


def _grouped_foreach_copy_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
    """Call torch._foreach_copy_ grouped by (dst_dtype, src_dtype) pairs.

    Preserves the dtype-grouping perf optimization used by the old
    ``DecodeInputBuffers.populate_from_forward_batch`` path.
    """

    def _foreach_copy(
        group_dsts: List[torch.Tensor], group_srcs: List[torch.Tensor]
    ) -> None:
        if _has_foreach_copy:
            torch._foreach_copy_(group_dsts, group_srcs)
        else:
            for dst, src in zip(group_dsts, group_srcs):
                dst.copy_(src)

    groups: Dict[Tuple[torch.dtype, torch.dtype], Tuple[List, List]] = {}
    for dst, src in zip(dsts, srcs):
        key = (dst.dtype, src.dtype)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(dst)
        groups[key][1].append(src)
    for group_dsts, group_srcs in groups.values():
        _foreach_copy(group_dsts, group_srcs)


class PaddingPolicy(Enum):
    """How to handle ``raw_n < padded_n`` for a slot.

    KEEP_PAD      — Leave the padded region as-is (caller proves the
                    padded tail will not be read).
    FILL_SENTINEL — Reset the padded region to ``slot.pad_value`` before
                    copy (e.g. ``seq_lens`` filled with
                    ``seq_len_fill_value``).
    ZERO          — Reset the padded region to ``0`` (e.g.
                    ``out_cache_loc`` / ``req_pool_indices`` — padded
                    rows must point at slot 0 so dummy attention reads
                    land harmlessly).
    FOREACH_COPY  — Always copy ``raw_n`` from src; padded region is
                    left as whatever the previous replay (or the init
                    zeros) wrote. Caller is responsible for proving
                    safety.
    """

    KEEP_PAD = "keep_pad"
    FILL_SENTINEL = "fill_sentinel"
    ZERO = "zero"
    FOREACH_COPY = "foreach_copy"


@dataclass
class GraphSlot:
    """A single FB-mirrored buffer.

    Each slot mirrors one ``ForwardBatch`` attribute. ``name`` MUST match
    the FB attribute name so ``fill_from`` can ``getattr(fb, name)`` and
    ``extract_buffer`` can ``setattr`` the view back into a FB replace.

    Fields:
        name           — the FB attribute name mirrored by this slot.
        shape_fn       — ``(max_bs, max_num_tokens) -> shape`` callable
                         used at ``register_slot`` time to allocate the
                         physical buffer.
        dtype          — buffer dtype.
        axis           — ``"bs"`` (slot is sliced ``[:bs]``) or
                         ``"tokens"`` (sliced ``[:num_tokens]``) or
                         ``"none"`` (no slicing — full buffer always
                         exposed; used for scalar buffers and global
                         counters).
        device         — buffer device. ``None`` means use registry
                         default; can be ``"cpu"`` for slots like
                         ``seq_lens_cpu`` that must live on host.
        padding_policy — see ``PaddingPolicy``.
        pad_value      — sentinel for ``FILL_SENTINEL``.
        enabled        — runtime gate; disabled slots are not allocated
                         and skipped during fill / extract.
        post_fill      — optional ``(buffer, forward_batch, raw_n,
                         padded_n) -> None`` hook run after the
                         grouped copy. Used for compute-then-write
                         slots like the local-num-token-non-padded
                         transform.
        slice_fn       — optional ``(buffer, padded_n) -> Tensor``
                         override for slots with non-trivial slicing
                         (e.g. ``mrope_positions`` shape ``[3, T]`` is
                         sliced on axis 1 not 0).
    """

    name: str
    shape_fn: Callable[[int, int], Tuple[int, ...]]
    dtype: torch.dtype
    axis: str = "tokens"
    device: Optional[torch.device] = None
    padding_policy: PaddingPolicy = PaddingPolicy.FOREACH_COPY
    pad_value: Optional[Any] = None
    enabled: bool = True
    post_fill: Optional[Callable[[torch.Tensor, "ForwardBatch", int, int], None]] = None
    slice_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None

    # runtime
    buffer: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.axis not in ("bs", "tokens", "none"):
            raise ValueError(
                f"GraphSlot {self.name!r}: axis must be one of "
                f"'bs'/'tokens'/'none', got {self.axis!r}"
            )

    def _padded_n(self, padded_bs: int, padded_num_tokens: int) -> int:
        if self.axis == "bs":
            return padded_bs
        if self.axis == "tokens":
            return padded_num_tokens
        # axis == "none": no slicing
        return self.buffer.shape[0] if self.buffer is not None else 0

    def _raw_n(self, raw_bs: int, raw_num_tokens: int) -> int:
        if self.axis == "bs":
            return raw_bs
        if self.axis == "tokens":
            return raw_num_tokens
        return self.buffer.shape[0] if self.buffer is not None else 0

    def view(self, padded_bs: int, padded_num_tokens: int) -> torch.Tensor:
        """Return the ``[:padded_n]`` slice consumed by callers."""
        if self.buffer is None:
            raise RuntimeError(f"GraphSlot {self.name!r}: buffer not allocated")
        if self.slice_fn is not None:
            return self.slice_fn(
                self.buffer, self._padded_n(padded_bs, padded_num_tokens)
            )
        if self.axis == "none":
            return self.buffer
        return self.buffer[: self._padded_n(padded_bs, padded_num_tokens)]

    def reset_padding(self, raw_n: int, padded_n: int) -> None:
        """Reset the padded tail according to ``padding_policy``."""
        if self.buffer is None or raw_n >= padded_n:
            return
        if self.padding_policy in (PaddingPolicy.KEEP_PAD, PaddingPolicy.FOREACH_COPY):
            return
        # slice_fn governs non-trivial layouts (e.g. mrope_positions [3, T]);
        # the pad region is the same axis the slot exposes via view().
        if self.slice_fn is not None:
            # slice_fn returns the [:padded_n] portion already; we need the
            # tail [raw_n:padded_n]. We rely on slice_fn slicing the same
            # axis used by view(): take the padded view first, then index
            # the tail with the standard slice on axis 0 of the result.
            padded_view = self.slice_fn(self.buffer, padded_n)
            tail = (
                padded_view[..., raw_n:padded_n]
                if padded_view.dim() > 1
                else padded_view[raw_n:padded_n]
            )
        else:
            tail = self.buffer[raw_n:padded_n]
        if self.padding_policy == PaddingPolicy.FILL_SENTINEL:
            if self.pad_value is None:
                raise RuntimeError(
                    f"GraphSlot {self.name!r}: FILL_SENTINEL requires pad_value"
                )
            tail.fill_(self.pad_value)
        elif self.padding_policy == PaddingPolicy.ZERO:
            tail.zero_()


class CudaGraphBufferRegistry:
    """FB → graph-resident buffer mirror, shared across eager / capture / replay.

    The registry holds a dict of ``GraphSlot`` instances, each mirroring
    one ``ForwardBatch`` attribute. Slots are registered up-front (during
    runner init), allocated lazily on first ``register_slot`` call, then
    filled per-iter via ``fill_from(fb, ...)`` and consumed via
    ``extract_buffer(template) -> ForwardBatch``.

    The registry is also a **cross-stream isolation boundary** — the
    schedule_stream owns the FB tensors; ``fill_from`` issues D2D copies
    on the forward_stream so the registry buffer becomes the
    forward_stream's exclusive view of FB-shared inputs. This removes the
    need for FB-shared-field clones on stream handoff.

    Backend-private buffers (kernel workspace, derived page tables) are
    NOT managed here — backends keep them on ``self.cuda_graph_*`` and
    allocate via ``AttentionBackend.init_cuda_graph_state(...)``.

    Usage::

        registry = CudaGraphBufferRegistry(device=..., max_bs=..., max_num_tokens=...)
        registry.register_slot(GraphSlot(name="input_ids", ...))
        registry.register_slot(GraphSlot(name="seq_lens",
                                         padding_policy=PaddingPolicy.FILL_SENTINEL,
                                         pad_value=seq_len_fill_value, ...))
        # per-iter:
        registry.fill_from(fb, raw_bs=..., padded_bs=..., raw_num_tokens=...,
                           padded_num_tokens=...)
        fb_view = registry.extract_buffer(padded_bs=..., padded_num_tokens=...,
                                          forward_batch_template=fb)
        attn_backend.init_forward_metadata(fb_view)
        model.forward(fb_view.input_ids, fb_view.positions, fb_view)
    """

    def __init__(
        self,
        *,
        device: torch.device,
        max_bs: int,
        max_num_tokens: int,
        share_pool: bool = False,
    ) -> None:
        self.device = device
        self.max_bs = max_bs
        self.max_num_tokens = max_num_tokens
        # When True, slot buffers are coalesced by name through the global
        # ForwardInputBuffers pool, so a registry can share physical storage
        # (and data_ptr) with the legacy DecodeInputBuffers during migration.
        self.share_pool = share_pool
        self._slots: Dict[str, GraphSlot] = {}

    # ---- registration ------------------------------------------------------

    def register_slot(
        self, slot: GraphSlot, bind: Optional[torch.Tensor] = None
    ) -> GraphSlot:
        """Register a slot and allocate (or adopt) its physical buffer.

        If ``bind`` is given, the slot adopts that existing tensor instead of
        allocating a fresh one (and skips the pool / sentinel init — the bound
        tensor is assumed already initialized). This lets a registry share
        storage with the legacy ``DecodeInputBuffers`` by adopting its fields,
        guaranteeing a stable, identical ``data_ptr`` for capture vs replay.

        Returns the slot for caller convenience. Re-registering an existing
        name raises.
        """
        if slot.name in self._slots:
            raise ValueError(
                f"GraphSlot {slot.name!r} already registered; "
                "use enable()/disable() to gate per-iter."
            )
        if not slot.enabled:
            # Even when disabled, keep the spec so callers can introspect
            # by name; just don't allocate.
            self._slots[slot.name] = slot
            return slot
        shape = slot.shape_fn(self.max_bs, self.max_num_tokens)
        device = slot.device if slot.device is not None else self.device
        if bind is not None:
            expected = tuple(shape)
            if tuple(bind.shape) != expected:
                raise ValueError(
                    f"bind tensor for slot {slot.name!r} has shape "
                    f"{tuple(bind.shape)}, expected {expected}."
                )
            if bind.dtype != slot.dtype:
                raise ValueError(
                    f"bind tensor for slot {slot.name!r} has dtype {bind.dtype}, "
                    f"expected {slot.dtype}."
                )
            slot.buffer = bind
            self._slots[slot.name] = slot
            return slot
        buffer = torch.zeros(shape, dtype=slot.dtype, device=device)
        if self.share_pool:
            # Coalesce with any same-named buffer (e.g. the legacy
            # DecodeInputBuffers field) so capture and replay see one
            # physical allocation with a stable data_ptr.
            buffer = share_input_buffer(slot.name, buffer)
        if (
            slot.padding_policy == PaddingPolicy.FILL_SENTINEL
            and slot.pad_value is not None
        ):
            buffer.fill_(slot.pad_value)
        slot.buffer = buffer
        self._slots[slot.name] = slot
        return slot

    def has_slot(self, name: str) -> bool:
        return name in self._slots and self._slots[name].enabled

    def get_slot(self, name: str) -> GraphSlot:
        return self._slots[name]

    def slot_names(self) -> List[str]:
        return [name for name, s in self._slots.items() if s.enabled]

    # ---- per-iter ----------------------------------------------------------

    def fill_from(
        self,
        forward_batch: "ForwardBatch",
        *,
        raw_bs: int,
        padded_bs: int,
        raw_num_tokens: int,
        padded_num_tokens: int,
    ) -> None:
        """Copy FB → registry buffers.

        Phase 1 — reset the padded tail per slot ``padding_policy``.
        Phase 2 — grouped D2D copy of all enabled slots from FB.
        Phase 3 — run ``post_fill`` hooks for slots that need
                  post-copy transforms.

        Slots whose FB attribute is ``None`` are silently skipped (the
        FB doesn't carry that field for the current request).
        """
        # Phase 1: reset padded regions where it matters.
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None:
                continue
            raw_n = slot._raw_n(raw_bs, raw_num_tokens)
            padded_n = slot._padded_n(padded_bs, padded_num_tokens)
            slot.reset_padding(raw_n, padded_n)

        # Phase 2: collect (dst, src) pairs and dispatch a grouped copy.
        gpu_dsts: List[torch.Tensor] = []
        gpu_srcs: List[torch.Tensor] = []
        cpu_dsts: List[torch.Tensor] = []
        cpu_srcs: List[torch.Tensor] = []
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None:
                continue
            src = getattr(forward_batch, slot.name, None)
            if src is None:
                continue
            if not isinstance(src, torch.Tensor):
                # Non-tensor FB fields (e.g. dicts, dataclasses) are not
                # auto-copied — caller handles via custom slot or
                # post_fill hook.
                continue
            raw_n = slot._raw_n(raw_bs, raw_num_tokens)
            if slot.slice_fn is not None:
                dst = slot.slice_fn(slot.buffer, raw_n)
            elif slot.axis == "none":
                dst = slot.buffer
            else:
                dst = slot.buffer[:raw_n]
            # foreach_copy_ requires same-device tensors per call — bucket
            # by device.
            if dst.device.type == "cpu":
                cpu_dsts.append(dst)
                cpu_srcs.append(src)
            else:
                gpu_dsts.append(dst)
                gpu_srcs.append(src)
        if gpu_dsts:
            _grouped_foreach_copy_(gpu_dsts, gpu_srcs)
        for dst, src in zip(cpu_dsts, cpu_srcs):
            dst.copy_(src)

        # Phase 3: post-fill hooks.
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None or slot.post_fill is None:
                continue
            raw_n = slot._raw_n(raw_bs, raw_num_tokens)
            padded_n = slot._padded_n(padded_bs, padded_num_tokens)
            slot.post_fill(slot.buffer, forward_batch, raw_n, padded_n)

    def extract_buffer(
        self,
        *,
        padded_bs: int,
        padded_num_tokens: int,
        forward_batch_template: "ForwardBatch",
    ) -> "ForwardBatch":
        """Return a FB view backed by registry slot buffers.

        ``forward_batch_template`` provides the non-slot fields
        (``forward_mode`` / ``spec_info`` / ``sampling_info`` /
        ``capture_hidden_mode`` / ``dp_*`` / ``lora_ids`` / ...). Slot
        fields are replaced with views into the registry buffers via
        ``dataclasses.replace`` — the template itself is not mutated.
        """
        import dataclasses

        replace_kwargs: Dict[str, Any] = {"batch_size": padded_bs}
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None:
                continue
            replace_kwargs[slot.name] = slot.view(padded_bs, padded_num_tokens)
        return dataclasses.replace(forward_batch_template, **replace_kwargs)


def build_decode_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    seq_len_fill_value: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool = False,
    share_pool: bool = True,
    source: Optional[Any] = None,
) -> CudaGraphBufferRegistry:
    """Registry mirroring the always-on (+ mamba / mrope) FB-shared decode
    buffers, with padding policies matching
    ``DecodeInputBuffers.populate_from_forward_batch``:

      - ``seq_lens`` / ``seq_lens_cpu`` -> FILL_SENTINEL(seq_len_fill_value)
      - ``req_pool_indices`` / ``out_cache_loc`` / ``mamba_track_*`` -> ZERO
      - ``input_ids`` / ``positions`` / ``mrope_positions`` -> FOREACH_COPY
        (head ``[:raw_n]`` is always overwritten by the copy; the old code's
        full-buffer ``zero_()`` / ``fill_()`` on ``bs != raw_bs`` is therefore
        equivalent to the tail-only reset the policies apply here).

    Computed / structured / init-fill-without-per-iter-reset fields
    (``num_token_non_padded``, ``global_num_tokens_*``, ``encoder_lens``,
    ``pp_proxy_tensors``, ``ngram_embedding_info``, ``custom_mask``,
    ``next_token_logits_buffer``, ``input_embeds``, canary ids) are *not*
    registered here; the runner fills them out-of-band until the registry
    grows the matching hooks.

    When ``source`` is given, each slot adopts the same-named tensor off
    ``source`` (e.g. a ``DecodeInputBuffers``) instead of allocating, so the
    registry shares one physical allocation with that object.
    """
    reg = CudaGraphBufferRegistry(
        device=device,
        max_bs=max_bs,
        max_num_tokens=max_num_token,
        share_pool=share_pool,
    )

    def _tokens(_bs: int, mt: int) -> Tuple[int, ...]:
        return (mt,)

    def _bs(bs: int, _mt: int) -> Tuple[int, ...]:
        return (bs,)

    slots = [
        GraphSlot("input_ids", _tokens, torch.int64, axis="tokens"),
        GraphSlot("positions", _tokens, torch.int64, axis="tokens"),
        GraphSlot(
            "out_cache_loc",
            _tokens,
            cache_loc_dtype,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "req_pool_indices",
            _bs,
            torch.int64,
            axis="bs",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "seq_lens",
            _bs,
            torch.int32,
            axis="bs",
            padding_policy=PaddingPolicy.FILL_SENTINEL,
            pad_value=seq_len_fill_value,
        ),
        GraphSlot(
            "seq_lens_cpu",
            _bs,
            torch.int32,
            axis="bs",
            device=torch.device("cpu"),
            padding_policy=PaddingPolicy.FILL_SENTINEL,
            pad_value=seq_len_fill_value,
        ),
        GraphSlot(
            "mrope_positions",
            lambda _bs2, mt: (3, mt),
            torch.int64,
            axis="tokens",
            slice_fn=lambda buf, n: buf[:, :n],
        ),
    ]
    if enable_mamba_track:
        slots.append(
            GraphSlot(
                "mamba_track_indices",
                _bs,
                torch.int64,
                axis="bs",
                padding_policy=PaddingPolicy.ZERO,
            )
        )
        slots.append(
            GraphSlot(
                "mamba_track_mask",
                _bs,
                torch.bool,
                axis="bs",
                padding_policy=PaddingPolicy.ZERO,
            )
        )

    for slot in slots:
        bind = None
        if source is not None:
            bind = getattr(source, slot.name, None)
            if bind is None:
                raise ValueError(
                    f"source is missing buffer {slot.name!r} required by the "
                    "decode registry; cannot adopt."
                )
        reg.register_slot(slot, bind=bind)
    return reg
