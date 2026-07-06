# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FB-shared slot registry for the CUDA graph forward paths.

``CudaGraphBufferRegistry`` is the ForwardBatch → graph-resident buffer mirror
used by capture / replay. It replaces the per-runner ``DecodeInputBuffers`` /
``PrefillInputBuffers`` dataclasses and their hand-written
``populate_from_forward_batch`` methods with a single ``GraphSlot``-driven
registry.

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

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_has_foreach_copy = hasattr(torch, "_foreach_copy_")
_has_triton = triton is not None

if _has_triton:

    @triton.jit
    def _fused_copy_8_kernel(
        dst0,
        src0,
        n0,
        dst1,
        src1,
        n1,
        dst2,
        src2,
        n2,
        dst3,
        src3,
        n3,
        dst4,
        src4,
        n4,
        dst5,
        src5,
        n5,
        dst6,
        src6,
        n6,
        dst7,
        src7,
        n7,
        BLOCK: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK + tl.arange(0, BLOCK)
        if slot == 0:
            mask = offs < n0
            tl.store(dst0 + offs, tl.load(src0 + offs, mask=mask), mask=mask)
        elif slot == 1:
            mask = offs < n1
            tl.store(dst1 + offs, tl.load(src1 + offs, mask=mask), mask=mask)
        elif slot == 2:
            mask = offs < n2
            tl.store(dst2 + offs, tl.load(src2 + offs, mask=mask), mask=mask)
        elif slot == 3:
            mask = offs < n3
            tl.store(dst3 + offs, tl.load(src3 + offs, mask=mask), mask=mask)
        elif slot == 4:
            mask = offs < n4
            tl.store(dst4 + offs, tl.load(src4 + offs, mask=mask), mask=mask)
        elif slot == 5:
            mask = offs < n5
            tl.store(dst5 + offs, tl.load(src5 + offs, mask=mask), mask=mask)
        elif slot == 6:
            mask = offs < n6
            tl.store(dst6 + offs, tl.load(src6 + offs, mask=mask), mask=mask)
        elif slot == 7:
            mask = offs < n7
            tl.store(dst7 + offs, tl.load(src7 + offs, mask=mask), mask=mask)


def _can_fuse_copy(dst: torch.Tensor, src: torch.Tensor) -> bool:
    return (
        _has_triton
        and dst.device.type == "cuda"
        and src.device.type == "cuda"
        and dst.device == src.device
        and dst.dtype == src.dtype
        and dst.numel() == src.numel()
        and dst.is_contiguous()
        and src.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    )


def _fused_copy_chunks_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
    block = 256
    for start in range(0, len(dsts), 8):
        chunk_dsts = dsts[start : start + 8]
        chunk_srcs = srcs[start : start + 8]
        num_slots = len(chunk_dsts)
        max_n = max(dst.numel() for dst in chunk_dsts)
        padded_dsts = chunk_dsts + [chunk_dsts[0]] * (8 - num_slots)
        padded_srcs = chunk_srcs + [chunk_srcs[0]] * (8 - num_slots)
        ns = [dst.numel() for dst in chunk_dsts] + [0] * (8 - num_slots)
        _fused_copy_8_kernel[(num_slots, triton.cdiv(max_n, block))](
            padded_dsts[0],
            padded_srcs[0],
            ns[0],
            padded_dsts[1],
            padded_srcs[1],
            ns[1],
            padded_dsts[2],
            padded_srcs[2],
            ns[2],
            padded_dsts[3],
            padded_srcs[3],
            ns[3],
            padded_dsts[4],
            padded_srcs[4],
            ns[4],
            padded_dsts[5],
            padded_srcs[5],
            ns[5],
            padded_dsts[6],
            padded_srcs[6],
            ns[6],
            padded_dsts[7],
            padded_srcs[7],
            ns[7],
            BLOCK=block,
        )


def _grouped_foreach_copy_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
    """Call torch._foreach_copy_ grouped by (dst_dtype, src_dtype) pairs
    (a single foreach call requires a uniform dtype pair)."""

    def _foreach_copy(
        group_dsts: List[torch.Tensor], group_srcs: List[torch.Tensor]
    ) -> None:
        fused_dsts: List[torch.Tensor] = []
        fused_srcs: List[torch.Tensor] = []
        fallback_dsts: List[torch.Tensor] = []
        fallback_srcs: List[torch.Tensor] = []
        for dst, src in zip(group_dsts, group_srcs):
            if _can_fuse_copy(dst, src):
                fused_dsts.append(dst)
                fused_srcs.append(src)
            else:
                fallback_dsts.append(dst)
                fallback_srcs.append(src)

        if len(fused_dsts) >= 2:
            _fused_copy_chunks_(fused_dsts, fused_srcs)
        elif len(fused_dsts) == 1:
            fallback_dsts.extend(fused_dsts)
            fallback_srcs.extend(fused_srcs)

        if not fallback_dsts:
            return
        if _has_foreach_copy:
            torch._foreach_copy_(fallback_dsts, fallback_srcs)
        else:
            for dst, src in zip(fallback_dsts, fallback_srcs):
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
    FILL_ONCE     — Fill the whole buffer to ``pad_value`` once at alloc;
                    never reset per iter (e.g. ``encoder_lens`` init to
                    ``encoder_len_fill_value``, copied head-only with the
                    tail kept).
    """

    KEEP_PAD = "keep_pad"
    FILL_SENTINEL = "fill_sentinel"
    ZERO = "zero"
    FOREACH_COPY = "foreach_copy"
    FILL_ONCE = "fill_once"


@dataclass
class FillContext:
    """Per-iteration shape context passed to ``GraphSlot.post_fill``.

    Carries both the bs-axis and tokens-axis raw/padded counts so a hook can
    derive values regardless of its own slot's axis — e.g. the padded token
    count (``padded_num_tokens`` == padded_bs * num_tokens_per_bs), which the
    global-num-tokens fill and the local-num-token-non-padded transform need.
    """

    raw_bs: int
    padded_bs: int
    raw_num_tokens: int
    padded_num_tokens: int
    # Side inputs that are not ForwardBatch attributes but are needed by a
    # slot's source_fn — e.g. the pipeline-parallel proxy tensors, which the
    # replay path receives as a separate argument rather than off the FB.
    pp_proxy_tensors: Optional[Any] = None


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
        copy_from_fb   — when ``True`` (default), ``fill_from`` copies the
                         same-named FB tensor into the buffer head. Set
                         ``False`` for computed slots whose value is not a
                         straight FB copy (e.g. ``global_num_tokens_*``,
                         filled by a ``post_fill`` instead).
        post_fill      — optional ``(buffer, forward_batch, FillContext)
                         -> None`` hook run after the grouped copy. Used for
                         compute-then-write slots (local-num-token-non-padded
                         transform, global-num-tokens fill).
        slice_fn       — optional ``(buffer, padded_n) -> Tensor``
                         override for slots with non-trivial slicing
                         (e.g. ``mrope_positions`` shape ``[3, T]`` is
                         sliced on axis 1 not 0).
        source_fn      — optional ``(forward_batch, FillContext) -> Tensor |
                         None`` override for the copy *source*. When set,
                         ``fill_from`` copies ``source_fn(fb, ctx)`` (instead of
                         the same-named FB attribute) into
                         ``buffer[:src.shape[0]]`` — a source-length slice for
                         structured / side-sourced fields whose data lives on a
                         nested FB dataclass (``ngram_embedding_info.*``) or an
                         out-of-band argument (``pp_proxy_tensors``, carried on
                         ``FillContext``). Returning ``None`` skips the copy for
                         that iteration. Such slots use dotted names and are
                         skipped by ``extract_buffer``.
    """

    name: str
    shape_fn: Callable[[int, int], Tuple[int, ...]]
    dtype: torch.dtype
    axis: str = "tokens"
    device: Optional[torch.device] = None
    padding_policy: PaddingPolicy = PaddingPolicy.FOREACH_COPY
    pad_value: Optional[Any] = None
    enabled: bool = True
    copy_from_fb: bool = True
    post_fill: Optional[Callable[[torch.Tensor, ForwardBatch, FillContext], None]] = (
        None
    )
    slice_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None
    source_fn: Optional[
        Callable[[ForwardBatch, FillContext], Optional[torch.Tensor]]
    ] = None

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

    def slice_for(self, padded_bs: int, padded_num_tokens: int) -> torch.Tensor:
        """Return the ``[:padded_n]`` slice of the buffer consumed by callers.

        This truncates the (full-length) buffer to the active region for the
        current iteration — it is a slice, not a tensor reshape.
        """
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
        if self.padding_policy in (
            PaddingPolicy.KEEP_PAD,
            PaddingPolicy.FOREACH_COPY,
            PaddingPolicy.FILL_ONCE,
        ):
            return
        # slice_fn governs non-trivial layouts (e.g. mrope_positions [3, T]);
        # the pad region is the same axis the slot exposes via slice_for().
        if self.slice_fn is not None:
            # slice_fn returns the [:padded_n] portion already; we need the
            # tail [raw_n:padded_n]. We rely on slice_fn slicing the same
            # axis used by slice_for(): take the padded slice first, then index
            # the tail with the standard slice on axis 0 of the result.
            padded_slice = self.slice_fn(self.buffer, padded_n)
            tail = (
                padded_slice[..., raw_n:padded_n]
                if padded_slice.dim() > 1
                else padded_slice[raw_n:padded_n]
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
    runner init), allocated at ``register_slot``, then filled per-iter via
    ``fill_from(fb, ...)`` and consumed via ``extract_buffer(template) ->
    ForwardBatch``. ``fill_from`` issues plain D2D copies on the caller's
    current stream; cross-stream correctness (stream handoff) is handled by
    the runners, not here.

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
        # Coalesce allocated slot buffers through the global pool; only applies
        # when allocating (bind/source bypasses the pool).
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
            slot.padding_policy
            in (PaddingPolicy.FILL_SENTINEL, PaddingPolicy.FILL_ONCE)
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
        forward_batch: ForwardBatch,
        *,
        raw_bs: int,
        padded_bs: int,
        raw_num_tokens: int,
        padded_num_tokens: int,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> None:
        """Copy FB → registry buffers.

        Phase 1 — reset the padded tail per slot ``padding_policy``.
        Phase 2 — grouped D2D copy of all enabled slots from FB (or from a
                  slot's ``source_fn`` for structured / side-sourced fields).
        Phase 3 — run ``post_fill`` hooks for slots that need
                  post-copy transforms.

        ``pp_proxy_tensors`` is the out-of-band pipeline-parallel input; it is
        not an FB attribute, so it reaches ``source_fn`` slots via
        ``FillContext.pp_proxy_tensors``.

        Slots whose FB attribute (or ``source_fn`` result) is ``None`` are
        silently skipped (the FB doesn't carry that field for the current
        request).
        """
        ctx = FillContext(
            raw_bs=raw_bs,
            padded_bs=padded_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=padded_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )

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
            if not slot.enabled or slot.buffer is None or not slot.copy_from_fb:
                continue
            if slot.source_fn is not None:
                # Structured / side-sourced slot: source comes from a nested FB
                # dataclass or an out-of-band input, and the copy is sliced to
                # the source's own length rather than a bs/tokens axis.
                src = slot.source_fn(forward_batch, ctx)
                if src is None:
                    continue
                dst = slot.buffer[: src.shape[0]]
            else:
                src = getattr(forward_batch, slot.name, None)
                if src is None:
                    continue
                if not isinstance(src, torch.Tensor):
                    # Non-tensor FB fields (e.g. dicts, dataclasses) are not
                    # auto-copied — caller handles via source_fn or post_fill.
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

        # Phase 3: post-fill hooks (compute-then-write slots).
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None or slot.post_fill is None:
                continue
            slot.post_fill(slot.buffer, forward_batch, ctx)

    def extract_buffer(
        self,
        *,
        padded_bs: int,
        padded_num_tokens: int,
        forward_batch_template: ForwardBatch,
    ) -> ForwardBatch:
        """Return a FB view (``dataclasses.replace`` of ``forward_batch_template``)
        whose slot fields are buffer views and whose non-slot fields are carried
        from the template. A plain copy slot whose FB field is ``None`` this iter
        is carried (not exposed as a stale buffer); computed slots are always
        exposed.
        """
        import dataclasses

        replace_kwargs: Dict[str, Any] = {"batch_size": padded_bs}
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None:
                continue
            # Structured slots use dotted names ("<field>.<sub>") and are not
            # top-level FB attributes — their data is consumed in place off the
            # adopted backing object, not re-attached to the FB view here.
            if "." in slot.name:
                continue
            is_computed = slot.post_fill is not None or not slot.copy_from_fb
            if (
                not is_computed
                and slot.source_fn is None
                and getattr(forward_batch_template, slot.name, None) is None
            ):
                # Absent this iter (fill_from skipped it): carry the template.
                continue
            replace_kwargs[slot.name] = slot.slice_for(padded_bs, padded_num_tokens)
        return dataclasses.replace(forward_batch_template, **replace_kwargs)


def build_decode_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    seq_len_fill_value: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool = False,
    is_encoder_decoder: bool = False,
    encoder_len_fill_value: int = 0,
    encoder_lens_dtype: torch.dtype = torch.int32,
    enable_num_token_non_padded: bool = False,
    require_gathered_buffer: bool = False,
    enable_prefill_cp: bool = False,
    require_mlp_tp_gather: bool = False,
    dp_size: int = 1,
    register_global_num_tokens: bool = True,
    share_pool: bool = True,
    source: Optional[Any] = None,
) -> CudaGraphBufferRegistry:
    """Registry mirroring the always-on (+ mamba / mrope) FB-shared decode
    buffers, with the per-slot padding policy that resets the padded tail on
    each replay:

      - ``seq_lens`` / ``seq_lens_cpu`` -> FILL_SENTINEL(seq_len_fill_value)
      - ``req_pool_indices`` / ``out_cache_loc`` / ``mamba_track_*`` -> ZERO
      - ``positions`` / ``mrope_positions`` -> ZERO: the flashinfer verify-path
        plan reads the padded tail, so leaving stale out-of-range values there
        triggers an illegal memory access (issue #24361).
      - ``input_ids`` -> FOREACH_COPY: head ``[:raw_n]`` is overwritten by the
        copy and the padded tail is not read.

    ``custom_mask`` / ``next_token_logits_buffer`` / ``input_embeds`` are not
    registered here — they are not per-replay FB copies (allocated and written
    elsewhere), so the runner keeps owning them.

    When ``source`` is given (the decode buffer namespace from
    ``_allocate_decode_buffers``), each slot adopts the same-named tensor off
    it instead of allocating, so the registry shares one physical allocation
    with that object. With ``source=None`` the registry allocates its own.
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
        GraphSlot(
            "positions",
            _tokens,
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
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
            torch.int64,
            axis="bs",
            padding_policy=PaddingPolicy.FILL_SENTINEL,
            pad_value=seq_len_fill_value,
        ),
        GraphSlot(
            "seq_lens_cpu",
            _bs,
            torch.int64,
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
            padding_policy=PaddingPolicy.ZERO,
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
    if is_encoder_decoder:
        # Initialized once to encoder_len_fill_value, copied head-only, never
        # reset per iter — matching the legacy DecodeInputBuffers behavior.
        slots.append(
            GraphSlot(
                "encoder_lens",
                _bs,
                encoder_lens_dtype,
                axis="bs",
                padding_policy=PaddingPolicy.FILL_ONCE,
                pad_value=encoder_len_fill_value,
            )
        )
    if enable_num_token_non_padded:
        from sglang.srt.model_executor.forward_batch_info import (
            compute_local_num_token_non_padded,
        )

        def _num_token_non_padded_post_fill(buf, fb, ctx):
            # Gathered (DP) path overwrites the plain FB copy with this rank's
            # local count; the non-gathered path keeps the copied value.
            if require_gathered_buffer and not enable_prefill_cp:
                buf.copy_(
                    compute_local_num_token_non_padded(
                        global_num_token_non_padded=fb.num_token_non_padded,
                        num_tokens_per_dp=ctx.padded_num_tokens,
                    )
                )

        slots.append(
            GraphSlot(
                "num_token_non_padded",
                lambda _bs, _mt: (1,),
                torch.int32,
                axis="none",
                post_fill=_num_token_non_padded_post_fill,
            )
        )

    # Computed slots, always exposed by extract_buffer; callers that already set
    # global_num_tokens_* on the batch pass register_global_num_tokens=False.
    if register_global_num_tokens:

        def _global_num_tokens_post_fill(buf, fb, ctx):
            # Only the gathered (DP) path writes a value; otherwise left as init.
            if require_gathered_buffer:
                buf.fill_(ctx.padded_num_tokens)

        _global_shape = (
            (lambda _bs, _mt: (dp_size,))
            if require_mlp_tp_gather
            else (lambda _bs, _mt: (1,))
        )
        for _global_name in (
            "global_num_tokens_gpu",
            "global_num_tokens_for_logprob_gpu",
        ):
            slots.append(
                GraphSlot(
                    _global_name,
                    _global_shape,
                    torch.int32,
                    axis="none",
                    copy_from_fb=False,
                    post_fill=_global_num_tokens_post_fill,
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

    # Structured slots whose backing storage still lives on the source object
    # (adopt-only during migration): registered only when the source actually
    # carries them. The per-replay copy source is a nested FB dataclass field,
    # supplied via source_fn; head is copied (source-length slice), tail kept.
    if source is not None:
        ngram = getattr(source, "ngram_embedding_info", None)
        if ngram is not None:

            def _ngram_source(attr):
                def _fn(fb, _ctx):
                    info = getattr(fb, "ngram_embedding_info", None)
                    return None if info is None else getattr(info, attr)

                return _fn

            for _attr in ("column_starts", "req_lens"):
                backing = getattr(ngram, _attr)
                reg.register_slot(
                    GraphSlot(
                        name=f"ngram_embedding_info.{_attr}",
                        shape_fn=lambda _bs, _mt, _s=tuple(backing.shape): _s,
                        dtype=backing.dtype,
                        axis="none",
                        padding_policy=PaddingPolicy.KEEP_PAD,
                        source_fn=_ngram_source(_attr),
                    ),
                    bind=backing,
                )

        # Pipeline-parallel proxy tensors: a dict of per-key buffers, sourced
        # from the out-of-band pp input on FillContext rather than the FB.
        pp = getattr(source, "pp_proxy_tensors", None)
        if pp is not None:

            def _pp_source(key):
                def _fn(_fb, ctx):
                    ppx = ctx.pp_proxy_tensors
                    return None if ppx is None else ppx.tensors[key]

                return _fn

            for _key, _backing in pp.items():
                reg.register_slot(
                    GraphSlot(
                        name=f"pp_proxy_tensors.{_key}",
                        shape_fn=lambda _bs, _mt, _s=tuple(_backing.shape): _s,
                        dtype=_backing.dtype,
                        axis="none",
                        padding_policy=PaddingPolicy.KEEP_PAD,
                        source_fn=_pp_source(_key),
                    ),
                    bind=_backing,
                )

        # KV-canary id buffers (off by default): plain bs-axis FB copies,
        # adopt-only when the source carries them. Head [:raw_bs] is copied;
        # the tail keeps its init (rids_int 0, bootstrap_room_ids_int -1).
        for _cname in ("rids_int", "bootstrap_room_ids_int"):
            canary = getattr(source, _cname, None)
            if canary is not None:
                reg.register_slot(
                    GraphSlot(
                        name=_cname,
                        shape_fn=lambda _bs, _mt, _s=tuple(canary.shape): _s,
                        dtype=canary.dtype,
                        axis="bs",
                    ),
                    bind=canary,
                )

    return reg


def build_prefill_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    cache_loc_dtype: torch.dtype,
    is_multimodal: bool = False,
    hidden_size: int = 0,
    embed_dtype: Optional[torch.dtype] = None,
    enable_mamba_track: bool = False,
    register_input_embeds: bool = True,
    share_pool: bool = True,
    source: Optional[Any] = None,
) -> CudaGraphBufferRegistry:
    """Registry mirroring the **token-axis** FB-shared buffers for the
    piecewise / breakable (prefill) cuda-graph runners.

    ``register_input_embeds`` (default ``True``) registers the multimodal
    ``input_embeds`` slot; the eager extend path passes ``False`` so it is
    carried from the batch (a read input) rather than written in-graph.

    Padding policies match the inline copy/zero in
    ``PiecewiseCudaGraphRunner.load_batch``: ``input_ids`` / ``positions``
    / ``out_cache_loc`` / ``mrope_positions`` / ``input_embeds`` reset their
    padded tail ``[raw_num_tokens:padded_num_tokens]`` to ``0`` (the padded
    tokens *are* processed by the graph, so they must be benign), then the head
    ``[:raw_num_tokens]`` is copied from the FB. ``input_embeds`` is not an FB
    copy — the model writes the embeds into it inside the graph — so it is
    reset-only (``copy_from_fb=False``). ``mamba_track_*`` are bs-axis copies
    with no padding reset (bs is not padded on this path).

    The piecewise / breakable runners pass ``source=None``, so the registry
    allocates (and owns) these buffers directly; ``share_pool`` then coalesces
    them through the process-wide pool. (A ``source`` object, if given, would be
    adopted instead — one shared allocation with stable ``data_ptr``.)
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
        GraphSlot(
            "input_ids",
            _tokens,
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "positions",
            _tokens,
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "out_cache_loc",
            _tokens,
            cache_loc_dtype,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
    ]
    if is_multimodal:
        slots.append(
            GraphSlot(
                "mrope_positions",
                lambda _bs2, mt: (3, mt),
                torch.int64,
                axis="tokens",
                padding_policy=PaddingPolicy.ZERO,
                slice_fn=lambda buf, n: buf[:, :n],
            )
        )
        if register_input_embeds:
            slots.append(
                GraphSlot(
                    "input_embeds",
                    lambda _bs2, mt: (mt, hidden_size),
                    embed_dtype,
                    axis="tokens",
                    padding_policy=PaddingPolicy.ZERO,
                    copy_from_fb=False,
                )
            )
    if enable_mamba_track:
        slots.append(GraphSlot("mamba_track_indices", _bs, torch.int64, axis="bs"))
        slots.append(GraphSlot("mamba_track_mask", _bs, torch.bool, axis="bs"))
        slots.append(GraphSlot("mamba_track_seqlens", _bs, torch.int32, axis="bs"))

    for slot in slots:
        bind = None
        if source is not None:
            bind = getattr(source, slot.name, None)
            if bind is None:
                raise ValueError(
                    f"source is missing buffer {slot.name!r} required by the "
                    "prefill registry; cannot adopt."
                )
        reg.register_slot(slot, bind=bind)
    return reg


def build_eager_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool = False,
    is_encoder_decoder: bool = False,
    encoder_len_fill_value: int = 0,
    encoder_lens_dtype: torch.dtype = torch.int32,
    dp_size: int = 1,
) -> CudaGraphBufferRegistry:
    """One fixed-max input registry for the ``EagerRunner``, serving BOTH eager
    decode and eager prefill.

    The decode slot set is a superset of eager prefill's needs (eager prefill
    carries ``input_embeds`` from the batch and reads the bs-axis fields live),
    so we reuse it, sized at ``(max_bs, max_num_token)`` where ``max_num_token``
    is the prefill token ceiling. ``seq_len_fill_value=0`` because eager never
    pads, so the sentinel tail is never read.

    ``share_pool=True`` so same-named / same-size slots coalesce through the
    process-wide pool. The ``EagerRunner`` is built before the cuda-graph runners
    (see ``ModelRunner.init_backends``), so its (largest) allocations are
    canonical and the cg runners' matching slots (prefill's token-axis at
    ``max_num_token``, decode's bs-axis at ``max_bs``) adopt them.
    """
    return build_decode_registry(
        device=device,
        max_bs=max_bs,
        max_num_token=max_num_token,
        seq_len_fill_value=0,
        cache_loc_dtype=cache_loc_dtype,
        enable_mamba_track=enable_mamba_track,
        is_encoder_decoder=is_encoder_decoder,
        encoder_len_fill_value=encoder_len_fill_value,
        encoder_lens_dtype=encoder_lens_dtype,
        enable_num_token_non_padded=False,
        register_global_num_tokens=False,
        require_gathered_buffer=False,
        require_mlp_tp_gather=False,
        dp_size=dp_size,
        share_pool=True,
        source=None,
    )
