"""Test-only Python reference for the canary plan kernel.

Defines :class:`BatchPlanGpu`, the device-side fixed-shape plan-tensor
bundle that connects the plan kernel to :func:`canary_step
<sglang.jit_kernel.kv_cache_canary.canary_step>`. The plan kernel
produces one instance per forward; ``canary_step`` consumes it as a
single dataclass argument instead of unpacking ~13 individual tensors at
every call site.

Nothing in this module is on the production hot path; the dataclass is
imported by ``canary_step`` (host wrapper) and by the host-state code
that fills the buffers in-place each forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPlanGpu:
    """Device-side fixed-shape plan tensors connecting plan kernel → canary_step.

    Produced by :func:`sglang.jit_kernel.kv_cache_canary_plan.plan_batch_from_forward_batch`
    once per forward and consumed verbatim by
    :func:`sglang.jit_kernel.kv_cache_canary.canary_step`.

    Plan layout — three parallel tile sets describing the verify + write work for one canary
    launch:

    - **Per-verify-entry** (active length ``verify_num_valid[0]``; capacity =
      ``verify_slot_indices.shape[0]``): ``(slot_idx, position, prev_slot_idx)`` for each slot
      the kernel verifies. ``prev_slot_idx == -1`` flags a chain-seed entry (position 0 of a
      req, or the head of an SWA window — anchor on ``CanaryConfig.seed`` instead of a
      predecessor slot).
    - **Per-write-entry** (active length implied by per-write-req ``entry_start`` /
      ``entry_count``; capacity = ``write_slot_indices.shape[0]``): ``(slot_idx, token_id,
      position)`` for each slot the kernel fingerprints, flattened across reqs in the order
      they appear in the source ``ForwardBatch.req_pool_indices``. ``(expected_write_token_ids,
      expected_write_positions)`` are the pseudo-mode oracle predictions per write entry; both
      filled with ``-1`` skip-sentinel in real-mode (the kernel pays no per-entry cost on the
      skip path).
    - **Per-write-req** (active length ``write_req_num_valid[0]``; capacity =
      ``write_req_seed_slot_indices.shape[0]``): ``(seed_slot_idx, entry_start, entry_count)``
      per req that has at least one write entry. ``seed_slot_idx == -1`` flags
      ``K_req_old == 0`` (chain anchors on ``seed``). ``entry_start`` is the exclusive
      prefix-sum offset into the per-write-entry arrays.

    Every per-entry tensor is sized to a **cuda-graph-captured fixed maximum**; the active
    prefix is reported via the ``verify_num_valid`` / ``write_req_num_valid`` scalars. Padding
    tail entries (indices ``>= num_valid``) are unspecified — ``canary_step`` early-exits via
    ``tid >= num_valid[0]`` before reading them. The per-write-entry tile has no separate
    ``write_num_valid`` because writes are gated through the per-write-req tile via
    ``entry_start`` / ``entry_count``.

    Lifecycle:

    - Allocate once at runner init via :meth:`allocate`; the buffer addresses are baked into
      cuda-graph capture.
    - Refilled in-place every forward by
      :func:`plan_batch_from_forward_batch <sglang.jit_kernel.kv_cache_canary_plan.plan_batch_from_forward_batch>`;
      never reallocated.
    - Read in-place by ``canary_step`` (head + tail launches both consume the same instance,
      and the K-half / V-half launches share the same plan as well).
    - Reset to all-zero / skip-sentinel via :meth:`reset_to_skip_sentinel` at capture-time
      teardown / re-init.

    All fields are ``int32`` so kernel atomic / scan operations stay in int32 throughout;
    callers must not change dtypes without updating both plan kernel and ``canary_step``.

    Fields:
        verify_slot_indices:         ``int32 [verify_capacity]`` — canary buffer slot index of
                                     each verify entry. Already SWA-translated for the SWA
                                     group; see plan kernel docstring for the LUT-ordering rule.
        verify_positions:            ``int32 [verify_capacity]`` — expected sequence position
                                     for each verify entry.
        verify_prev_slot_indices:    ``int32 [verify_capacity]`` — slot index of the chain
                                     predecessor, or ``-1`` for chain-seed entries.
        write_slot_indices:          ``int32 [write_capacity]`` — canary buffer slot index of
                                     each write entry. Already SWA-translated.
        write_token_ids:             ``int32 [write_capacity]`` — token id to fingerprint into
                                     each write slot.
        write_positions:             ``int32 [write_capacity]`` — sequence position to
                                     fingerprint into each write slot.
        expected_write_token_ids:    ``int32 [write_capacity]`` — pseudo-mode oracle's expected
                                     token id; ``-1`` skip-sentinel in real-mode.
        expected_write_positions:    ``int32 [write_capacity]`` — pseudo-mode oracle's expected
                                     sequence position; ``-1`` skip-sentinel in real-mode.
        write_req_seed_slot_indices: ``int32 [write_req_capacity]`` — chain-seed slot per write
                                     req, or ``-1`` when the req has no prefix
                                     (``K_req_old == 0``).
        write_req_entry_starts:      ``int32 [write_req_capacity]`` — exclusive prefix-sum
                                     offset of each write req into the per-write-entry arrays.
        write_req_entry_counts:      ``int32 [write_req_capacity]`` — number of write entries
                                     owned by each write req.
        verify_num_valid:            ``int32 [1]`` — count of active per-verify-entry rows.
                                     ``canary_step`` skips threads with ``tid >=
                                     verify_num_valid[0]``.
        write_req_num_valid:         ``int32 [1]`` — count of active per-write-req rows.
                                     ``canary_step`` skips threads with ``req_tid >=
                                     write_req_num_valid[0]``.
    """

    verify_slot_indices: torch.Tensor
    verify_positions: torch.Tensor
    verify_prev_slot_indices: torch.Tensor

    write_slot_indices: torch.Tensor
    write_token_ids: torch.Tensor
    write_positions: torch.Tensor
    expected_write_token_ids: torch.Tensor
    expected_write_positions: torch.Tensor

    write_req_seed_slot_indices: torch.Tensor
    write_req_entry_starts: torch.Tensor
    write_req_entry_counts: torch.Tensor

    verify_num_valid: torch.Tensor
    write_req_num_valid: torch.Tensor
