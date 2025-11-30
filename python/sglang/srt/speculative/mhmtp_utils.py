"""MHMTP Speculative Decoding Data Structures.

This module defines data structures for MHMTP speculative decoding algorithm,
which extends EAGLE's verification and draft input with additional state for
multi-step token generation and tree-based verification.

Key Classes:
- MhmtpDraftInput: Speculative decoding draft input with multi-step state
- MhmtpVerifyInput: Input for target model verification phase
- MhmtpVerifyOutput: Output from target model verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.spec_info import SpecInputType

if TYPE_CHECKING:
    pass


@dataclass
class MhmtpDraftInput(EagleDraftInput):
    """Draft input for MHMTP speculative decoding with multi-step state.

    Extends EagleDraftInput with additional fields needed for multi-step
    draft token generation. Maintains state across multiple forward passes
    to enable efficient batch processing of token candidate trees.

    Attributes:
        pre_hiddens: Previous hidden states from last forward pass.
            Shape: (batch_size * 4, hidden_size)
            Used for sliding window context in multi-step generation.

        pre_verify_ids: Previously verified token IDs.
            Shape: (batch_size * 4,)
            Maintains recent token history for next step input construction.

        pre_out_locs: Output cache locations from previous steps.
            Type: List[List[torch.Tensor]]
            Structure: [step_index][batch_index]
            Tracks KV cache locations for efficient cache management.

        last_hidden_states: Hidden states from last step in each position.
            Type: List[List[torch.Tensor]]
            Structure: [step_index][batch_index]
            Maintains sliding window of hidden states for context.
    """

    # Multi-step state tensors
    pre_hiddens: Optional[torch.Tensor] = None
    """Previous hidden states from last forward pass (batch*4, hidden_size)."""

    pre_verify_ids: Optional[torch.Tensor] = None
    """Previously verified token IDs (batch*4,)."""

    # Cache location tracking
    pre_out_locs: List[List[torch.Tensor]] = field(default_factory=list)
    """Output cache locations per step and batch element."""

    # Hidden state history
    last_hidden_states: List[List[torch.Tensor]] = field(default_factory=list)
    """Hidden states history for sliding window context."""

    def __post_init__(self):
        self.spec_input_type = SpecInputType.MHMTP_DRAFT

    def prepare_for_draft_extend(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ) -> None:
        """Prepare batch state for draft extend phase.

        Configures the batch with state needed for multi-step draft token
        generation. Updates sequence lengths and request pool indices based
        on previous verification results.

        Args:
            batch: The schedule batch to prepare.
            speculative_num_steps: Number of speculative steps (typically 3).
        """
        # Use verified token IDs as input for draft extend
        batch.input_ids = self.verified_id

        # Calculate extended lengths (verified tokens + 1 for new prediction)
        batch.extend_lens = [x + 1 for x in batch.spec_info.accept_length_cpu]
        batch.extend_num_tokens = sum(batch.extend_lens)

        # Update sequence lengths for draft model
        batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend

        # Update request pool indices for draft extend
        batch.req_pool_indices = batch.spec_info.req_pool_indices_for_draft_extend

        # Disable logprob calculation during draft extend
        batch.return_logprob = False

    def merge_batch(self, spec_info: MhmtpDraftInput) -> None:
        """Merge another draft input into this one.

        Combines two draft inputs, concatenating tensors and merging
        list-based state. Used when batching multiple requests together.

        Args:
            spec_info: The draft input to merge.
        """
        # Merge parent class state
        super().merge_batch(spec_info)

        # Merge cache locations (list of lists)
        self.pre_out_locs = [
            self.pre_out_locs[i] + spec_info.pre_out_locs[i]
            for i in range(len(self.pre_out_locs))
        ]

        # Merge hidden state history (list of lists)
        self.last_hidden_states = [
            self.last_hidden_states[i] + spec_info.last_hidden_states[i]
            for i in range(len(self.last_hidden_states))
        ]


@dataclass
class MhmtpVerifyInput(EagleVerifyInput):
    """Input for target model verification phase in MHMTP speculative decoding.

    Prepares draft token candidates for verification by target model.
    Stores tree structure information and maintains state for verifying
    multiple candidate sequences in a single batch.

    Extends EagleVerifyInput with MHMTP-specific state tracking for
    multi-step generation.

    Attributes:
        pre_hiddens: Hidden states before verification (batch*4, hidden_size).
        pre_verify_ids: Verified token IDs from previous steps (batch*4,).
        pre_out_locs: Cache locations per step and batch (List[torch.Tensor]).
        last_hidden_states: Hidden state history per step (List[torch.Tensor]).
    """

    # Hidden state and verification tracking
    pre_hiddens: Optional[torch.Tensor] = None
    """Hidden states before verification for sliding window context."""

    pre_verify_ids: Optional[torch.Tensor] = None
    """Previously verified token IDs for context."""

    # Cache and history tracking
    pre_out_locs: List[torch.Tensor] = field(default_factory=list)
    """Output cache locations from previous steps."""

    last_hidden_states: List[torch.Tensor] = field(default_factory=list)
    """Hidden state history for sliding window."""

    def __post_init__(self):
        self.spec_input_type = SpecInputType.MHMTP_VERIFY

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,
    ) -> MhmtpVerifyOutput:
        """Verify draft tokens using target model logits.

        Runs the EAGLE verification algorithm and converts the result
        to MhmtpVerifyOutput with additional state tracking.

        Args:
            batch: The schedule batch being verified.
            logits_output: Logits from target model forward pass.
            token_to_kv_pool_allocator: KV cache allocator for managing memory.
            page_size: Page size for KV cache (typically 16).
            vocab_mask: Optional grammar mask for vocabulary constraint.

        Returns:
            MhmtpVerifyOutput with verification results and draft input
            for next iteration.
        """
        # Run parent EAGLE verification
        eagle_verify_output = super().verify(
            batch,
            logits_output,
            token_to_kv_pool_allocator,
            page_size,
            vocab_mask,
        )

        # Convert to MHMTP-specific output type
        mhmtp_verify_output = eagle_verify_output
        mhmtp_verify_output.__class__ = MhmtpVerifyOutput

        # Extract EAGLE draft input and convert to MHMTP format
        eagle_draft_input = mhmtp_verify_output.draft_input
        mhmtp_draft_input = self._create_mhmtp_draft_input(eagle_draft_input, batch)

        mhmtp_verify_output.draft_input = mhmtp_draft_input
        return mhmtp_verify_output

    def _create_mhmtp_draft_input(
        self,
        eagle_draft_input: EagleDraftInput,
        batch: ScheduleBatch,
    ) -> MhmtpDraftInput:
        """Convert EAGLE draft input to MHMTP draft input.

        Maps EAGLE output to MHMTP-specific fields, preserving all
        necessary state for multi-step generation.

        Args:
            eagle_draft_input: Output from EAGLE verification.
            batch: The schedule batch with previous state.

        Returns:
            MhmtpDraftInput with all required fields initialized.
        """
        mhmtp_draft_input = MhmtpDraftInput()

        # Copy core verification results
        mhmtp_draft_input.hidden_states = eagle_draft_input.hidden_states
        mhmtp_draft_input.verified_id = eagle_draft_input.verified_id
        mhmtp_draft_input.accept_length = eagle_draft_input.accept_length
        mhmtp_draft_input.accept_length_cpu = eagle_draft_input.accept_length_cpu

        # Copy draft extend metadata (if available)
        if hasattr(eagle_draft_input, "seq_lens_for_draft_extend"):
            mhmtp_draft_input.seq_lens_for_draft_extend = (
                eagle_draft_input.seq_lens_for_draft_extend
            )

        if hasattr(eagle_draft_input, "req_pool_indices_for_draft_extend"):
            mhmtp_draft_input.req_pool_indices_for_draft_extend = (
                eagle_draft_input.req_pool_indices_for_draft_extend
            )

        # Copy request state (for handling finished requests)
        if hasattr(eagle_draft_input, "unfinished_index_device"):
            mhmtp_draft_input.unfinished_index_device = (
                eagle_draft_input.unfinished_index_device
            )

        if hasattr(eagle_draft_input, "unfinished_index"):
            mhmtp_draft_input.unfinished_index = eagle_draft_input.unfinished_index

        # Copy KV cache metadata
        mhmtp_draft_input.kv_indices = eagle_draft_input.kv_indices
        mhmtp_draft_input.kv_indptr = eagle_draft_input.kv_indptr

        # Copy top-k selection results
        mhmtp_draft_input.topk_p = eagle_draft_input.topk_p
        mhmtp_draft_input.topk_index = eagle_draft_input.topk_index

        # Copy generation metadata
        mhmtp_draft_input.capture_hidden_mode = eagle_draft_input.capture_hidden_mode

        # Copy MHMTP-specific state from batch
        mhmtp_draft_input.pre_out_locs = batch.spec_info.pre_out_locs
        mhmtp_draft_input.last_hidden_states = batch.spec_info.last_hidden_states
        mhmtp_draft_input.pre_hiddens = batch.spec_info.pre_hiddens
        mhmtp_draft_input.pre_verify_ids = batch.spec_info.pre_verify_ids

        return mhmtp_draft_input

    @staticmethod
    def _safe_cat(
        tensor_a: Optional[torch.Tensor],
        tensor_b: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Safely concatenate two tensors, handling None values.

        Args:
            tensor_a: First tensor or None.
            tensor_b: Second tensor or None.

        Returns:
            Concatenated tensor, or the non-None tensor if one is None.
        """
        if tensor_a is not None and tensor_b is not None:
            return torch.cat([tensor_a, tensor_b])
        elif tensor_a is not None:
            return tensor_a
        else:
            return tensor_b

    def merge_batch(self, spec_info: MhmtpVerifyInput) -> None:
        """Merge another verification input into this one.

        Combines two verification inputs, concatenating tree information
        and merging batch state. Used when batching multiple requests.

        Args:
            spec_info: The verification input to merge.
        """
        # Safely concatenate tensor fields
        self.draft_token = self._safe_cat(self.draft_token, spec_info.draft_token)
        self.custom_mask = self._safe_cat(self.custom_mask, spec_info.custom_mask)
        self.positions = self._safe_cat(self.positions, spec_info.positions)

        # Merge tree structure with adjusted indices
        self.retrive_index = self._safe_cat(
            self.retrive_index,
            spec_info.retrive_index + self.retrive_index[-1][-1] + 1,
        )
        self.retrive_next_sibling = self._safe_cat(
            self.retrive_next_sibling, spec_info.retrive_next_sibling
        )
        self.retrive_next_token = self._safe_cat(
            self.retrive_next_token, spec_info.retrive_next_token
        )

        # Merge MHMTP-specific state (list-based)
        self.pre_out_locs = [
            self.pre_out_locs[i] + spec_info.pre_out_locs[i]
            for i in range(len(self.pre_out_locs))
        ]
        self.last_hidden_states = [
            self.last_hidden_states[i] + spec_info.last_hidden_states[i]
            for i in range(len(self.last_hidden_states))
        ]

        # Merge tensor-based state (concatenate)
        self.pre_hiddens = torch.cat([self.pre_hiddens, spec_info.pre_hiddens], dim=0)
        self.pre_verify_ids = torch.cat(
            [self.pre_verify_ids, spec_info.pre_verify_ids], dim=0
        )

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        has_been_filtered: bool = True,
    ) -> None:
        """Filter batch by keeping only specified indices.

        This operation is typically handled by parent class verification.
        Currently a no-op for MHMTP as filtering is managed elsewhere.

        Args:
            new_indices: Indices to keep.
            has_been_filtered: Whether batch has already been filtered.
        """
        # Filtering handled by parent class or batch manager
        pass


@dataclass
class MhmtpVerifyOutput(EagleVerifyOutput):
    """Output from target model verification in MHMTP speculative decoding.

    Contains verification results and provides the draft input for the next
    iteration of speculative generation.

    Extends EagleVerifyOutput with MHMTP-specific draft input that includes
    multi-step state tracking for efficient candidate generation.

    Attributes:
        draft_input: MhmtpDraftInput for next draft extend phase.
    """

    draft_input: MhmtpDraftInput
    """Draft input prepared for next speculative generation iteration."""
