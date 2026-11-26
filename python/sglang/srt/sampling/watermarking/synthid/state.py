# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
# Modified by the SGLang Team: adapted from google-deepmind/synthid-text@addb4a15
# (src/synthid_text/logits_processing.py and src/synthid_text/hashing_function.py).
"""SynthID state per request slot.

A slot holds the last ``ngram_len - 1`` committed tokens (the context), a ring
of recent context hashes (a context seen there is not watermarked again), and
the number of tokens committed so far.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from sglang.srt.sampling.watermarking.synthid.algorithm import (
    SynthIDConfig,
    hash_context,
    hash_contexts,
)


class SynthIDStatePool:
    """SynthID state for every request slot, on one device: Triton on CUDA; the
    CPU torch path is the reference the kernel tests compare against."""

    def __init__(
        self,
        config: SynthIDConfig,
        num_slots: int,
        device: str | torch.device,
    ):
        self.config = config
        self.context = torch.zeros(
            (num_slots, config.context_width), dtype=torch.int64, device=device
        )
        self.context_history = torch.zeros(
            (num_slots, config.context_history_size),
            dtype=torch.int64,
            device=device,
        )
        self.history_cursor = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self.num_calls = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self.use_triton = self.context.is_cuda

    def seed(
        self,
        slots: torch.Tensor,
        replay_token_ids: Sequence[Sequence[int]],
    ) -> None:
        """Start each slot over, then commit its replayed tokens.

        ``slots`` is ``[N]`` on the pool's device. Row ``i`` of
        ``replay_token_ids`` lists the tokens a resumed request already
        generated (all of them, so the history ring and its cursor match the
        uninterrupted run); an empty row is a fresh request with a zero context.
        """
        slots = slots.to(torch.int64)
        self.context.index_fill_(0, slots, 0)
        self.context_history.index_fill_(0, slots, 0)
        self.history_cursor.index_fill_(0, slots, 0)
        self.num_calls.index_fill_(0, slots, 0)

        replay_rows = [row for row, tokens in enumerate(replay_token_ids) if tokens]
        if not replay_rows:
            return
        replayed = [self._replay(replay_token_ids[row]) for row in replay_rows]
        device = self.context.device
        replay_slots = slots[torch.tensor(replay_rows, device=device)]
        contexts, histories, cursors, call_counts = zip(*replayed)
        self.context[replay_slots] = torch.tensor(contexts, device=device)
        self.context_history[replay_slots] = torch.tensor(histories, device=device)
        self.history_cursor[replay_slots] = torch.tensor(cursors, device=device)
        self.num_calls[replay_slots] = torch.tensor(call_counts, device=device)

    def _replay(
        self, token_ids: Sequence[int]
    ) -> tuple[list[int], list[int], int, int]:
        config = self.config
        context = [0] * config.context_width
        history = [0] * config.context_history_size
        cursor = 0
        num_calls = 0
        for token_id in token_ids:
            if not config.skip_first_ngram_calls or num_calls + 1 >= config.ngram_len:
                history[cursor] = hash_context(context, config.hash_iv)
                cursor = (cursor + 1) % config.context_history_size
            context = context[1:] + [int(token_id)]
            num_calls += 1
        return context, history, cursor, num_calls

    def current(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Context hashes ``[N]`` and apply mask ``[N]`` for each slot's next token."""
        if self.use_triton:
            # A one-column chain scores the next token from the committed state;
            # its column 0 is never read.
            return self.project(slots, slots[:, None])
        context = self.context[slots]
        history = self.context_history[slots]
        return self._hashes_and_mask(context, history, self.num_calls[slots])

    def project(
        self,
        slots: torch.Tensor,
        verify_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Context hashes and apply masks along a linear chain, without committing.

        ``verify_tokens`` is ``[N, W]``: column 0 is the last committed token
        (already in the context) and columns ``1..W-1`` are draft tokens.
        Position ``j`` is scored as if columns ``1..j`` had been committed.
        Returns two ``[N * W]`` tensors in row-major order.
        """
        if self.use_triton:
            from sglang.kernels.ops.sampling.synthid_triton import (
                synthid_project_state_triton,
            )

            return synthid_project_state_triton(
                slots,
                verify_tokens.contiguous(),
                self.context,
                self.context_history,
                self.history_cursor,
                self.num_calls,
                hash_iv=self.config.hash_iv,
                skip_first_ngram_calls=self.config.skip_first_ngram_calls,
            )

        context = self.context[slots]
        history = self.context_history[slots]
        history_cursor = self.history_cursor[slots]
        num_calls = self.num_calls[slots]
        hashes_per_step = []
        masks_per_step = []

        for step in range(verify_tokens.shape[1]):
            context_hashes, apply_mask = self._hashes_and_mask(
                context, history, num_calls
            )
            eligible = self._eligible(num_calls)
            hashes_per_step.append(context_hashes)
            masks_per_step.append(apply_mask)

            write_cols = history_cursor[:, None]
            previous_hashes = history.gather(1, write_cols).view(-1)
            history.scatter_(
                1,
                write_cols,
                torch.where(eligible, context_hashes, previous_hashes)[:, None],
            )
            history_cursor = torch.where(
                eligible,
                (history_cursor + 1) % self.config.context_history_size,
                history_cursor,
            )

            if step + 1 < verify_tokens.shape[1]:
                context = torch.cat(
                    (context[:, 1:], verify_tokens[:, step + 1 : step + 2]), dim=1
                )
                num_calls = num_calls + 1

        return (
            torch.stack(hashes_per_step, dim=1).reshape(-1),
            torch.stack(masks_per_step, dim=1).reshape(-1),
        )

    def commit(
        self,
        slots: torch.Tensor,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> None:
        """Append tokens to each slot's state.

        ``token_ids`` is ``[N]`` (one sampled token per row) or ``[N, W]``
        (accepted tokens of a verify step). Row ``i`` commits its first
        ``lengths[i]`` tokens, or all of them when ``lengths`` is None; a zero
        length leaves the slot unchanged.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if self.use_triton:
            from sglang.kernels.ops.sampling.synthid_triton import (
                synthid_commit_state_triton,
            )

            synthid_commit_state_triton(
                token_ids.contiguous(),
                lengths,
                slots,
                self.context,
                self.context_history,
                self.history_cursor,
                self.num_calls,
                hash_iv=self.config.hash_iv,
                skip_first_ngram_calls=self.config.skip_first_ngram_calls,
            )
            return

        token_ids = token_ids.to(dtype=torch.int64)
        if lengths is None:
            lengths = torch.full(
                (token_ids.shape[0],),
                token_ids.shape[1],
                dtype=torch.int64,
                device=token_ids.device,
            )
        context = self.context[slots]
        history_cursor = self.history_cursor[slots]
        num_calls = self.num_calls[slots]

        for step in range(token_ids.shape[1]):
            committed = step < lengths
            context_hashes = hash_contexts(context, self.config.hash_iv)
            record = committed & self._eligible(num_calls)

            write_slots = slots[record]
            write_cols = history_cursor[record]
            self.context_history[write_slots, write_cols] = context_hashes[record]
            history_cursor[record] = (write_cols + 1) % self.config.context_history_size

            shifted = torch.cat((context[:, 1:], token_ids[:, step : step + 1]), dim=1)
            context = torch.where(committed[:, None], shifted, context)
            num_calls = num_calls + committed.to(torch.int64)

        self.context[slots] = context
        self.history_cursor[slots] = history_cursor
        self.num_calls[slots] = num_calls

    def _eligible(self, num_calls: torch.Tensor) -> torch.Tensor:
        if self.config.skip_first_ngram_calls:
            return num_calls + 1 >= self.config.ngram_len
        return torch.ones_like(num_calls, dtype=torch.bool)

    def _hashes_and_mask(
        self, context: torch.Tensor, history: torch.Tensor, num_calls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_hashes = hash_contexts(context, self.config.hash_iv)
        repeated = torch.any(history == context_hashes[:, None], dim=1)
        return context_hashes, self._eligible(num_calls) & ~repeated
