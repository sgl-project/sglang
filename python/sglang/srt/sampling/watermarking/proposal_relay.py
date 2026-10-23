"""Relay sparse ON and full-vocabulary OFF proposals without truncating OFF rows."""

from __future__ import annotations

import torch


def merge_proposals(left, left_ids, right, right_ids):
    """A non-overlap merge may combine sparse ON and dense OFF input batches."""
    if (left_ids is None) == (right_ids is None):
        return torch.cat((left, right)), (
            torch.cat((left_ids, right_ids)) if left_ids is not None else None
        )
    vocab_size = left.shape[-1] if left_ids is None else right.shape[-1]

    def dense(probs, token_ids):
        if token_ids is None:
            return probs
        return torch.zeros(
            (*probs.shape[:-1], vocab_size), dtype=probs.dtype, device=probs.device
        ).scatter_(-1, token_ids, probs)

    return torch.cat((dense(left, left_ids), dense(right, right_ids))), None


class RequestProposalRelay:
    def __init__(self, *, slots: int, candidate_count: int, device):
        self.slots = slots
        self.candidate_count = candidate_count
        self.device = device
        self.sparse_probs = None
        self.sparse_tokens = None
        self.dense_probs = None
        self._sparse_slots: list[bool | None] = [None] * slots

    @staticmethod
    def _host_indices(indices, indices_cpu):
        if indices_cpu is None:
            if indices.device.type != "cpu":
                raise RuntimeError("Proposal relay requires CPU request-slot metadata")
            indices_cpu = indices
        if indices_cpu.device.type != "cpu" or len(indices_cpu) != len(indices):
            raise ValueError("Proposal relay CPU request-slot metadata mismatch")
        return indices_cpu.tolist()

    def store(self, indices, probs, token_ids, enabled_host, *, indices_cpu=None):
        if probs is None:
            raise RuntimeError("Per-request proposal relay has no probabilities")
        if len(enabled_host) != len(indices):
            raise ValueError("Per-request proposal policy cardinality mismatch")
        slots = self._host_indices(indices, indices_cpu)
        for slot in slots:
            self._sparse_slots[slot] = token_ids is not None
        if token_ids is not None:
            if self.sparse_probs is None:
                self.sparse_probs = torch.empty(
                    (self.slots, probs.shape[1]), dtype=probs.dtype, device=self.device
                )
                self.sparse_tokens = torch.empty(
                    (self.slots, probs.shape[1]),
                    dtype=token_ids.dtype,
                    device=self.device,
                )
            self.sparse_probs[indices] = probs
            self.sparse_tokens[indices] = token_ids
            return
        if self.dense_probs is None:
            self.dense_probs = torch.empty(
                (self.slots, probs.shape[1]), dtype=probs.dtype, device=self.device
            )
        self.dense_probs[indices] = probs

    def resolve(self, indices, state, *, indices_cpu=None):
        enabled_host = state.request_enabled_host or [True] * len(indices)
        slots = self._host_indices(indices, indices_cpu)
        layouts = [self._sparse_slots[slot] for slot in slots]
        if any(layout is None for layout in layouts):
            raise RuntimeError("Proposal relay resolved an unstashed request slot")
        if any(
            layout and not enabled for layout, enabled in zip(layouts, enabled_host)
        ):
            raise RuntimeError("OFF proposal relay cannot consume a sparse ON payload")
        if all(layouts):
            if self.sparse_probs is None:
                raise RuntimeError("ON proposal relay has no sparse payload")
            return self.sparse_probs[indices], self.sparse_tokens[indices]
        if self.dense_probs is None:
            raise RuntimeError("OFF proposal relay has no full-vocabulary payload")
        dense = self.dense_probs[indices]
        sparse_rows = [i for i, layout in enumerate(layouts) if layout]
        if sparse_rows:
            if self.sparse_probs is None:
                raise RuntimeError("Mixed proposal relay has no ON payload")
            rows = torch.tensor(sparse_rows, dtype=torch.int64, device=self.device)
            slots = indices[rows]
            on_probs = torch.zeros_like(dense[rows]).scatter_(
                1, self.sparse_tokens[slots], self.sparse_probs[slots]
            )
            dense[rows] = on_probs
        if all(enabled_host):
            # Compress only when a formerly mixed batch becomes homogeneous ON.
            probs, tokens = torch.topk(
                dense, min(self.candidate_count, dense.shape[-1]), dim=-1
            )
            tokens, order = tokens.sort(dim=-1)
            return probs.gather(1, order), tokens
        return dense, None
