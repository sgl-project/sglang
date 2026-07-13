"""Per-request speculative state store for PP + EAGLE/MTP.

This module provides CPU-owned, explicitly-cloned per-request state that
survives microbatch recomposition between PP rounds. The state is keyed by
RID (not batch position) because the microbatch composition can change
between rounds due to request finish, retract, merge, or filter.

Key invariants:
- All tensors are CPU-owned, contiguous, and cloned on store
- No alias to a larger GPU allocation
- No per-request CUDA allocation
- Explicit round/version tracking
- Cleanup when a request finishes, aborts, retracts, is evicted, or errors
- Bounded memory growth (one entry per live request)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class PPSpecRequestState:
    """CPU-owned speculative state for a single request in PP + spec decode.

    Attributes:
        rid: Request identifier
        round_id: Monotonically increasing round counter for this request
        bonus_token: The accepted bonus token (root of next verify chain)
        chain_tokens: CPU tensor of shape (num_draft_tokens,), contiguous, cloned.
                      chain_tokens[0] = bonus_token (root)
                      chain_tokens[1:] = draft proposals (zero-padded if no chain)
        committed_length: KV committed length at the time of storage
    """

    rid: str
    round_id: int
    bonus_token: int
    chain_tokens: torch.Tensor  # CPU, contiguous, cloned
    committed_length: int

    def __post_init__(self):
        assert self.chain_tokens.device.type == "cpu", (
            f"chain_tokens must be CPU-owned, got {self.chain_tokens.device}"
        )
        assert self.chain_tokens.is_contiguous(), (
            "chain_tokens must be contiguous"
        )


class PPSpecRequestStateStore:
    """CPU-owned per-request speculative state store.

    Thread safety: This store is accessed only from the scheduler thread
    (single-threaded event loop), so no locking is needed.
    """

    def __init__(self, num_draft_tokens: int):
        self._store: Dict[str, PPSpecRequestState] = {}
        self._num_draft_tokens = num_draft_tokens
        self._global_round_id: int = 0

        # Metrics / debug counters
        self.missing_chain_count: int = 0
        self.round_mismatch_count: int = 0
        self.max_live_entries: int = 0

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    @property
    def global_round_id(self) -> int:
        return self._global_round_id

    def __len__(self) -> int:
        return len(self._store)

    def store(
        self,
        rid: str,
        bonus_token: int,
        chain_tokens: Optional[torch.Tensor] = None,
        committed_length: int = 0,
    ) -> int:
        """Store or update state for a request.

        Args:
            rid: Request identifier
            bonus_token: The accepted bonus token (root of next chain)
            chain_tokens: Optional draft chain for next round. If None, a
                zero-padded chain with bonus as root is stored.
            committed_length: KV committed length at storage time

        Returns:
            The round_id assigned to this state
        """
        self._global_round_id += 1
        round_id = self._global_round_id

        if chain_tokens is not None:
            # Clone and move to CPU to ensure no GPU alias
            chain = chain_tokens.to("cpu", dtype=torch.int64).clone().contiguous()
            if chain.ndim == 1:
                # Reshape to (num_draft_tokens,) if flat
                if chain.numel() == self._num_draft_tokens:
                    pass
                elif chain.numel() > self._num_draft_tokens:
                    chain = chain[: self._num_draft_tokens]
                else:
                    padded = torch.zeros(
                        self._num_draft_tokens, dtype=torch.int64
                    )
                    padded[: chain.numel()] = chain
                    chain = padded
            elif chain.ndim == 2:
                # Shape (1, num_draft_tokens) — squeeze
                chain = chain.squeeze(0)[: self._num_draft_tokens]
                if chain.numel() < self._num_draft_tokens:
                    padded = torch.zeros(
                        self._num_draft_tokens, dtype=torch.int64
                    )
                    padded[: chain.numel()] = chain
                    chain = padded
        else:
            # Degenerate chain: bonus as root, zeros for drafts
            chain = torch.zeros(self._num_draft_tokens, dtype=torch.int64)
            chain[0] = bonus_token

        # Ensure chain[0] = bonus_token
        chain[0] = bonus_token

        self._store[rid] = PPSpecRequestState(
            rid=rid,
            round_id=round_id,
            bonus_token=bonus_token,
            chain_tokens=chain,
            committed_length=committed_length,
        )

        # Update max live entries
        if len(self._store) > self.max_live_entries:
            self.max_live_entries = len(self._store)

        return round_id

    def get(
        self,
        rid: str,
        expected_round: Optional[int] = None,
        strict: bool = True,
    ) -> PPSpecRequestState:
        """Get state for a request.

        Args:
            rid: Request identifier
            expected_round: If provided, check that the stored round matches
            strict: If True, raise on missing/mismatch. If False, log and
                return a zero-state fallback.

        Returns:
            The stored state for this request

        Raises:
            KeyError: If rid is not found and strict=True
            ValueError: If round mismatch and strict=True
        """
        if rid not in self._store:
            self.missing_chain_count += 1
            if strict:
                raise KeyError(
                    f"PP spec state missing for rid={rid}. "
                    f"This may indicate a lifecycle bug: request finished/"
                    f"retracted/evicted before result processing, or a stale "
                    f"result. missing_chain_count={self.missing_chain_count}"
                )
            logger.warning(
                f"PP spec state missing for rid={rid}, using zero fallback. "
                f"missing_chain_count={self.missing_chain_count}"
            )
            # Return a zero-state fallback
            chain = torch.zeros(self._num_draft_tokens, dtype=torch.int64)
            return PPSpecRequestState(
                rid=rid,
                round_id=-1,
                bonus_token=0,
                chain_tokens=chain,
                committed_length=0,
            )

        state = self._store[rid]

        if expected_round is not None and state.round_id != expected_round:
            self.round_mismatch_count += 1
            if strict:
                raise ValueError(
                    f"PP spec round mismatch for rid={rid}: "
                    f"expected round {expected_round}, got {state.round_id}. "
                    f"round_mismatch_count={self.round_mismatch_count}"
                )
            logger.warning(
                f"PP spec round mismatch for rid={rid}: "
                f"expected {expected_round}, got {state.round_id}. "
                f"Using latest state. round_mismatch_count={self.round_mismatch_count}"
            )

        return state

    def remove(self, rid: str) -> None:
        """Remove state for a request (finish/abort/retract/evict/error)."""
        self._store.pop(rid, None)

    def remove_many(self, rids: list) -> None:
        """Remove state for multiple requests."""
        for rid in rids:
            self._store.pop(rid, None)

    def cleanup_finished(self, batch_reqs: list) -> int:
        """Remove state for requests that have finished or been retracted.

        Args:
            batch_reqs: List of Req objects in the current batch

        Returns:
            Number of entries removed
        """
        # Collect all live RIDs from the batch
        live_rids = set()
        for req in batch_reqs:
            if not req.finished() and not req.is_retracted:
                live_rids.add(req.rid)

        # Remove entries not in live_rids
        to_remove = [rid for rid in self._store if rid not in live_rids]
        for rid in to_remove:
            del self._store[rid]

        return len(to_remove)

    def get_stats(self) -> dict:
        """Return current stats for monitoring."""
        return {
            "live_entries": len(self._store),
            "max_live_entries": self.max_live_entries,
            "missing_chain_count": self.missing_chain_count,
            "round_mismatch_count": self.round_mismatch_count,
            "global_round_id": self._global_round_id,
        }
