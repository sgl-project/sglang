"""What a speculative verify accepted."""

from __future__ import annotations

from typing import Optional

import msgspec
import torch


class VerifyAccept(msgspec.Struct, frozen=True):
    """A tree verify (EAGLE, NGRAM) gives ``tokens`` as its flat per-position
    predictions and ``accept_index`` [bs, width] into them, -1 past the
    accepted path. A chain verify (DFlash, DSpark) gives ``tokens`` as its
    [bs, width] blocks and no ``accept_index``: each row accepts a prefix."""

    tokens: torch.Tensor
    # [bs], including the bonus token.
    accept_lens: torch.Tensor
    accept_index: Optional[torch.Tensor] = None

    def accept_tokens(self) -> torch.Tensor:
        """[bs, width]; row i's first ``accept_lens[i]`` are its accepted tokens."""
        if self.accept_index is None:
            return self.tokens
        return self.tokens[self.accept_index]

    def accept_mask(self) -> torch.Tensor:
        """[bs, width] bool, set at the accepted positions of ``accept_tokens``."""
        if self.accept_index is not None:
            return self.accept_index >= 0
        positions = torch.arange(self.tokens.shape[1], device=self.tokens.device)
        return positions < self.accept_lens[:, None]


class ChainAccept(msgspec.Struct):
    """A chain verify's (DFlash, DSpark) accept, per request: the first
    ``correct_len`` drafts match the target, ``bonus`` follows them, and the
    first ``commit_lens`` tokens of ``out_tokens`` [bs, width] commit. A
    request capped by ``VerifyStep.run`` commits one token, not
    ``correct_len + 1``; ``correct_len`` stays the accept kernel's, for the
    acceptance estimators."""

    correct_len: torch.Tensor
    bonus: torch.Tensor
    commit_lens: torch.Tensor
    out_tokens: torch.Tensor
    # The accepted drafts a planned verify length cut; None without a cutoff.
    cap_trim_lens: Optional[torch.Tensor] = None
    # prefix_lens + commit_lens; None when the accept did not compute it.
    new_seq_lens: Optional[torch.Tensor] = None
    # The accept ran in a CUDA graph (DSpark's verify-graph fold or its
    # sampling-accept region): its tensors are views of the graph's buffers.
    in_graph: bool = False
