"""What a speculative verify accepted, and its commit to per-request sampling state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.watermarking.interface import VerifyWatermark


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


def commit_verify_accept(
    sampling_info: SamplingBatchInfo,
    accept: VerifyAccept,
    *,
    watermark: Optional[VerifyWatermark],
) -> None:
    """Commit the accepted tokens to the penalty history and the watermark context."""
    if sampling_info.step_penalties is not None:
        # Guarded here: building the arguments launches device work.
        sampling_info.step_penalties.commit(
            accept.accept_tokens(), accept_mask=accept.accept_mask()
        )
    if watermark is not None:
        watermark.commit(accept)
