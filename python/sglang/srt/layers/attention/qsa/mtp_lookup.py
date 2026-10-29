from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeferredMtpSparseIndices:
    """Borrowed selection and dynamic tail, resolved by the native KV packer."""

    indices: torch.Tensor
    captured_len: torch.Tensor
    requests: torch.Tensor
    positions: torch.Tensor
    tail_offsets: torch.Tensor

    @property
    def shape(self):
        return (self.requests.numel(), self.indices.shape[1])

    @property
    def tail_width(self):
        return self.tail_offsets.numel()

    def record_stream(self, stream):
        for tensor in (
            self.indices,
            self.captured_len,
            self.requests,
            self.positions,
            self.tail_offsets,
        ):
            tensor.record_stream(stream)

    def materialize(self):
        rows = self.requests.to(torch.long)
        out = self.indices[rows]
        base = self.captured_len[rows].to(torch.int64)
        tail = base.unsqueeze(1) + self.tail_offsets.unsqueeze(0)
        valid = tail <= self.positions.to(torch.int64).unsqueeze(1)
        out[:, -self.tail_width :] = torch.where(valid, tail, -1).to(out.dtype)
        return out
