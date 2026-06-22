from typing import Optional

import torch


class PPProxyTopKBuffer:
    __slots__ = ("buffer",)

    def __init__(self):
        self.buffer: Optional[torch.Tensor] = None

    def copy(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if topk_indices.numel() == 0:
            return topk_indices

        buffer = self.buffer
        if (
            buffer is None
            or buffer.numel() < topk_indices.numel()
            or buffer.dtype != topk_indices.dtype
            or buffer.device != topk_indices.device
        ):
            buffer = torch.empty(
                topk_indices.numel(),
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            )
            self.buffer = buffer

        buffer_view = buffer[: topk_indices.numel()].view_as(topk_indices)
        buffer_view.copy_(topk_indices)
        return buffer_view
