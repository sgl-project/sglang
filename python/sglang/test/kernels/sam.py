"""PyTorch references for the SAM relative-position bias and window kernels."""

import torch
import torch.nn.functional as F


def sam_rel_pos_bias_reference(
    rel_h: torch.Tensor, rel_w: torch.Tensor
) -> torch.Tensor:
    batch, heads, queries, key_height, _ = rel_h.shape
    key_width = rel_w.shape[-1]
    return (rel_h + rel_w).view(batch, heads, queries, key_height * key_width)


def make_windows(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, H, W, C] -> zero-padded [B * num_windows, ws, ws, C] windows."""
    batch, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    padded = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return (
        padded.view(
            batch,
            (height + pad_h) // window_size,
            window_size,
            (width + pad_w) // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, channels)
    )


def window_unpartition_add_reference(
    windows: torch.Tensor,
    shortcut: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    batch, height, width, _ = shortcut.shape
    padded_height = (height + window_size - 1) // window_size * window_size
    padded_width = (width + window_size - 1) // window_size * window_size
    restored = windows.view(
        batch,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        -1,
    )
    restored = restored.permute(0, 1, 3, 2, 4, 5).contiguous()
    restored = restored.view(batch, padded_height, padded_width, -1)
    return shortcut + restored[:, :height, :width, :].contiguous()
