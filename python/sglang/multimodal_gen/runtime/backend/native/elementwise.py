import torch


def mul_add(
    self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
) -> torch.Tensor:
    # a.shape: [batch_size, seq_len, inner_dim]
    if b.dim() == 4:
        # b.shape: [batch_size, num_frames, 1, inner_dim]
        num_frames = b.shape[1]
        frame_seqlen = a.shape[1] // num_frames
        return c + (a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * b).flatten(
            1, 2
        )
    else:
        # b.shape: [batch_size, 1, inner_dim]
        return c + a * (k + b)
