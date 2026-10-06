import torch

from sglang.kernels.ops.diffusion import fuse_scale_shift_kernel
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp


class GatedResidual2Seg(CustomOp):
    """x + y * gate, with gate [b, 2, dim] switching rows at a token index: row 0 for
    tokens [0, seg_idx), row 1 after."""

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward_native(
        self, x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, seg_idx: int
    ) -> torch.Tensor:
        y = torch.cat(
            [
                y[:, :seg_idx].float() * gate[:, :1],
                y[:, seg_idx:].float() * gate[:, 1:],
            ],
            dim=1,
        )
        return x + y.to(dtype=x.dtype)

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_xpu(
        self, x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, seg_idx: int
    ) -> torch.Tensor:
        # bf16 addcmul into one preallocated output: no fp32 products and no cat copy.
        out = torch.empty_like(x)
        gate = gate.to(x.dtype)
        torch.addcmul(x[:, :seg_idx], y[:, :seg_idx], gate[:, :1], out=out[:, :seg_idx])
        torch.addcmul(x[:, seg_idx:], y[:, seg_idx:], gate[:, 1:], out=out[:, seg_idx:])
        return out


class MulAdd(CustomOp):
    """
    Fuse elementwise mul and add
    Input: a, b, c, OptionalInt[k]
    Output: a * (k + b) + c
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward_native(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ) -> torch.Tensor:
        # a.shape: [batch_size, seq_len, inner_dim]
        if b.dim() == 4:
            # b.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = b.shape[1]
            frame_seqlen = a.shape[1] // num_frames
            return c + (
                a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (k + b)
            ).flatten(1, 2)
        else:
            # b.shape: [batch_size, 1, inner_dim]
            return c + a * (k + b)

    def forward_cuda(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        return fuse_scale_shift_kernel(a, b, c, scale_constant=k)

    def forward_xpu(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        return self.forward_cuda(a, b, c, k=k)

    @torch.compile
    def forward_musa(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        return self.forward_native(a, b, c, k=k)

    def forward_npu(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        from sgl_kernel_npu.norm.scale_shift import fused_scale_shift

        return fused_scale_shift(a, b, c, scale_constant=k)
