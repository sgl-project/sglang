"""
Test that the optimized PatchEmbed (reshape + F.linear) is equivalent
to the original Conv3d-based PatchEmbed from upstream/main.

The opt_krea branch replaces Conv3d forward with manual
reshape + permute + F.linear for 5D input. This is valid because
Conv3d with stride==kernel_size is a non-overlapping patch extraction
followed by linear projection, which is exactly what the manual path does.

We disable TF32 so cuDNN (Conv3d) and cuBLAS (F.linear) both use
full FP32 precision, enabling strict numerical comparison.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """PatchEmbed from upstream/main: uses Conv3d directly."""

    def __init__(
        self, patch_size, in_chans, embed_dim, flatten=True, bias=True, dtype=None
    ):
        super().__init__()
        if isinstance(patch_size, list | tuple):
            if len(patch_size) == 1:
                patch_size = (patch_size[0], patch_size[0])
        else:
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=dtype,
        )
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """PatchEmbed from opt_krea: replaces Conv3d with reshape + F.linear for 5D input."""

    def __init__(
        self, patch_size, in_chans, embed_dim, flatten=True, bias=True, dtype=None
    ):
        super().__init__()
        if isinstance(patch_size, list | tuple):
            if len(patch_size) == 1:
                patch_size = (1, patch_size[0], patch_size[0])
            elif len(patch_size) == 2:
                patch_size = (1, patch_size[0], patch_size[1])
        else:
            patch_size = (1, patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=dtype,
        )
        self.norm = nn.Identity()

    def forward(self, x):
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            pt, ph, pw = self.patch_size
            T_ = T // pt
            H_ = H // ph
            W_ = W // pw
            x = x.reshape(B, C, T_, pt, H_, ph, W_, pw)
            x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.reshape(B, T_ * H_ * W_, C * pt * ph * pw)
            w = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
            x = F.linear(x, w, self.proj.bias)
            if not self.flatten:
                x = x.reshape(B, T_, H_, W_, -1).permute(0, 4, 1, 2, 3).contiguous()
            x = self.norm(x)
            return x
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _copy_weights(src, dst):
    dst.proj.weight.data.copy_(src.proj.weight.data)
    if src.proj.bias is not None:
        dst.proj.bias.data.copy_(src.proj.bias.data)


def _run_equivalence(
    patch_size,
    in_chans,
    embed_dim,
    flatten,
    bias,
    weight_dtype,
    input_dtype,
    B,
    T,
    H,
    W,
    atol,
    rtol,
):
    """Helper: build both models with shared weights, run forward, compare.

    Args:
        weight_dtype: dtype for Conv3d weights (None = FP32).
        input_dtype:  dtype for the input tensor (None = FP32).
    """
    torch.manual_seed(42)
    main = (
        PatchEmbed3D(patch_size, in_chans, embed_dim, flatten, bias, dtype=weight_dtype)
        .to(DEVICE)
        .eval()
    )
    opt = (
        PatchEmbed(patch_size, in_chans, embed_dim, flatten, bias, dtype=weight_dtype)
        .to(DEVICE)
        .eval()
    )
    _copy_weights(main, opt)

    x = torch.randn(
        B, in_chans, T, H, W, device=DEVICE, dtype=input_dtype or torch.float32
    )
    with torch.no_grad():
        out_main = main(x)
        out_opt = opt(x)

    assert (
        out_main.shape == out_opt.shape
    ), f"Shape mismatch: {out_main.shape} vs {out_opt.shape}"
    assert (
        out_main.dtype == out_opt.dtype
    ), f"Dtype mismatch: {out_main.dtype} vs {out_opt.dtype}"
    torch.testing.assert_close(out_main, out_opt, atol=atol, rtol=rtol)


@pytest.fixture(autouse=True)
def _disable_tf32():
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = prev_cudnn
    torch.backends.cuda.matmul.allow_tf32 = prev_matmul


# ── Wan2.1 / Wan2.2 / CausalWan / Helios ────────────────────────────────────
# patch_size=(1,2,2), in_channels=16, embed_dim=5120, flatten=False
# Real usage: weight=FP32 (no dtype passed), input=BF16 from VAE latent


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (None, 1e-4, 1e-4),  # weight=FP32, input=FP32
        (torch.bfloat16, 1e-2, 1e-2),  # weight=BF16, input=BF16
        (torch.float16, 1e-2, 1e-2),  # weight=FP16, input=FP16
    ],
    ids=["fp32", "bf16", "fp16"],
)
@pytest.mark.parametrize(
    "B,T,H,W",
    [
        (1, 21, 60, 104),  # 480p typical
        (2, 9, 40, 64),  # smaller resolution, batch=2
        (1, 33, 90, 160),  # 720p longer video
    ],
    ids=["480p-B1", "small-B2", "720p-B1"],
)
def test_wan_helios(dtype, atol, rtol, B, T, H, W):
    _run_equivalence(
        patch_size=(1, 2, 2),
        in_chans=16,
        embed_dim=5120,
        flatten=False,
        bias=True,
        weight_dtype=dtype,
        input_dtype=dtype,
        B=B,
        T=T,
        H=H,
        W=W,
        atol=atol,
        rtol=rtol,
    )


# ── HunyuanVideo ─────────────────────────────────────────────────────────────
# patch_size=[1,2,2] (list!), in_channels=16, embed_dim=3072, flatten=True
# Real usage: dtype passed to PatchEmbed, so weight & input share same dtype


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (None, 1e-4, 1e-4),  # weight=FP32, input=FP32
        (torch.bfloat16, 1e-2, 1e-2),  # weight=BF16, input=BF16
        (torch.float16, 1e-2, 1e-2),  # weight=FP16, input=FP16
    ],
    ids=["fp32", "bf16", "fp16"],
)
@pytest.mark.parametrize(
    "B,T,H,W",
    [
        (1, 21, 60, 104),
        (2, 9, 40, 64),
    ],
    ids=["480p-B1", "small-B2"],
)
def test_hunyuanvideo(dtype, atol, rtol, B, T, H, W):
    _run_equivalence(
        patch_size=[1, 2, 2],
        in_chans=16,
        embed_dim=3072,
        flatten=True,
        bias=True,
        weight_dtype=dtype,
        input_dtype=dtype,
        B=B,
        T=T,
        H=H,
        W=W,
        atol=atol,
        rtol=rtol,
    )


# ── No-bias variants ─────────────────────────────────────────────────────────


def test_wan_no_bias():
    _run_equivalence(
        patch_size=(1, 2, 2),
        in_chans=16,
        embed_dim=5120,
        flatten=False,
        bias=False,
        weight_dtype=None,
        input_dtype=None,
        B=1,
        T=21,
        H=60,
        W=104,
        atol=1e-4,
        rtol=1e-4,
    )


def test_hunyuanvideo_no_bias():
    _run_equivalence(
        patch_size=[1, 2, 2],
        in_chans=16,
        embed_dim=3072,
        flatten=True,
        bias=False,
        weight_dtype=None,
        input_dtype=None,
        B=1,
        T=21,
        H=60,
        W=104,
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
