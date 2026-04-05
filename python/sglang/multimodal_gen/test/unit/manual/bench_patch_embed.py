"""
Benchmark: Conv3d vs reshape + F.linear PatchEmbed.

Matches the real e2e pipeline conditions:
  - Conv3d weights are FP32 (no dtype passed to PatchEmbed.__init__)
  - Input latents are BF16 (cast by denoising loop)
  - torch.autocast(dtype=bf16) wraps the forward pass
  - .flatten(2).transpose(1, 2) follows PatchEmbed (wanvideo.py:1008)

Uses CUDA events for accurate GPU timing. Each case runs warmup iterations
followed by timed iterations, reports median latency and speedup.

Usage:
    python bench_patch_embed.py
    python bench_patch_embed.py --warmup 20 --iters 100
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """Conv3d-based PatchEmbed (upstream/main)."""

    def __init__(self, patch_size, in_chans, embed_dim, flatten=True, bias=True):
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
        )

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """Reshape + F.linear PatchEmbed (opt_krea)."""

    def __init__(self, patch_size, in_chans, embed_dim, flatten=True, bias=True):
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
        )

    def forward(self, x):
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
        return x


def _copy_weights(src, dst):
    dst.proj.weight.data.copy_(src.proj.weight.data)
    if src.proj.bias is not None:
        dst.proj.bias.data.copy_(src.proj.bias.data)


def bench_one(fn, warmup, iters):
    """Returns list of per-iteration latencies in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


# Real latent shapes: T = (num_frames-1)//4+1, H = height//8, W = width//8
# (name, patch_size, in_chans, embed_dim, flatten, B, T, H, W)
BENCH_CASES = [
    # Wan2.1-I2V-14B: 480x832
    ("Wan-21f-480x832", (1, 2, 2), 16, 5120, False, 1, 6, 60, 104),  # 21 frames
    ("Wan-41f-480x832", (1, 2, 2), 16, 5120, False, 1, 11, 60, 104),  # 41 frames
    ("Wan-81f-480x832", (1, 2, 2), 16, 5120, False, 1, 21, 60, 104),  # 81 frames
    ("Wan-101f-480x832", (1, 2, 2), 16, 5120, False, 1, 26, 60, 104),  # 101 frames
    # Wan2.1-I2V-14B: 720x1280
    ("Wan-21f-720x1280", (1, 2, 2), 16, 5120, False, 1, 6, 90, 160),  # 21 frames 720p
    ("Wan-41f-720x1280", (1, 2, 2), 16, 5120, False, 1, 11, 90, 160),  # 41 frames 720p
    # HunyuanVideo
    ("HunYuan-21f-480x832", (1, 2, 2), 16, 3072, True, 1, 6, 60, 104),
    ("HunYuan-41f-480x832", (1, 2, 2), 16, 3072, True, 1, 11, 60, 104),
]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PatchEmbed: Conv3d vs F.linear"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"

    # ── Real pipeline conditions ──────────────────────────────────────────
    # 1. Weights are FP32 (PatchEmbed.__init__ has no dtype arg in real code)
    # 2. Input is BF16 (latents.to(target_dtype) in denoising loop)
    # 3. torch.autocast(dtype=bf16) wraps the denoising loop
    # 4. .flatten(2).transpose(1, 2) follows PatchEmbed (wanvideo.py:1008)
    # ──────────────────────────────────────────────────────────────────────

    header = f"{'Case':<25} {'Conv3d(ms)':>10} {'F.linear(ms)':>12} {'Speedup':>8}"
    print("Real pipeline conditions: FP32 weights, BF16 input, autocast(bf16)")
    print(header)
    print("-" * len(header))

    for name, patch_size, in_chans, embed_dim, flatten, B, T, H, W in BENCH_CASES:
        torch.manual_seed(42)

        # FP32 weights – matches real model init (no dtype passed)
        conv_model = (
            PatchEmbed3D(
                patch_size,
                in_chans,
                embed_dim,
                flatten,
            )
            .to(device)
            .eval()
        )
        lin_model = (
            PatchEmbed(
                patch_size,
                in_chans,
                embed_dim,
                flatten,
            )
            .to(device)
            .eval()
        )
        _copy_weights(conv_model, lin_model)

        # BF16 input – matches real latent dtype
        x = torch.randn(B, in_chans, T, H, W, device=device, dtype=torch.bfloat16)

        # Include the .flatten(2).transpose(1, 2) that follows PatchEmbed
        # in WanTransformer3DModel.forward (wanvideo.py:1008)
        def conv_fn():
            out = conv_model(x)
            return out.flatten(2).transpose(1, 2)

        def lin_fn():
            out = lin_model(x)
            return out.flatten(2).transpose(1, 2)

        # autocast(bf16) – matches real denoising loop (denoising.py:1016)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t_conv = bench_one(conv_fn, args.warmup, args.iters)
            t_lin = bench_one(lin_fn, args.warmup, args.iters)

        med_conv = sorted(t_conv)[len(t_conv) // 2]
        med_lin = sorted(t_lin)[len(t_lin) // 2]
        speedup = med_conv / med_lin if med_lin > 0 else float("inf")

        print(f"{name:<25} {med_conv:>10.3f} {med_lin:>12.3f} {speedup:>7.2f}x")

    print()


if __name__ == "__main__":
    main()
