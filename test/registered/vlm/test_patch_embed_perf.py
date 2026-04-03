import os
import statistics

import pytest
import torch
import torch.nn as nn

from sglang.srt.models.glm4v import Glm4vVisionPatchEmbed
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-test-1-gpu-large")

PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
IN_CHANNELS = 3
HIDDEN_SIZE = 1536
FLAT_DIM = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE


class ReferenceConv3dPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.proj(x).view(-1, self.hidden_size)
        return x


def _build_modules(device: str, dtype: torch.dtype):
    conv_mod = ReferenceConv3dPatchEmbed().to(device=device, dtype=dtype).eval()
    linear_mod = (
        Glm4vVisionPatchEmbed(
            patch_size=PATCH_SIZE,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            in_channels=IN_CHANNELS,
            hidden_size=HIDDEN_SIZE,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    with torch.no_grad():
        linear_mod.proj.weight.copy_(conv_mod.proj.weight)
        linear_mod.proj.bias.copy_(conv_mod.proj.bias)

        linear_mod.copy_conv3d_weight_to_linear()

    return conv_mod, linear_mod


def _benchmark_cuda_module(
    module: nn.Module,
    x: torch.Tensor,
    warmup: int = 50,
    inner_iters: int = 200,
    repeats: int = 10,
) -> float:
    assert x.is_cuda
    module.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize()

        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(inner_iters):
                module(x)
            end.record()

            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / inner_iters)

    return statistics.median(samples)


def test_patch_embed_linear_matches_conv3d():
    torch.manual_seed(0)

    device = "cpu"
    dtype = torch.float32

    conv_mod, linear_mod = _build_modules(device=device, dtype=dtype)

    x = torch.randn(512, FLAT_DIM, device=device, dtype=dtype)

    with torch.inference_mode():
        y_conv = conv_mod(x)
        y_linear = linear_mod(x)

    torch.testing.assert_close(
        y_conv,
        y_linear,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for perf benchmark"
)
def test_patch_embed_linear_conv3d():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    conv_mod, linear_mod = _build_modules(device=device, dtype=dtype)

    num_patches = int(os.getenv("GLM4V_NUM_PATCHES", "4096"))
    warmup = int(os.getenv("GLM4V_WARMUP", "50"))
    inner_iters = int(os.getenv("GLM4V_INNER_ITERS", "200"))
    repeats = int(os.getenv("GLM4V_REPEATS", "10"))

    x = torch.randn(num_patches, FLAT_DIM, device=device, dtype=dtype).contiguous()

    conv_ms = _benchmark_cuda_module(
        conv_mod, x, warmup=warmup, inner_iters=inner_iters, repeats=repeats
    )
    linear_ms = _benchmark_cuda_module(
        linear_mod, x, warmup=warmup, inner_iters=inner_iters, repeats=repeats
    )

    speedup = conv_ms / linear_ms
    print(
        f"\n[patch_embed perf] conv3d={conv_ms:.4f} ms | "
        f"linear={linear_ms:.4f} ms | speedup={speedup:.3f}x"
    )

    min_speedup = float(os.getenv("GLM4V_MIN_SPEEDUP", "1.00"))
    assert speedup >= min_speedup, (
        f"Expected speedup >= {min_speedup:.3f}x, but got {speedup:.3f}x "
        f"(conv3d={conv_ms:.4f} ms, linear={linear_ms:.4f} ms)"
    )
