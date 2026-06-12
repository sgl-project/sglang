# SPDX-License-Identifier: Apache-2.0
"""Wan VAE decode microbenchmark (eager vs RMSNorm+SiLU fusion).

    python wan_vae_decode_bench.py [off|on|correctness]

`off`/`on` time the first 3 requests + steady-state per shape; `correctness`
reports eager-vs-fused max abs diff. Run each arm in its own process.
"""

import glob
import os
import sys
import time

import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    AutoencoderKLWan,
    _fuse_wan_vae_rmsnorm_silu,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.test.server.accuracy_utils import load_checkpoint_weights

DEV, DT = "cuda", torch.bfloat16
COMP = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/*/vae"
    )
)[0]

# z_dim=16, spatial /8, temporal (T-1)/4+1; 480p=832x480, 720p=1280x720, 16fps.
SHAPES = {
    "480p_6s": (1, 16, 25, 60, 104),
    "480p_10s": (1, 16, 41, 60, 104),
    "720p_6s": (1, 16, 25, 90, 160),
    "720p_10s": (1, 16, 41, 90, 160),
}


def build_vae(fuse):
    vc = WanT2V480PConfig().vae_config
    vc.update_model_arch(get_diffusers_component_config(component_path=COMP))
    vae = AutoencoderKLWan(vc)
    load_checkpoint_weights(vae, COMP)
    vae = vae.to(DEV, DT).eval()
    if fuse:
        _fuse_wan_vae_rmsnorm_silu(vae)
    return vae


@torch.no_grad()
def decode_ms(vae, z):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vae.decode(z)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def bench(fuse):
    vae = build_vae(fuse)
    print(f"{'shape':<10} {'req1':>8} {'req2':>8} {'req3':>8} {'warm':>8}  (ms)")
    for name, shp in SHAPES.items():
        torch.manual_seed(0)
        z = torch.randn(shp, device=DEV, dtype=DT)
        r = [decode_ms(vae, z) for _ in range(3)]
        warm = min(decode_ms(vae, z) for _ in range(4))
        print(f"{name:<10} {r[0]:>8.0f} {r[1]:>8.0f} {r[2]:>8.0f} {warm:>8.0f}")
        torch.cuda.empty_cache()


def correctness():
    off, on = build_vae(False), build_vae(True)
    for name, shp in SHAPES.items():
        torch.manual_seed(0)
        z = torch.randn(shp, device=DEV, dtype=DT)
        with torch.no_grad():
            d = (off.decode(z).float() - on.decode(z).float()).abs().max().item()
        print(f"{name}: max_abs_diff={d:.5f}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    arm = sys.argv[1] if len(sys.argv) > 1 else "off"
    correctness() if arm == "correctness" else bench(fuse=arm == "on")
