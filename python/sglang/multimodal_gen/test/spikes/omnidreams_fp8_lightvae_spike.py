#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Feasibility spike: FP8 vs bf16 for the OmniDreams LightVAE encoder.

GPU-ONLY. NOT imported by any inference path. Run manually on the target GPU
(ideally consumer Blackwell ``sm_120``: RTX 5090 / RTX PRO 6000):

    python python/sglang/multimodal_gen/test/spikes/omnidreams_fp8_lightvae_spike.py \
        --ckpt /path/to/lightvaew2_1.pth

Proves (a) FP8 builds/runs on this GPU, (b) FP8 GEMM is faster than bf16, and
(c) the per-channel-FP8 (``float8_e4m3fn``, scale_max=24 -- the
export_lightvae_fp8_state.py recipe) numerical error vs bf16 is within budget.

It uses PyTorch-native FP8 (``torch.float8_e4m3fn`` + ``torch._scaled_mm``) as the
buildable proxy for the full FlashDreams ``sm_120`` CUTLASS/TIN16 kernels, so it
de-risks the numerics + speedup direction WITHOUT the ~5900-LOC kernel port. See
``docs/superpowers/omnidreams_p4_fp8_design.md`` for the design + go/no-go.
"""

from __future__ import annotations

import argparse
import time


def _cap() -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        return False, "CUDA not available (spike is GPU-only)."
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    sm = f"sm_{major}{minor}"
    note = "" if sm == "sm_120" else " (target is sm_120; the CUTLASS port is sm_120-only)"
    if getattr(torch, "float8_e4m3fn", None) is None:
        return False, f"{name} {sm}: torch.float8_e4m3fn unavailable."
    return True, f"{name} {sm}{note}"


def _quantize_fp8_per_channel(w, scale_max: float):
    """Per-output-channel symmetric FP8 (e4m3fn) quant of a 2D weight [O, I].

    Mirrors export_lightvae_fp8_state.py: per-channel amax -> scale (clamped to
    scale_max), cast to float8_e4m3fn. Returns (w_fp8, inv_scale[O,1]).
    """
    import torch

    amax = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    scale = (amax / scale_max).clamp_(min=1e-8)
    w_fp8 = (w / scale).to(torch.float8_e4m3fn)
    return w_fp8, scale


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/root/blockdata/lightvaew2_1.pth")
    ap.add_argument("--scale-max", type=float, default=24.0)
    ap.add_argument("--m", type=int, default=8192, help="GEMM rows (tokens).")
    args = ap.parse_args()

    ok, msg = _cap()
    print(f"[spike] device: {msg}")
    if not ok:
        print("[spike] SKIP:", msg)
        return

    import torch

    dev = torch.device("cuda")

    # (b)+(c): representative GEMM (a LightVAE conv lowered to matmul). Use the
    # encoder's largest conv weight if the ckpt is present; else a synthetic one.
    try:
        from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
            LightVAEEncoder,
        )

        enc = LightVAEEncoder(
            args.ckpt, latents_mean=[0.0] * 16, latents_std=[1.0] * 16,
            dtype=torch.bfloat16,
        ).to(dev)
        # Flatten the head conv weight [O, I, kt, kh, kw] -> [O, I*kt*kh*kw].
        w = enc.encoder.head[2].weight.detach().reshape(
            enc.encoder.head[2].weight.shape[0], -1
        ).to(dev)
        print(f"[spike] using head conv weight {tuple(w.shape)} from {args.ckpt}")
    except Exception as e:  # noqa: BLE001 - spike: any failure -> synthetic fallback
        print(f"[spike] ckpt load failed ({e}); using synthetic [512, 2304] weight")
        w = torch.randn(512, 2304, device=dev)

    o, i = w.shape
    x = torch.randn(args.m, i, device=dev)

    # bf16 reference.
    w_bf16 = w.to(torch.bfloat16)
    x_bf16 = x.to(torch.bfloat16)
    ref = (x_bf16 @ w_bf16.t()).float()
    bf16_ms = _bench(lambda: x_bf16 @ w_bf16.t())

    # FP8 per-channel (RowWise format for sm_120 _scaled_mm contract).
    w_fp8, w_scale = _quantize_fp8_per_channel(w.float(), args.scale_max)
    x_amax = x.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    x_scale_row = (x_amax / args.scale_max).clamp_(min=1e-8)
    x_fp8 = (x / x_scale_row).to(torch.float8_e4m3fn)

    def fp8_mm():
        # PyTorch 2.11 sm_120: RowWise format — scale_a (M,1), scale_b (1,N).
        out = torch._scaled_mm(
            x_fp8, w_fp8.t(),
            scale_a=x_scale_row.float().contiguous(),
            scale_b=w_scale.reshape(1, o).float().contiguous(),
            out_dtype=torch.bfloat16,
        )
        return out

    try:
        fp8_out = fp8_mm().float()
        fp8_ms = _bench(fp8_mm)
    except Exception as e:  # noqa: BLE001
        print(f"[spike] FP8 _scaled_mm failed on this GPU: {e}")
        return

    abs_err = (fp8_out - ref).abs().max().item()
    rel_err = ((fp8_out - ref).abs() / ref.abs().clamp_min(1e-3)).mean().item()
    print(f"[spike] bf16 GEMM: {bf16_ms:.4f} ms | FP8 GEMM: {fp8_ms:.4f} ms "
          f"| speedup {bf16_ms / fp8_ms:.2f}x")
    print(f"[spike] FP8 vs bf16: max_abs_err={abs_err:.4f} mean_rel_err={rel_err:.4%}")
    print("[spike] go/no-go (layer-level lower bound): "
          f"speedup>=1.5x -> {bf16_ms / fp8_ms >= 1.5} | "
          f"rel_err<=5% -> {rel_err <= 0.05}")


if __name__ == "__main__":
    main()
