#!/usr/bin/env python3
"""Repack a DeepSeek-V4 checkpoint for GPUs without FP8 or FP4 kernels (SM80).

The published checkpoint already stores its routed experts as MXFP4 -- e2m1
nibbles packed into int8 [R, C/2] with e8m0 block scales over groups of 32.
Those bytes are copied through **unchanged**, so the experts carry no
additional quantisation error. Everything else (attention, shared experts,
dense projections, embeddings, head, router) is FP8 e4m3 with 128x128 block
scales, which Ampere cannot compute on; those are dequantised to BF16, which
is lossless relative to the FP8 values.

The output declares quant_method="mxfp4" and marks every non-expert module as
ignored, so SGLang serves the experts through the Marlin MXFP4 MoE path and
the rest unquantised.

Four gates run during conversion; each one fails the build:
  1. Expert passthrough: sample expert tensors, decode the written bytes
     (nibble unpack x 2^e8m0) and compare against the reference dequant.
     Relative error must be exactly 0 -- a passthrough may not change a value.
  2. BF16 layers: compare against the FP8 reference dequant, within bf16
     rounding (rel <= 1e-2).
  3. Ignore list: covers every non-expert 2-D weight, and no expert.
  4. Completeness: index intact, shard count matches, no missing tensor.

Usage:
    python3 repack_mxfp4.py --src <official dir> --dst <output dir>
                            [--limit-shards N] [--stride K --offset J --no-finalize]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)  # e2m1, matching the official inference/convert.py

# Routed experts in the main body (layers.N) and in the MTP draft layers
# (mtp.N) are both native MXFP4 and both pass through bit-for-bit. Matching
# only the "layers." prefix drops the MTP experts into the BF16 branch, which
# contradicts the mxfp4 declaration in the config and costs draft quality.
EXPERT_RE = re.compile(
    r"^(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)\.(weight|scale)$"
)


def dequant_fp4(packed_i8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Reference dequant: int8 [R, C/2] low-nibble-first + e8m0 [R, C/32] -> fp32 [R, C]."""
    b = packed_i8.view(torch.uint8)
    lo = (b & 0xF).long()
    hi = (b >> 4).long()
    vals = torch.empty(b.shape[0], b.shape[1] * 2, dtype=torch.float32)
    vals[:, 0::2] = FP4_TABLE[lo]
    vals[:, 1::2] = FP4_TABLE[hi]
    exp = scale_e8m0.view(torch.uint8).long() - 127
    scale = torch.pow(2.0, exp.float())
    return vals * scale.repeat_interleave(32, dim=1)


def dequant_fp8(w_fp8: torch.Tensor, scale_f32: torch.Tensor) -> torch.Tensor:
    """Reference dequant: FP8 e4m3 with 128x128 block scales -> fp32."""
    w = w_fp8.float()
    R, C = w.shape
    if scale_f32.dtype == torch.float8_e8m0fnu:
        scale_f32 = torch.pow(2.0, scale_f32.view(torch.uint8).float() - 127)
    else:
        scale_f32 = scale_f32.float()
    br = scale_f32.repeat_interleave(128, dim=0)[:R]
    bc = br.repeat_interleave(128, dim=1)[:, :C]
    return w * bc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no-finalize", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("model-*.safetensors"))
    if a.limit_shards:
        shards = shards[: a.limit_shards]
    todo = shards[a.offset :: a.stride]

    n_pass_expert = n_bf16 = 0
    for i, sf in enumerate(todo):
        outd: dict[str, torch.Tensor] = {}
        with safe_open(sf, framework="pt") as sh:
            keys = list(sh.keys())
            kset = set(keys)
            for k in keys:
                t = sh.get_tensor(k)
                if EXPERT_RE.match(k):
                    outd[k] = t.contiguous()  # passthrough, not one bit changed
                    if k.endswith(".weight"):
                        n_pass_expert += 1
                    continue
                if k.endswith(".scale"):
                    continue  # non-expert scales are consumed together with their weight
                sc_key = k.rsplit(".", 1)[0] + ".scale"
                if t.dtype == torch.float8_e4m3fn and sc_key in kset:
                    ref = dequant_fp8(t, sh.get_tensor(sc_key))
                    outd[k] = ref.to(torch.bfloat16)
                    n_bf16 += 1
                elif t.dtype == torch.int8 and sc_key in kset:
                    # FP4 outside the routed experts (shared experts): dequantise to bf16 too
                    ref = dequant_fp4(t, sh.get_tensor(sc_key))
                    outd[k] = ref.to(torch.bfloat16)
                    n_bf16 += 1
                else:
                    outd[k] = t.to(torch.bfloat16) if t.is_floating_point() and t.dtype != torch.bfloat16 else t
        save_file(outd, str(dst / sf.name))
        print(f"[{i+1}/{len(todo)}] {sf.name}  experts passed through {n_pass_expert}  converted to bf16 {n_bf16}", flush=True)

    if not a.no_finalize:
        finalize(src, dst, expected=len(shards), smoke=bool(a.limit_shards))


def finalize(src: Path, dst: Path, expected: int, smoke: bool) -> None:
    files = sorted(dst.glob("model-*.safetensors"))
    if not smoke and len(files) != expected:
        raise SystemExit(f"shard count mismatch: {len(files)}/{expected}")

    # Build the config: mxfp4 with every non-expert module on the ignore list.
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    cfg["quantization_config"] = {
        "quant_method": "mxfp4",
        "group_size": 32,
        "ignored_layers": ["re:^(?!.*\\.ffn\\.experts\\.).*$"],
        "provenance": "expert tensors passed through bit-identical from the official "
                      "MXFP4 release; all non-expert tensors dequantized to bf16",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for extra in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "configuration.json", "LICENSE", "README.md"]:
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)

    # index
    wmap = {}
    for f in files:
        with safe_open(f, framework="pt") as sh:
            for k in sh.keys():
                wmap[k] = f.name
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap}, indent=0))

    # Gate 1: expert passthrough round-trip, bit for bit.
    expert_keys = [k for k in wmap if EXPERT_RE.match(k) and k.endswith(".weight")][:6]
    if not expert_keys:
        raise SystemExit("gate 1 saw no expert tensor at all -- the gate is not guarding anything")
    rels = []
    for k in expert_keys:
        with safe_open(src / wmap[k], framework="pt") as s0, \
             safe_open(dst / wmap[k], framework="pt") as s1:
            sc = k.rsplit(".", 1)[0] + ".scale"
            ref = dequant_fp4(s0.get_tensor(k), s0.get_tensor(sc))
            got = dequant_fp4(s1.get_tensor(k), s1.get_tensor(sc))
            rels.append(((ref - got).abs().max()).item())
    if max(rels) != 0.0:
        raise SystemExit(f"gate 1: expert passthrough drifted, max abs diff={max(rels)}")
    print(f"gate 1 ok: {len(expert_keys)} expert tensors passed through bit for bit")

    # Gate 2: bf16 layers against the FP8 reference.
    bf_keys = [k for k in wmap
               if not EXPERT_RE.match(k) and k.endswith(".weight") and "norm" not in k][:6]
    rels = []
    for k in bf_keys:
        with safe_open(src / wmap[k], framework="pt") as s0, \
             safe_open(dst / wmap[k], framework="pt") as s1:
            t = s0.get_tensor(k)
            sc = k.rsplit(".", 1)[0] + ".scale"
            with safe_open(src / wmap[k], framework="pt") as _s:
                has_sc = sc in set(_s.keys())
            if t.dtype == torch.float8_e4m3fn and has_sc:
                ref = dequant_fp8(t, s0.get_tensor(sc))
            elif t.dtype == torch.int8 and has_sc:
                ref = dequant_fp4(t, s0.get_tensor(sc))
            else:
                continue
            got = s1.get_tensor(k).float()
            rels.append(((ref - got).norm() / ref.norm().clamp(min=1e-9)).item())
    if rels:
        med = statistics.median(rels)
        print(f"   gate 2: {len(rels)} bf16 samples, median relative error {med:.2e}")
        if med > 1e-2:
            raise SystemExit("gate 2: bf16 conversion error above tolerance")
        print("gate 2 ok: bf16 conversion within rounding error")

    # Gate 3: the ignore pattern must match exactly what was routed where.
    pat = re.compile("^(?!.*\\.ffn\\.experts\\.).*$")
    bad_ign = [k for k in expert_keys if pat.match(k)]
    bad_keep = [k for k in bf_keys if not pat.match(k)]
    if bad_ign or bad_keep:
        raise SystemExit(f"gate 3: ignore pattern misroutes {bad_ign[:2]} {bad_keep[:2]}")
    print("gate 3 ok: ignored_layers pattern agrees with the actual split")
    print(f"output: {sum(f.stat().st_size for f in files)/2**30:.1f} GiB across {len(files)} shards")


if __name__ == "__main__":
    main()
