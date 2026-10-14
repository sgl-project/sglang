"""K3 MLA target-verify (QLEN=8, 12 heads, fp8 KV latent 576) hand-scheduled HIP kernel (gfx950).

Same contract and split/merge scheme as k3_mla_verify_v2 (one CTA per (request,
KV split) covering all 8 q_pos x 12 heads, so the KV is read once; per-split
normalized partial O + base-2 LSE, merged by v2's Triton reduce), but stage 1
is jit/csrc/kimi_k3/k3_mla_verify_hk.cuh:

  * 4 waves = 3 compute waves (32 query rows each, the 96 rows exactly, no
    padding) + 1 loader wave that streams 64-token KV tiles global -> LDS with
    buffer_load_dwordx4 ... lds (4-stage ring, one s_barrier per tile).
  * S^T = K Q^T (FP8 MFMA 32x32x64, Q quantized per row like v2 and held in
    VGPRs) so each lane owns one query row; O^T += V^T P^T with P (FP8) kept
    in registers and O^T (32 x 512 fp32 per wave) in AGPRs; V read with
    ds_read_b64_tr_b8.
  * The inner loop is hand-scheduled inline asm (gen script next to the .cuh):
    the 18 QK MFMAs of tile i+1 are interleaved with the softmax VALU of tile
    i, then the 16 PV MFMAs of tile i stream the transposed V reads.

No host syncs; the grid depends only on bs (CUDA-graph capturable).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

_LOG2E = 1.4426950408889634

_CSRC = os.path.join(os.path.dirname(__file__), "..", "..", "jit", "csrc", "kimi_k3")


_GEN = "gen_k3_mla_verify_hk_asm.py"


def _src_hash() -> str:
    h = hashlib.sha1()
    for name in ("k3_mla_verify_hk.cuh", _GEN):
        with open(os.path.join(_CSRC, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _asm_include_dir(tag: str) -> str:
    """Directory holding k3_mla_verify_hk_asm.inc, generated from _GEN."""
    d = os.path.join(tempfile.gettempdir(), f"sglang_k3_mla_verify_hk_{tag}")
    out = os.path.join(d, "k3_mla_verify_hk_asm.inc")
    if not os.path.exists(out):
        os.makedirs(d, exist_ok=True)
        tmp = f"{out}.{os.getpid()}"
        subprocess.run([sys.executable, os.path.join(_CSRC, _GEN), tmp], check=True)
        os.replace(tmp, out)
    return d


@cache_once
def _module():
    tag = _src_hash()
    return load_jit(
        "kimi_k3_mla_verify_hk",
        tag,
        cuda_files=["kimi_k3/k3_mla_verify_hk.cuh"],
        cuda_wrappers=[
            ("run", "K3MlaVerifyHK::run"),
            ("reduce", "K3MlaVerifyHK::reduce"),
        ],
        extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-spill-vgpr-to-agpr=0"],
        extra_include_paths=[_asm_include_dir(tag)],
    )


# fp8 partials (O / kv_scale as e4m3, half the partial traffic) for requests with at least this
# many active splits; only the HIP reduce reads them.
PART8_MIN_SPLITS = 96


def hk_stage1(
    q,
    kv,
    kv_indptr,
    kv_indices,
    kv_scale,
    o_part,
    lse_part,
    out,
    nsplit: int,
    min_chunk: int,
    sm_scale: float,
    part8_min: int = 0,
) -> None:
    """Stage 1: partials into o_part / lse_part with the v2 split formula. part8_min > 0: requests
    with >= part8_min active splits store fp8 partials (merge with hk_reduce(part8_min=...) only)."""
    _module().run(
        q,
        kv.view(torch.uint8),
        kv_indptr,
        kv_indices,
        kv_scale,
        o_part,
        lse_part,
        out,
        nsplit,
        min_chunk,
        float(sm_scale) * _LOG2E,
        part8_min,
    )


def hip_reduce_enabled(nsplit: int) -> bool:
    """One-round-trip HIP split merge (k3_mla_verify_hk_reduce1, also merges fp8 partials) instead
    of v2's Triton reduce: faster for nsplit >= 128 (bs <= 2, graph replay, MI355X); the Triton
    reduce is ~0.5-1 us faster at bs 4-8."""
    return 128 <= nsplit <= 256


def hk_reduce(
    o_part,
    lse_part,
    kv_indptr,
    out,
    nsplit: int,
    min_chunk: int,
    kv_scale,
    part8_min: int = 0,
    regime=None,
    g_logits=None,
    g_lse=None,
    g_ns: int = 1,
    g_block_n: int = 128,
) -> None:
    """Merge the HK splits (o_part / lse_part, base-2 LSE) into out (part8_min: as passed to
    hk_stage1). With regime (int32 [1], auto path): regime == 0 -> merge the Gluon splits
    (g_logits / g_lse, natural-log LSE) instead."""
    use = regime is not None
    _module().reduce(
        o_part,
        lse_part,
        kv_indptr,
        out,
        regime if use else lse_part,
        g_logits if (use and g_ns > 1) else o_part,
        g_lse if (use and g_ns > 1) else lse_part,
        kv_scale,
        nsplit,
        min_chunk,
        int(use),
        g_ns,
        g_block_n,
        part8_min,
    )
