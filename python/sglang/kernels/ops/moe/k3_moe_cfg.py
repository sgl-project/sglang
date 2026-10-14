# SPDX-License-Identifier: Apache-2.0
"""K3 decode MoE kernel configs for the AITER A8W4 runner.

aiter's kimik3_a8w4_tuned_fmoe.csv was tuned on uniform-random routing. DFlash verify
batches are 8 consecutive tokens per request, which route to ~40% fewer distinct experts
(M=16: ~139 vs 224), and at that shape the tuned M=16 stage1 (``_xcd4`` swizzle) is ~11 us
slower than the plain variant. These rows were re-tuned on real K3 routing:
same activation/weight quantization, same layouts, only the
tile schedule changes, so outputs match the default kernels to fp32-reference noise.

Installed by patching the K3 shape's rows in aiter's in-memory tuned table the first time
fused_moe looks one up; every other shape is untouched.
"""

from __future__ import annotations

import functools

# K3 TP8 routed experts: (model_dim, inter_dim, experts, topk)
_K3_SHAPE = (3584, 384, 896, 16)

_S2_FLY = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32"
# padded token tier -> (kernelName1, kernelName2, block_m)
K3_DECODE_CFG = {
    8: ("flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8", _S2_FLY, 32),
    16: ("flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8", _S2_FLY, 32),
}


def _patch_table(fm) -> int:
    from aiter import dtypes

    table = fm.cfg_2stages[0] if fm.cfg_2stages else None
    if not table:
        return 0
    fp8 = str(dtypes.fp8)
    n = 0
    for key, row in list(table.items()):
        tok, shape, q_a = key[2], tuple(key[3:7]), key[9]
        if shape == _K3_SHAPE and q_a == fp8 and tok in K3_DECODE_CFG:
            kn1, kn2, bm = K3_DECODE_CFG[tok]
            table[key] = {
                **row,
                "kernelName1": kn1,
                "kernelName2": kn2,
                "block_m": bm,
                "ksplit": 0,
            }
            n += 1
    return n


@functools.cache
def apply_k3_moe_decode_cfg() -> None:
    try:
        import aiter.fused_moe as fm
    except ImportError:
        return
    orig = fm.get_2stage_cfgs
    if getattr(orig, "_k3_decode_cfg", False):
        return
    state = {"done": False}

    def get_2stage_cfgs(token, model_dim, inter_dim, expert, topk, *args, **kwargs):
        if not state["done"] and (model_dim, inter_dim, expert, topk) == _K3_SHAPE:
            res = orig(
                token, model_dim, inter_dim, expert, topk, *args, **kwargs
            )  # loads the table
            n = _patch_table(fm)
            state["done"] = True
            if n:
                orig.cache_clear()
            else:
                return res
        return orig(token, model_dim, inter_dim, expert, topk, *args, **kwargs)

    get_2stage_cfgs._k3_decode_cfg = True
    get_2stage_cfgs.cache_clear = orig.cache_clear
    fm.get_2stage_cfgs = get_2stage_cfgs
