# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Adapted from FlashInfer (flashinfer-ai/flashinfer 449a35734, flashinfer/topk_varlen/kernels/gvr2_topk_host.py);
# the SGLang additions for K31HT_DSA_SELECT are marked "SGLang:".
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FlashInfer NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)

"""Self-sampling GVR top-K decode — host side (dispatch, workspace, entry).

Provenance: ported near-verbatim from TensorRT-LLM
``tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling_host.py``
(PR NVIDIA/TensorRT-LLM#17821, commit ed94d4cfbf). FlashInfer-local changes:
module rename and this note; ``_varlen_launcher`` keeps the kernels' row-read
bound at the physical width and inflates only the routing value (DKG #60);
``warmup_varlen`` populates the exact-row launchers on every call (DKG #60);
``validate_run_ws`` enforces the 16-byte alignment the compiled workspace
declares; ``run_varlen`` rejects overlapping row layouts; ``run``/``run_ws``/
``run_varlen`` re-enter under the logits device and every compile passes an
explicit ``--gpu-arch``; ``route()`` adds two register-kernel rungs in the
4K < n <= 8K band (VPT=2 for b <= sms, BLK=512 for sms < b <= 2*sms) and sizes its
register band by the device SM count ``sms`` instead of the hard-coded 148
(see the inline note; DKG #61); ``run_varlen`` accepts ``pre_idx=None`` and runs
the hint-free compiled engines (TRT-LLM #18410: identity sample, no hint loads;
``_varlen_launcher(..., hint_free=True)``).
FlashInfer's ``top_k_varlen(backend="gvr_2")``
calls ``run_varlen`` below; the batch-uniform ``run``/``run_ws`` entries are
kept for parity tests and benchmarking. Keep future diffs against upstream
mechanical.

Companion to ``gvr2_topk_decode.py`` (the device module).
Three sections:

1. dispatch — the CUDA host dispatch as a pure function
   ``route(b, n, npad, k)``;
2. workspace — one zero-initialised slab (20,973,568 B) per (device, CUDA
   stream) via the torch caching allocator (FlashInfer-local: upstream keys
   it per device), with keep-alive + double-checked locking, eager-only;
3. operator entry — ``run(logits, pre_idx, n_valid, indices)`` /
   ``run_ws(..., workspace)`` DPS forms with input hardening and a
   bind-once launch cache keyed on ``(b, n, npad, k)``.

OPERATOR CONTRACT (batch-uniform entries): ``n_valid`` is one host python
int for the whole batch — every row shares the same valid prefix, in
COMPRESSED index space (the caller applies any ``compressRatio`` division).
``pre_idx`` is consumed as-is — raw prev-step top-K indices, uniformly for
DSv3.2 / DSv4 Flash / Pro. The +1 temporal shift ``heuristicTopKDecode.cu``
applies for cr==1 is deliberately dropped: hints only steer the sampling
ladder (exactness never depends on them), and raw prev-step hints overlap
the current top-K at least as well as +1-shifted ones on real decode data,
so one offset-free hint convention serves all three models. The production
per-row contract (per-request ``kv_lens`` read on-device, per-row MTP
offsets — sync-free and CUDA-graph-replay safe with growing KV) is
implemented by ``run_varlen``, which is the entry the opt-in DSA dispatch
seam calls. The batch-uniform ``run``/``run_ws`` entries keep the simpler
contract (one host-side ``n_valid`` for the whole batch), are exercised for
unit tests and benchmarking only, and must not be substituted for
``run_varlen`` under continuous batching, MTP (``next_n > 1``), or
CUDA-graph capture.
"""

import math
import operator
import os
import threading
from collections.abc import Sequence

import torch

_dev_mod = None


def _device():
    """Lazy import of the merged device module (first routed shape compiles;
    a broken/absent device module only fails when actually reached)."""
    global _dev_mod
    if _dev_mod is None:
        try:
            from . import gvr2_topk_decode as _m  # in-tree
        except ImportError:  # standalone dir
            import gvr2_topk_decode as _m
        _dev_mod = _m
    return _dev_mod


_SM_COUNT = {}


def _sm_count() -> int:
    """SM count of the CURRENT device (per-device cache). FlashInfer-local
    knob of ``route()``'s register-resident band (``wide`` / one-wave tests);
    the streaming-path constants stay at upstream's 148 — see route()."""
    d = torch.cuda.current_device()
    v = _SM_COUNT.get(d)
    if v is None:
        v = _SM_COUNT[d] = int(torch.cuda.get_device_properties(d).multi_processor_count)
    return v


def _arch_token() -> str:
    """Compile-target token for the launcher caches (mirrors the device
    module's _compile_arch_token without forcing its cutlass import): a
    heterogeneous multi-GPU process must not reuse a launcher whose compiled
    engine targets another architecture. Reads the CURRENT device, exactly
    like the DSL compile target and the launch stream do — the public entries
    pin the current device to ``logits.device`` first (_on_other_device)."""
    import os

    env = os.environ.get("CUTE_DSL_ARCH")
    if env:
        return env
    try:
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    except Exception:  # noqa: BLE001 — no GPU context: single-arch fallback
        return "unknown"


def _on_other_device(logits: torch.Tensor) -> bool:
    """True when ``logits`` lives on a CUDA device other than the current one.

    The launcher cache keys (_arch_token), the DSL compile target
    (_compile_arch_token) and the launch stream all follow the CURRENT
    device, so ``run`` / ``run_ws`` / ``run_varlen`` re-enter themselves under
    ``torch.cuda.device(logits.device)`` in that case: a heterogeneous
    multi-GPU process must never cache, compile for, or launch on the wrong
    architecture. One device-index compare on the common (same-device) path."""
    return logits.is_cuda and logits.get_device() != torch.cuda.current_device()


# ===========================================================================
# ==== dispatch =============================================================
# ===========================================================================
"""Pure-Python mirror of the GVR CUDA host dispatch (gvr_topk_launch).

route(b, n, npad, k) is a PURE function of its four ints -- no env knobs, no
GPU, stdlib only.  It returns the kernel family, its compile-time template
tuple, the runtime scalar pack `rt`, grid/cluster/block geometry, smem size,
and whether the family needs the workspace.

rt carries the FULL runtime scalar list each kernel receives, in signature
order, always starting with (n, npad, k).

Dead ABI-parity args: gvr_main's `int SCAP_, int CMP_` params are NEVER read
by the kernel body -- it recomputes them as constexprs that mirror the host
formulas bit-identically.  They are kept in rt purely for ABI parity.
gvr_clus's SCAP/CMP are LIVE runtime args.  `aim` and `SFAC` are host-side
intermediates only (never cross the ABI), so they do not appear in rt.

C-semantics notes encoded here:
  * every `/` on ints is C truncating division -> Python `//` (all operands
    are non-negative on every reachable path);
  * `sel = (long long)SFAC * n / aim` and the TGT/TGT2 products are 64-bit in C;
    Python ints are exact, so `//` reproduces them;
  * `int r = (int)(0.5 + sqrt((double)(6LL*n)))` truncates toward zero after
    the +0.5 -> `int(0.5 + math.sqrt(float(6*n)))`;
  * `IMGW = (n + 3) & ~3` four-element float4 round-up;
  * the reg-block CMP (possibly widened to n by DEGE) is scoped to the
    register-resident block; the streaming path re-derives its own CMP.
"""


# ---- dispatch constants (must match the device kernels) ---------------------
NB = 1024  # register-path histogram bins
QUADC = 96  # crossing-bin O(mc^2) rank gate (streaming/reg paths)
SNB = 256  # streaming-path bin count
CMPC = 4096  # crossing-bin slots per CTA, clustered register path
BLKC = 1024  # CTA size of the clustered register path
# SGLang: (K31HT_DSA_SELECT, the only user of this copy): the non-SPLIT streaming main family runs the
# BLK 512 / MINB 2 template up to BLK512_MAX_ROWS rows and BLK 256 / MINB 4 above. Upstream uses 296; the default here
# is 424.
# K31HT_DSA_SELECT_BLK512_MAX_ROWS overrides it (296 = upstream). Read once at import.
_BLK512_MAX_ROWS_ENV = os.environ.get('K31HT_DSA_SELECT_BLK512_MAX_ROWS', '424')
if not _BLK512_MAX_ROWS_ENV.isdigit():
    raise ValueError(f'K31HT_DSA_SELECT_BLK512_MAX_ROWS must be a non-negative integer, got {_BLK512_MAX_ROWS_ENV!r}')
BLK512_MAX_ROWS = int(_BLK512_MAX_ROWS_ENV)

# SGLang: the candidate-output main family (K31HT_DSA_SELECT=gvrmf) compiles the warp-ballot P5 emit (p5b 1)
# for calls with num_rows >= P5B_MIN_ROWS. 'auto' (default) = SM count + 1: p5b 1 is slower while each row's CTA has
# its SM to itself and faster once CTAs share SMs (rows > SM count). A non-negative integer overrides (0 = always; a huge value = never). Same emitted set either way.
_P5B_MIN_ROWS_ENV = os.environ.get('K31HT_DSA_SELECT_P5B_MIN_ROWS', 'auto')
if not (_P5B_MIN_ROWS_ENV == 'auto' or _P5B_MIN_ROWS_ENV.isdigit()):
    raise ValueError(f"K31HT_DSA_SELECT_P5B_MIN_ROWS must be 'auto' or a non-negative integer, got {_P5B_MIN_ROWS_ENV!r}")
P5B_MIN_ROWS = None if _P5B_MIN_ROWS_ENV == 'auto' else int(_P5B_MIN_ROWS_ENV)
p5b_calls = 0  # candidate launches that ran the p5b 1 compile (host counter; graph replays not counted)

# SGLang: tile unroll U of the BLK 512 non-SPLIT main template (rows <= BLK512_MAX_ROWS) in both route functions.
# '4' (default) or '8'; '4' = 4 values per thread per tile (half the tile), the whole next tile prefetched (same
# selected set). The BLK 256 template and the 1024-thread slab keep their U. Set once per process (read at import; plan caches do not key on it).
_BLK512_U_ENV = os.environ.get('K31HT_DSA_SELECT_BLK512_U', '4')
if _BLK512_U_ENV not in ('8', '4'):
    raise ValueError(f"K31HT_DSA_SELECT_BLK512_U must be '8' or '4', got {_BLK512_U_ENV!r}")
BLK512_U = int(_BLK512_U_ENV)


def p5b_for(num_rows, sms):
    """1 if a candidate-output call of num_rows rows uses the warp-ballot P5 emit, else 0 (read at every call)."""
    lim = sms + 1 if P5B_MIN_ROWS is None else P5B_MIN_ROWS
    return 1 if num_rows >= lim else 0


def _big_regime(b: int, R: int, n4: int) -> bool:
    """Streaming-slab regime selector (one place for route / route_dynamic /
    route_streaming / the varlen launcher).

    Upstream: ``b * R <= 148`` (the grid fits one wave) selects the "big"
    configuration: 1024-thread CTAs, one per SM, with the large sample caps
    (SCAP 16384, CMP 4096). FlashInfer-local: UNSPLIT rows (R == 1) in the
    75-148 row band with rows of <= 64K columns take the b > 148 configuration
    instead ((512, U 8, MINB 2): 512 threads, the smaller caps). Measured
    same-node, K in {512, 1024, 2048}, N 32K-64K, 75-148 rows: 1.5-2.5x faster
    on B200, B300 and B100, 1.5-2.6x on DRIVE P2021 (68 SMs), 1.2-2.1x on Rubin;
    forcing the U=8 unroll on the 1024-thread kernel changes nothing and the
    host sample constants do not matter, so it is the 512-thread CTA shape with
    its BLK-derived caps. Outside that band it is not a win: 33-74 unsplit rows
    (K = 2048 only reaches the slab there) lose at 64K (0.75x at 37 rows) and
    rows above 64K are mixed to bad (0.5x at 75-148 rows x 256K), so both keep
    upstream's regime, as do the split slab (R > 1) and b > 148."""
    return b * R <= 148 and not (R == 1 and b >= 75 and n4 <= 16384)


def route(b: int, n: int, npad: int, k: int, sms: int = 148) -> dict[str, object]:
    """Mirror of the CUDA gvr_topk_launch dispatch. Pure. See module doc.

    FlashInfer-local deviations, marked inline: the register-resident band
    is sized by ``sms`` (the device SM count; upstream hard-codes 148) — the
    ``wide`` one-wave test, QC/CURE and the one-wave cutoff of the BLK=512
    rung — and the 4K < n <= 8K band has two extra rungs (wide VPT=2, and
    BLK=512 for sms < b <= 2*sms, extended to 4*sms for n >= 6144 and to
    1024 rows for K = 2048). The streaming-path constants keep upstream's
    148 on every part."""
    if b < 1:
        raise RuntimeError(f"route requires b >= 1, got {b}")
    # 148 is the B200 SM count, baked in by the upstream CUDA dispatch (route()
    # is a pure mirror of it). It only steers occupancy heuristics (one-wave
    # tests, split factors, cluster sizing), never correctness. FlashInfer-
    # local: the REGISTER-band tests use the real SM count `sms` (B300 160,
    # Rubin 208): rows in (148, sms] are one wave of 1024-thread CTAs there
    # and the register kernels beat the slab by 1.4-1.8x (B300 12288 x 160:
    # 7.2 -> 4.6 us; Rubin 12288 x 160-192: 5.6 -> 4.0 us; measured same-node
    # vs sglang, DKG #61). The STREAMING constants below stay at 148: scaling
    # them too made the 131072 x 192-256 slab cells 3-28% slower on both parts.
    wide = b <= sms

    # ======================= register-resident block ========================
    n4 = n >> 2
    CMP = n if n < 2560 else 2560
    QC = 1024 if b > sms else QUADC
    CURE = not (n < 2 * k and b > sms)
    DEGE = (n <= 3 * k) or (n <= 4 * k + 64)
    if DEGE and CMP < n:
        CMP = n
    # FlashInfer-local: the `n4 <= 2048 and not wide` window runs the BLK=512
    # register kernel with NBH = NB for b <= 296 (upstream: main slab, which
    # ignores NBSEL), so the NB selector must follow (IMGOFF == NBSEL == NBH
    # is asserted at launch).
    NBSEL = (2 * NB) if (n4 > 512 and not (n4 <= 2048 and not wide)) else NB
    IMGOFF = NBSEL
    smem_reg = (NBSEL + 2 * CMP) * 4

    def _reg(BLK, VPT, MINB, NBH):
        # DEG wins over the CUR flag; DEG forces KPT=1, else KPT ladder 1/2/4.
        if DEGE:
            tpl = (BLK, VPT, MINB, 1, CURE, True, False, NBH)
        else:
            kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else 4)
            tpl = (BLK, VPT, MINB, kpt, CURE, False, False, NBH)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # full ABI
                "CMP": CMP,
                "IMGOFF": IMGOFF,
                "QC": QC,
            },
            "grid": (b, 1),
            "cluster": 1,
            "block": BLK,
            "smem": smem_reg,
            "ws": False,
        }

    IMGW = (n + 3) & ~3
    smi = (NBSEL + (2 * CMP if 2 * CMP > IMGW else IMGW)) * 4
    IMGE = wide and (not DEGE) and k <= 1024

    if n4 <= 256:
        return _reg(256, 1, 8, NB)
    if n4 <= 512:
        return _reg(512, 1, 4, NB)
    if n4 <= 1024:
        if wide:
            if IMGE:
                # regimg launch: gvr_topk_reg<1024,1,2,1,true,false,true,2048>
                return {
                    "kernel": "regimg",
                    "tpl": (1024, 1, 2, 1, True, False, True, 2 * NB),
                    "rt": {
                        "n": n,
                        "npad": npad,
                        "k": k,  # full ABI
                        "CMP": CMP,
                        "IMGOFF": IMGOFF,
                        "QC": QC,
                    },
                    "grid": (b, 1),
                    "cluster": 1,
                    "block": 1024,
                    "smem": smi,
                    "ws": False,
                }
            return _reg(1024, 1, 2, 2 * NB)
        return _reg(512, 2, 4, NB)

    # ---- clustered register-resident path ----
    # FlashInfer-local: K > BLKC (K = 2048) is admitted too. Upstream gates the
    # family at k <= BLKC because its hint sample is one word per thread; the
    # selection stages (merged scan, cursor emit, crossing-bin tie select,
    # key-space fallback) are k-generic, and the identity write of short rows
    # strides over k. The bracket then comes from the first BLKC row values
    # instead of the first k (a looser bracket, same exact result). Measured
    # B200, K = 2048, random lengths, vs the streaming slab it replaces:
    # 1.4-2.3x for <= 32 rows at 20K-64K and for <= 64 rows at <= 32K; the
    # 33-37 row band above 32K needs 4-CTA clusters that no longer fit one GPC
    # wave (0.89x at 37 x 48K), hence the one-wave gate below for K > BLKC
    # above 32K: b * cs <= 7/8 of the SM count (129 on B200/B300-148: 32 x 4
    # admitted, 37 x 4 not; 182 on Rubin-208: 37-44 x 4 admitted, measured
    # 2.1-2.3x vs the slab there, 52 x 4 not, 1.2-1.35x, left on the slab).
    if n4 > 4096 and n4 <= 8 * BLKC * 4 and k <= 2 * BLKC:
        # FlashInfer-local: cluster availability from the real SM count on
        # parts with >= 148 SMs (Rubin 208: 4-CTA instead of 2-CTA clusters at
        # b 38-48, and reg_clus instead of main/clus at b 80-104 / 48K-64K,
        # 8-35% faster; B300 158-160: 12-34%); below 148 the upstream value
        # stays — on DRIVE P2021 (68 SMs) the SM-aware rule won 34/44 measured
        # cells but lost 10 by up to 40% (falls to clus/main at 48K-128K).
        av = max(sms, 148) // (b if b > 0 else 1)  # truncating
        amax = 1
        while (amax << 1) <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            # cs=8 co-residency veto: an 8-CTA cluster with b > 15 exceeds
            # GPC packing; such shapes fall through to the streaming path.
            for v in (1, 2, 4):
                c = 1  # 64-bit product in C
                while c * BLKC * v < n4:
                    c <<= 1
                if c == 8 and b > 15:  # the veto
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2 and (k <= BLKC or n4 <= 8192 or b * cs <= (7 * sms) // 8):
            smc = (3 * NB + 2 * CMPC) * 4
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},  # dims only
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": smc,
                "ws": False,
            }

    # FlashInfer-local rungs for the 4K < n <= 8K band (the upstream dispatch
    # runs `wide` rows here on the VPT=4 kernel below and everything else on
    # the streaming slab). Measured on B100/B200, K in {512, 1024, 2048},
    # exact in every cell (logs/sglang_perf/force_plan*.py):
    #   * wide (b <= 148): VPT=2 covers n4 <= 2048 exactly; the VPT=4 kernel
    #     iterates two empty float4 slots per thread in every pass and is
    #     ~18-20% slower (4.2 -> 3.4 us at n=8192, K=512).
    #   * sms < b <= 2*sms (one wave at MINB=2, two CTAs per SM): the
    #     BLK=512/VPT=4 register kernel beats the main slab by 1.3-1.7x
    #     (8.0 -> 4.9 us at b=256, n=8192, K=512; Rubin 8192 x 400: 6.3 ->
    #     4.2 us). A second wave (b <= 4*sms) still wins for the upper half
    #     of the band (n >= 6144: B200 b=512 +11..12%, and the ragged-row
    #     6144-8192 x 320-400 cells where the slab lost to sglang by up to
    #     22%) but loses for the lower half (n=4160, b=512: -13%), so the
    #     slab keeps that.
    #   * K = 2048 up to 1024 rows, the whole band: the BLK=512 register
    #     kernel beats the slab by 15-39% in every measured cell (uniform and
    #     mixed lengths, n in {4160, 6144, 8192}, b in [600, 1024]) on B200,
    #     148-SM B300 and Rubin (2026-09 same-node A/Bs). TRT-LLM #18410 ships
    #     this as an SM103-only exception; it holds on SM100 and SM107 too.
    #     For K <= 1024 the same extension wins 20-40% on mixed lengths but
    #     loses 5-26% on uniform full-length rows at n=4160 and n=8192, so
    #     those keep the rule above.
    # Reported upstream for adoption as DKG issue #61.
    if n4 <= 2048:
        if wide:
            return _reg(1024, 2, 1, 2 * NB)
        if b <= 2 * sms or (b <= 4 * sms and n4 >= 1536) or (k == 2048 and b <= 1024):
            return _reg(512, 4, 2, NB)
    if n4 <= 4096 and wide:
        return _reg(1024, 4, 1, 2 * NB)

    # ====================== streaming / collect path ========================
    R = 1
    if b <= 32:
        r1 = 148 // b
        if r1 < 1:
            r1 = 1
        r2 = ((n >> 2) + 1023) // 1024
        if r2 < 1:
            r2 = 1
        R = r1 if r1 < r2 else r2
        if R < 1:
            R = 1
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:  # shallow R=2 split
        R = 2

    useclus = False
    if 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        # gvr_clus cs=8 hits the same GPC packing wall as the clustered
        # register path; same veto, same b > 15 threshold.
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True

    big = _big_regime(b, R, n >> 2)
    SCAP = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)
    CMP = (4096 if k > 1024 else 2048) if big else 1024

    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    q = 6 * n  # 6LL * n
    r = int(0.5 + math.sqrt(float(q)))  # C cast trunc
    if r > aim:
        aim = r
    SFAC = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (SCAP >> 1):
        aim = SCAP >> 1
    if aim < k:
        aim = k

    n4s = n >> 2
    SMP, SS2, TGT, TGT2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= SCAP and n > 2 * k
    if (n > SCAP or small_dense) and n4s >= 4:  # PAIR sample
        sel = SFAC * n // aim  # 64-bit
        if sel < 256:
            sel = 256
        if sel > n // 2:
            sel = n // 2
        pairs = sel >> 3
        if pairs < 1:
            pairs = 1
        half = n4s >> 1
        if half < 1:
            half = 1
        if pairs > half:
            pairs = half
        SS2 = half // pairs
        if SS2 < 1:
            SS2 = 1
        SMP = half // SS2
        if SMP < 1:
            SMP = 1
        TGT = (aim * (SMP * 8)) // n  # 64-bit
        if TGT < 1:
            TGT = 1
        TGT2 = (k * (SMP * 8)) // n  # 64-bit
        if TGT2 < 1:
            TGT2 = 1
    Q = (n4s + R - 1) // R

    if useclus:
        if n > SCAP and n4s >= 4:  # QUAD override
            sel = SFAC * n // aim
            if sel < 256:
                sel = 256
            if sel > n // 2:
                sel = n // 2
            quads = sel >> 4
            if quads < 1:
                quads = 1
            quarter = n4s >> 2
            if quarter < 1:
                quarter = 1
            if quads > quarter:
                quads = quarter
            SS2 = quarter // quads
            if SS2 < 1:
                SS2 = 1
            SMP = quarter // SS2
            if SMP < 1:
                SMP = 1
            TGT = (aim * (SMP * 16)) // n
            if TGT < 1:
                TGT = 1
            TGT2 = (k * (SMP * 16)) // n
            if TGT2 < 1:
                TGT2 = 1
        smc = SNB * 8 + (SCAP + 4) * 8 + CMP * 8
        per = Q >> 10
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        CS = 2 if R == 2 else (4 if R == 4 else 8)
        return {
            "kernel": "clus",
            "tpl": (1024, U, 1, SNB, CS),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # ABI (live)
                "SCAP": SCAP,
                "CMP": CMP,
                "SMP": SMP,
                "TGT": TGT,
                "Q": Q,
                "SS2": SS2,
                "TGT2": TGT2,
            },
            "grid": (CS, b),
            "cluster": CS,
            "block": 1024,
            "smem": smc,
            "ws": False,
        }

    smem_main = (SCAP + 4) * (8 if (R > 1 or b <= BLK512_MAX_ROWS) else 4) + (CMP + 1) * 8

    def _main(BLK, MINB, U, SPLIT):
        # KPT ladder 1/2/4/8; grid = (R, b).
        kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else (4 if k <= 4 * BLK else 8))
        # TSH-floor staging gate.  The CUDA form is a grid-uniform RUNTIME
        # gate (gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768); here it
        # is a compile-time key -- per-launch semantics are identical
        # because the gate is uniform over the grid.
        tshg = bool(SPLIT) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (BLK, U, MINB, SNB, kpt, SPLIT, tshg),
            # SCAP_/CMP_ are dead ABI-parity args: gvr_main never reads them
            # (it recomputes them as constexprs).
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # full ABI
                "SCAP_": SCAP,
                "CMP_": CMP,
                "R": R,
                "SMP": SMP,
                "TGT": TGT,
                "Q": Q,
                "SS2": SS2,
                "TGT2": TGT2,
            },
            "grid": (R, b),
            "cluster": 1,
            "block": BLK,
            "smem": smem_main,
            "ws": True,
        }

    if big:
        per = Q >> 10
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, U, R > 1)  # SPLIT iff R>1
    if b <= BLK512_MAX_ROWS:
        return _main(512, 2, BLK512_U, False)
    return _main(256, 4, 8, False)


if __name__ == "__main__":
    smoke = [
        # (b, n, npad, k)                          expected family
        (64, 1024, 1024, 512),  # reg   n4<=256 rung (DEG: n<=3k)
        (64, 2048, 2048, 512),  # reg   n4<=512 rung
        (1024, 4096, 4096, 1024),  # reg   n4<=1024, b>148 -> (512,2,4)
        (64, 4096, 4096, 512),  # regimg wide !DEGE k<=1024
        (64, 4096, 4096, 1024),  # reg   wide but DEGE (n<=4k+64)
        (8, 65536, 65536, 1024),  # reg_clus (vsel=2, cs=8; b<=15 no veto)
        (16, 131072, 131072, 512),  # main  cs=8 veto fall-through -> SPLIT slab, tshg=True
        (64, 16384, 16384, 1024),  # reg   wide 4k fallback (1024,4,1)
        (64, 262144, 262144, 1024),  # clus  R=2 shallow cluster split
        (1, 1048576, 1048576, 1024),  # main  deep slab SPLIT R=148
        (20, 262144, 262144, 2048),  # main  k>1024 split (no useclus)
        (512, 131072, 131072, 1024),  # main  b>296 BLK=256
        (256, 6144, 6144, 2048),  # main  small_dense sample gate
        (256, 262144, 262144, 2048),  # main  KBIG-domain (k>1024), BLK=512 KPT=4
    ]
    for shp in smoke:
        print(shp, "->", route(*shp))


# ---------------------------------------------------------------------------
# two-time-scale dispatch split (per-row varlen / CUDA-graph groundwork)
# ---------------------------------------------------------------------------
# route(b, n, npad, k) factored into
#   route_static(b, n, npad, k)  — everything that must be frozen per launch:
#       family, compile tuple, grid, cluster, block, and the rt scalars that
#       change only at discrete n-thresholds;
#   route_dynamic(static, n)     — the n-continuous scalars a per-row kernel
#       recomputes from its own row length (the device code will mirror these
#       formulas): n, CMP (reg families), the sampling ladder
#       SMP/TGT/SS2/TGT2/Q (streaming families), and the reg-family smem
#       footprint.
# INVARIANT: merging route_dynamic back into route_static reproduces
# route() EXACTLY for every n. The policy of which n to freeze the static
# half at (e.g. max_seq_len) is a perf-only choice — the factorization
# itself is lossless.

_DYN_RT = {
    "reg": ("n", "CMP"),
    "regimg": ("n", "CMP"),
    "reg_clus": ("n",),
    "clus": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
    "main": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
}
_DYN_SMEM = ("reg", "regimg")  # smem depends on CMP/IMGW -> recomputed per n


def route_static(b: int, n: int, npad: int, k: int, sms: int = 148) -> dict[str, object]:
    """route() with the n-continuous fields redacted (see _DYN_RT/_DYN_SMEM).
    Constant on maximal n-intervals ("bands"); every redacted field is
    reconstructible from (static, n) by route_dynamic."""
    plan = route(b, n, npad, k, sms=sms)
    st = {key: (dict(val) if isinstance(val, dict) else val) for key, val in plan.items()}
    for f in _DYN_RT[st["kernel"]]:
        st["rt"].pop(f)
    if st["kernel"] in _DYN_SMEM:
        st.pop("smem")
    return st


def route_dynamic(static: dict[str, object], n: int) -> tuple[dict[str, object], int]:
    """Recompute the redacted n-continuous scalars from (static, n).
    Returns (rt_updates, smem). Must stay equivalent to route(); the
    device-side per-row engine mirrors exactly these formulas."""
    fam = static["kernel"]
    k = static["rt"]["k"]
    if fam in ("reg", "regimg"):
        dege = static["tpl"][5]
        cmp_ = n if dege else (n if n < 2560 else 2560)
        nbsel = static["rt"]["IMGOFF"]
        if fam == "regimg":
            imgw = (n + 3) & ~3
            smem = (nbsel + (2 * cmp_ if 2 * cmp_ > imgw else imgw)) * 4
        else:
            smem = (nbsel + 2 * cmp_) * 4
        return {"n": n, "CMP": cmp_}, smem
    if fam == "reg_clus":
        return {"n": n}, static["smem"]

    # streaming families (main / clus): the sampling-ladder scalars
    b = static["grid"][1]
    if fam == "clus":
        R = static["cluster"]
        scap = static["rt"]["SCAP"]
    else:
        R = static["rt"]["R"]
        scap = static["rt"]["SCAP_"]
    # the regime is a static-plan property (a band boundary at 64K for the
    # unsplit 75-148 row slab): clus is split-only (n-independent), main's
    # big configuration is the only one with 1024-thread CTAs
    big = _big_regime(b, R, 0) if fam == "clus" else static["tpl"][0] == 1024
    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    r_ = int(0.5 + math.sqrt(float(6 * n)))
    if r_ > aim:
        aim = r_
    sfac = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (scap >> 1):
        aim = scap >> 1
    if aim < k:
        aim = k

    n4s = n >> 2
    smp, ss2, tgt, tgt2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= scap and n > 2 * k
    if (n > scap or small_dense) and n4s >= 4:
        sel = sfac * n // aim
        sel = 256 if sel < 256 else sel
        sel = n // 2 if sel > n // 2 else sel
        pairs = max(sel >> 3, 1)
        half = max(n4s >> 1, 1)
        pairs = half if pairs > half else pairs
        ss2 = max(half // pairs, 1)
        smp = max(half // ss2, 1)
        tgt = max((aim * (smp * 8)) // n, 1)
        tgt2 = max((k * (smp * 8)) // n, 1)
    q_ = (n4s + R - 1) // R
    if fam == "clus" and n > scap and n4s >= 4:
        sel = sfac * n // aim
        sel = 256 if sel < 256 else sel
        sel = n // 2 if sel > n // 2 else sel
        quads = max(sel >> 4, 1)
        quarter = max(n4s >> 2, 1)
        quads = quarter if quads > quarter else quads
        ss2 = max(quarter // quads, 1)
        smp = max(quarter // ss2, 1)
        tgt = max((aim * (smp * 16)) // n, 1)
        tgt2 = max((k * (smp * 16)) // n, 1)
    return (
        {"n": n, "SMP": smp, "TGT": tgt, "Q": q_, "SS2": ss2, "TGT2": tgt2},
        static["smem"],
    )


def route_split(b: int, n: int, npad: int, k: int, sms: int = 148) -> dict[str, object]:
    """route_static + route_dynamic recombined — must equal route() exactly
    (the factorization fuzz in the unit tests asserts this)."""
    st = route_static(b, n, npad, k, sms=sms)
    dyn, smem = route_dynamic(st, n)
    plan = {key: (dict(val) if isinstance(val, dict) else val) for key, val in st.items()}
    plan["rt"].update(dyn)
    plan["smem"] = smem
    return plan


def route_streaming(
    b: int, n: int, npad: int, k: int, force_main: bool = False
) -> dict[str, object]:
    """route() restricted to its STREAMING half (main / clus) — the varlen
    capture policy: per-row kernels must be picked from the families that are
    correct for ANY row length, so the register-resident specialists are
    skipped even when the envelope n would normally land on them.  Where
    route() itself lands on main/clus this is IDENTICAL to route().
    force_main additionally skips the clus rounding, so the raw
    min(r1, r2) R matches the CUDA else-branch exactly."""
    if b < 1:
        raise RuntimeError(f"route_streaming requires b >= 1, got {b}")
    R = 1
    if b <= 32:
        r1 = max(148 // b, 1)
        r2 = max(((n >> 2) + 1023) // 1024, 1)
        R = max(min(r1, r2), 1)
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:
        R = 2
    useclus = False
    if not force_main and 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True
    big = _big_regime(b, R, n >> 2)
    scap = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)
    cmp_ = (4096 if k > 1024 else 2048) if big else 1024
    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    r_ = int(0.5 + math.sqrt(float(6 * n)))
    if r_ > aim:
        aim = r_
    sfac = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (scap >> 1):
        aim = scap >> 1
    if aim < k:
        aim = k
    n4s = n >> 2
    smp, ss2, tgt, tgt2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= scap and n > 2 * k
    if (n > scap or small_dense) and n4s >= 4:
        sel = min(max(sfac * n // aim, 256), n // 2)
        pairs = min(max(sel >> 3, 1), max(n4s >> 1, 1))
        half = max(n4s >> 1, 1)
        ss2 = max(half // pairs, 1)
        smp = max(half // ss2, 1)
        tgt = max((aim * (smp * 8)) // n, 1)
        tgt2 = max((k * (smp * 8)) // n, 1)
    q_ = (n4s + R - 1) // R
    if useclus:
        if n > scap and n4s >= 4:
            sel = min(max(sfac * n // aim, 256), n // 2)
            quads = min(max(sel >> 4, 1), max(n4s >> 2, 1))
            quarter = max(n4s >> 2, 1)
            ss2 = max(quarter // quads, 1)
            smp = max(quarter // ss2, 1)
            tgt = max((aim * (smp * 16)) // n, 1)
            tgt2 = max((k * (smp * 16)) // n, 1)
        smc = SNB * 8 + (scap + 4) * 8 + cmp_ * 8
        per = q_ >> 10
        u_ = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        cs = 2 if R == 2 else (4 if R == 4 else 8)
        return {
            "kernel": "clus",
            "tpl": (1024, u_, 1, SNB, cs),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "SCAP": scap,
                "CMP": cmp_,
                "SMP": smp,
                "TGT": tgt,
                "Q": q_,
                "SS2": ss2,
                "TGT2": tgt2,
            },
            "grid": (cs, b),
            "cluster": cs,
            "block": 1024,
            "smem": smc,
            "ws": False,
        }
    smem_main = (scap + 4) * (8 if (R > 1 or b <= BLK512_MAX_ROWS) else 4) + (cmp_ + 1) * 8

    def _main(blk_, minb_, u_, split_):
        kpt = 1 if k <= blk_ else (2 if k <= 2 * blk_ else (4 if k <= 4 * blk_ else 8))
        tshg = bool(split_) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (blk_, u_, minb_, SNB, kpt, split_, tshg),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "SCAP_": scap,
                "CMP_": cmp_,
                "R": R,
                "SMP": smp,
                "TGT": tgt,
                "Q": q_,
                "SS2": ss2,
                "TGT2": tgt2,
            },
            "grid": (R, b),
            "cluster": 1,
            "block": blk_,
            "smem": smem_main,
            "ws": True,
        }

    if big:
        per = q_ >> 10
        u_ = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, u_, R > 1)
    if b <= BLK512_MAX_ROWS:
        return _main(512, 2, BLK512_U, False)
    return _main(256, 4, 8, False)


_VARLEN_CACHE = {}

# ---------------------------------------------------------------------------
# first-call gate for compiled DSL kernel objects
# ---------------------------------------------------------------------------
# libcute_dsl_runtime's ``cuda_dialect_init_library_once`` (the per-kernel
# CUDA init that runs inside the FIRST invocation of a compiled kernel) is a
# double-checked once-init behind ONE process-wide spinlock. In
# nvidia-cutlass-dsl 4.6.3, 4.7.0 and 4.7.1 the thread that acquires the lock
# after another thread already initialised the same kernel returns WITHOUT
# releasing it, so the next first call of ANY other kernel in the process
# spins forever (fixed in 4.8.0.dev0 / nightlies >= 2026-08-18; reported as
# DKG issue, see PR #4986). Until the DSL floor is past the fix, concurrent
# first calls of one kernel object are serialized here; once the first call
# has returned (the once-init is synchronous inside it) the wrapper costs one
# list-element load per launch.
_GATE_LOCK = threading.Lock()
_GATED = {}  # id(compiled fn) -> (compiled fn kept alive, gated wrapper, first-call-done flag)


def _gate_first_call(raw):
    """Wrap a compiled DSL kernel object so its first invocation is exclusive."""
    with _GATE_LOCK:
        ent = _GATED.get(id(raw))
        if ent is not None:
            return ent[1]
        lock = threading.Lock()
        warm = [False]

        def gated(*args, _raw=raw, _lock=lock, _warm=warm):
            if _warm[0]:
                return _raw(*args)
            with _lock:
                out = _raw(*args)
                _warm[0] = True  # only after a completed first call
            return out

        _GATED[id(raw)] = (raw, gated, warm)
        return gated


def _first_call_done(gated_fn) -> bool:
    """TESTING: True once ``gated_fn`` (a ``_gate_first_call`` wrapper) has
    completed its first call, i.e. the compiled object's once-init already ran
    eagerly -- the state ``warmup_varlen`` / ``warmup_prefill`` promise for
    every launcher they populate."""
    with _GATE_LOCK:
        for _raw, gated, warm in _GATED.values():
            if gated is gated_fn:
                return bool(warm[0])
    raise KeyError("not a gated kernel object")


# Programmatic dependent launch for the register families of the decode path: the
# kernel waits (griddepcontrol.wait) before its first read, so with the launch
# attribute set the NEXT kernel's launch processing overlaps this kernel's tail
# (in a CUDA graph and in a serving stream alike). Measured on B200 decode cells
# (K=2048, N=4096, 16 rows): 4.65 -> 4.0 us per call in graph replay.
_DECODE_PDL = True
# clustered register family, paged output: winners staged into rank 0's smem
# and written in one coalesced epilogue (the paged transform applied there)
# instead of per-winner scattered stores with the transform inline
_REGCLUS_STAGE_OUT = True


def _reg_stage_out(num_rows: int, sms: int) -> bool:
    """Register family: staged (coalesced) emit for one-wave launches only. A
    compile dimension of the engine chosen by the row count, so
    ``warmup_varlen`` keys its band representatives on it as well: a band
    spanning the SM count holds two objects and both need an eager launch."""
    return num_rows <= sms


def _regclus_stage_out(num_rows: int, k: int, paged: bool) -> bool:
    """Clustered register family: staged emit for paged output, except K > 1024
    below 16 rows. Same role as ``_reg_stage_out`` (a compile dimension the
    warmup band key mirrors)."""
    return bool(_REGCLUS_STAGE_OUT) and paged and (k <= 1024 or num_rows >= 16)


def _varlen_launcher(num_rows, npad, k, n_env, next_n, cr, hint_free=False, page_shift=None):
    """Capture-time varlen plan + compiled launcher.  The gvr_main port is
    the universally correct fallback; specialist family tiers below.  Every
    choice here is a function of capture-stable quantities only — mirroring
    the in-tree runner's pick_tuning(graph_capture=...) discipline.

    ``hint_free`` selects the engines with the hint-gather sites compiled out
    (upstream PR NVIDIA/TensorRT-LLM#18410): the register families bracket on
    the first k row values they already hold, the streaming families run on the
    sample alone. Part of the cache key (distinct compiled objects).
    ``page_shift`` (log2 page size, hint-free only) selects the PAGED engines: the
    page table rides in the dead pre_idx slot and every emitted index leaves the
    kernel as a physical KV slot (see ``_pt_xf`` in the device module)."""
    sms = _sm_count()
    hint_free = bool(hint_free)
    paged = page_shift is not None
    if paged and not hint_free:
        raise RuntimeError("paged engines are hint-free only")
    pg = dict(paged=paged, page_shift=int(page_shift) if paged else 0)
    # every family stages the page-table row in smem when it is small (<= 4 KB):
    # entries needed for the widest row, rounded to a power of two
    pt_smem = 0
    if paged:
        need_pages = (min(n_env, npad) + (1 << page_shift) - 1) >> page_shift
        if need_pages <= 1024:
            pt_smem = 1 << max(need_pages - 1, 0).bit_length()
    key = (num_rows, npad, k, n_env, next_n, cr, hint_free, page_shift, _arch_token(), sms)
    hit = _VARLEN_CACHE.get(key)
    if hit is not None:
        return hit
    # Two envelopes (FlashInfer-local split, reported upstream as DKG #60):
    # `n_kernel` is what the kernels get as their per-row clamp bound and must
    # never exceed the physical row width, else a request whose kv length
    # exceeds k makes the register families read k + 1 elements at the row
    # stride (into the next row, then past the tensor) when N <= k; `n_route`
    # is the routing-only value, inflated to k + 1 so route() sees a
    # non-degenerate problem. With n_kernel <= k every row takes the
    # in-kernel short path (identity + -1 tail).
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    cr_shift = 0 if cr == 1 else 2
    dev = _device()
    # ---- route() parity, family tier 1: clustered register-resident --------
    # Admit reg_clus exactly where the free route picks it; its whole
    # admission window (n4 <= 32768) fits capture-frozen envelopes. The
    # choice is a pure function of this cache key, so CUDA-graph replay
    # safety is unchanged; per-row n / short-row handling lives in-kernel.
    plan_free = route(num_rows, n_route, npad, k, sms=sms)
    if plan_free["kernel"] == "reg_clus":
        # Paged output: winners staged into rank 0's smem and mapped in one
        # coalesced epilogue, which reads the page table from global (L2-hot
        # by then) instead of staging the table row in the prologue. Measured
        # B200 vs the scattered emit with the smem-staged table: 0.2-0.7 us
        # (3-9%) on every clustered cell except K = 2048 below 16 rows, where
        # the two-pass epilogue over 2048 words costs more than it saves (-0.3
        # us at 1 x 32K); those keep the scattered emit. Unpaged output stays
        # scattered: funnelling every rank's winners through rank 0 lost
        # 0.3-0.8 us at 16-64 rows there.
        stage_rc = _regclus_stage_out(num_rows, k, paged)
        fn = _gate_first_call(
            dev.get_compiled__regclus(
                tuple(plan_free["tpl"]),
                pdl=_DECODE_PDL,
                varlen=True,
                next_n=next_n,
                cr_shift=cr_shift,
                hint_free=hint_free,
                pt_smem=0 if stage_rc else pt_smem,
                stage_out=stage_rc,
                **pg,
            )
        )
        lc = ("reg_clus", fn, n_kernel)
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 2: register-resident (+img flavor) ----
    # Same admission rule as tier 1: exactly where the free route picks
    # reg/regimg (the whole small/mid-N band across all row counts). CMP/QC/
    # smem are envelope-derived launch constants -- in-kernel they are pure
    # capacity clamps (CMP), a fast-path threshold (QC) and the launch smem
    # size, all safe upper bounds for every per-row n <= envelope; per-row n
    # / short-row handling lives in-kernel.
    if plan_free["kernel"] in ("reg", "regimg"):
        # staged coalesced emit only for one-wave launches (rows <= SMs): it trades
        # scattered stores for a barrier + copy, a 5-25% win when latency-bound and
        # a 3-4% loss when the grid is several waves deep (measured B200/B300)
        stage_out = _reg_stage_out(num_rows, sms)
        fn = _gate_first_call(
            dev.get_compiled__reg(
                tuple(plan_free["tpl"]),
                pdl=_DECODE_PDL,
                varlen=True,
                next_n=next_n,
                cr_shift=cr_shift,
                hint_free=hint_free,
                pt_smem=pt_smem,
                stage_out=stage_out,
                **pg,
            )
        )
        rt_f = plan_free["rt"]
        lc = (
            "reg",
            fn,
            (
                n_kernel,
                rt_f["CMP"],
                rt_f["QC"],
                dev.STATIC_BYTES + plan_free["smem"] + 4 * pt_smem + (4 * k if stage_out else 0),
            ),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 3: cluster split (clus) ---------------
    # Same admission rule: exactly where the free route picks clus (the
    # large-N mid-rows band). SCAP/CMP are launch-stable (pure functions of
    # rows/CS/k — never of n) so the envelope values are the per-row values;
    # the sampling-ladder scalars (SMP/TGT/Q/SS2/TGT2) are dead launch slots,
    # re-derived per row in-kernel by the route_dynamic clus mirror.
    # Per-row n / short-row handling in-kernel.
    if plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = _gate_first_call(
            dev.get_compiled__clus(
                tuple(plan_free["tpl"]),
                scap=rt_f["SCAP"],
                cmp_=rt_f["CMP"],
                varlen=True,
                next_n=next_n,
                cr_shift=cr_shift,
                hint_free=hint_free,
                pt_smem=pt_smem,
                pdl=_DECODE_PDL,
                **pg,
            )
        )
        lc = (
            "clus",
            fn,
            (n_kernel, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    plan = route_streaming(num_rows, n_route, npad, k, force_main=True)
    tpl = tuple(plan["tpl"])  # (BLK, U, MINB, SNB, KPT, SPLIT, TSHG)
    rt = plan["rt"]
    r_const = rt["R"]
    # TSHG (tpl[6]) is dead under varlen (the ctor compiles the TSH
    # machinery in whenever SPLIT); normalize it out of the compile key so
    # row counts differing only in that slot share one engine
    fn = _gate_first_call(
        dev.get_compiled(
            tpl[:6] + (False,) + (next_n, cr_shift, r_const),
            hint_free=hint_free,
            pt_smem=pt_smem,
            pdl=_DECODE_PDL,
            **pg,
        )
    )
    big = _big_regime(num_rows, r_const, n_route >> 2)
    aim_base = (
        ((4 * k if k >= 1024 else 2 * k) if r_const == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = (
        (32 if r_const == 2 else (48 if k > 1024 else 16))
        if r_const > 1
        else (64 if k >= 1024 else 32)
    )
    amin = 3 * k if r_const == 2 else (7 * k) // 2
    sd_en = 1 if (k > 1024 and not big) else 0
    # TSH-floor staging: gate on SPLIT and K only. Gating additionally on
    # num_rows > 15 would strand small batches in SPLIT-main without the
    # staged floor (a distribution-dependent tail regression); the kernel
    # gates TSH per row at runtime anyway.
    tsh_en = 1 if (tpl[5] and k <= 1024) else 0
    # slot 0 (`n`) carries the envelope: the varlen prologue clamps each row's
    # kv-derived length to it, never to the row stride npad (arena tails)
    pre = (n_kernel, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
    tail = (aim_base, sfac, amin, sd_en, tsh_en)
    lc = ("main", fn, pre, tail)
    _VARLEN_CACHE[key] = lc
    return lc


_PREFILL_DSL_OK: bool | None = None


def _prefill_env_error() -> str | None:
    """Why the windowed engines cannot run on the CURRENT device, or None.
    Mirrors the public gvr_2 admission (the CuTe DSL importable; a compute-
    capability-10 part, i.e. SM100 / SM103 / SM107 = the ``_GVR_CCS`` list of
    the ``top_k_varlen`` checker) so ``warmup_prefill`` fails with one sentence
    instead of an opaque compile or launch error and ``prefill_ready`` can
    answer without compiling anything."""
    global _PREFILL_DSL_OK
    if _PREFILL_DSL_OK is None:
        try:
            from ...cute_dsl.availability import is_cute_dsl_available

            _PREFILL_DSL_OK = bool(is_cute_dsl_available())
        except ImportError:  # standalone dir
            import importlib.util

            _PREFILL_DSL_OK = importlib.util.find_spec("cutlass") is not None
    if not _PREFILL_DSL_OK:
        return "the CuTe DSL (nvidia-cutlass-dsl) is not importable"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        return (
            f"compute capability {major}.{minor} is outside the gvr_2 support set "
            "(SM100 / SM103 / SM107)"
        )
    return None


# ---- prefill (windowed) launcher cache --------------------------------------
# Windowed rows (TRT-LLM #18702 port) run either a register rung — when the
# caller's window bound fits a BLK=512 x VPT register batch, see
# _prefill_reg_route — or the streaming `main` family with R == 1 (one CTA per
# row, no SPLIT, no workspace publish). A slab launcher depends only on the row
# tier, k and a power-of-two envelope bucket, a rung launcher on its (BLK, VPT,
# MINB) plan — never on the exact row count or npad — so the cache stays
# bounded on a long-running server; the per-call envelope (ke clamp) rides in
# the `n` slot.
_PREFILL_CACHE: dict = {}
_PREFILL_ROW_SLAB = 32768  # gridDim.y <= 65535; slab so keys stay bounded
_PREFILL_TIER_ROWS = (75, 149, 297)  # (rows<=148, 149..296, >296) band reps
# The tier-0 plan is the BLK=1024 non-split slab, whose compile-time pair-sample
# gate is n > 16384, so under an envelope <= 16384 every row runs the unsampled
# path. The tier-1 BLK=512 plan samples above its own gate (4096 for k <= 1024,
# 8192 for k > 1024); <= 148-row launches take it when the envelope is above
# that gate by a margin (just above the gate the freshly sampling BLK=512 plan
# is slower than the unsampled BLK=1024 plan for b >= 32) and at most 16384
# (above that the BLK=1024 plan samples too and is the better slab). Upstream
# TRT-LLM refinement of #18702.
_PREFILL_T1_MARGIN = 256
_PREFILL_T1_MAX = 16384


def _prefill_scpb_tier1(k: int) -> int:
    return 8192 if k > 1024 else 4096


def _prefill_tier(rows: int, n_env: int, k: int) -> int:
    tier = 0 if rows <= 148 else 1 if rows <= 296 else 2
    if tier == 0 and _prefill_scpb_tier1(k) + _PREFILL_T1_MARGIN < n_env <= _PREFILL_T1_MAX:
        tier = 1
    return tier


def _prefill_bucket(n_env: int) -> int:
    # pow2-quantize the envelope so a growing envelope reuses one plan; cap at
    # 32768 because U=8 for every n>=32768 on the tier-0 arm.
    return min(1 << max(int(n_env) - 1, 1).bit_length(), 32768)


def _prefill_bucket_for(n_env: int, k: int) -> int:
    """Envelope bucket of a windowed launch. A bound at or below ``k`` makes
    every row an identity row; such launches share the smallest non-identity
    bucket's launcher (a launcher built for a larger envelope ranks shorter
    windows exactly), so they need no compile of their own and the buckets
    ``warmup_prefill`` compiles, ``prefill_ready`` checks and ``_launch_prefill``
    looks up are the same."""
    return _prefill_bucket(max(int(n_env), int(k) + 1))


# ---- family routing for windowed rows ---------------------------------------
# The window engines exist for two families: the register-resident `reg`
# kernels (BLK=512, VPT in {1,2,4}: the row is read once into registers) and
# the streaming `main` slab. The register rungs need the whole scan extent
# (max window length + up to 3 lead lanes) to fit their capacity, so they are
# only reachable when the caller's ``max_window_len`` bounds the windows; without
# it every launch takes the slab. A row that violates the bound on a register
# rung is reported as all -1 (never a truncated ranking). Table and numbers:
# _prefill_reg_route.
_PREFILL_REG_MAX_N4 = 2048  # register capacity ceiling: 512 threads x VPT 4 float4


def _reg_plan(blk: int, vpt: int, minb: int, n: int, k: int, b: int) -> dict:
    """Register-family plan for the non-``wide`` band (b > sms): mirror of
    route()'s register-band constants (CMP / QC / CURE / DEGE / NBSEL) for an
    explicit (BLK, VPT, MINB) rung — the windowed router picks rungs the decode
    router does not (occupancy variants), so it cannot go through route()."""
    sms = _sm_count()
    wide = b <= sms
    cmp_ = n if n < 2560 else 2560
    qc = 1024 if b > sms else QUADC
    cure = not (n < 2 * k and b > sms)
    dege = (n <= 3 * k) or (n <= 4 * k + 64)
    if dege and cmp_ < n:
        cmp_ = n
    n4 = n >> 2
    nbsel = (2 * NB) if (n4 > 512 and not (n4 <= 2048 and not wide)) else NB
    if dege:
        tpl = (blk, vpt, minb, 1, cure, True, False, nbsel)
    else:
        kpt = 1 if k <= blk else (2 if k <= 2 * blk else 4)
        tpl = (blk, vpt, minb, kpt, cure, False, False, nbsel)
    return {
        "kernel": "reg",
        "tpl": tpl,
        "rt": {"n": n, "npad": n, "k": k, "CMP": cmp_, "IMGOFF": nbsel, "QC": qc},
        "smem": (nbsel + 2 * cmp_) * 4,
    }


def _prefill_reg_route(rows: int, k: int, n_hint: int) -> dict | None:
    """Register rung (BLK=512, VPT, MINB) for windowed rows whose scan extent
    ``n_hint + 3`` (lead lanes) fits its capacity, or None for the slab.

    Per-part table from the causal-staircase sweep (rows 2K-32K, K in {512,
    1024, 2048}, window bounds filling each class; logs prefill_reg_sweep, same
    node vs sglang topk_v2, 3 interleaved reps, min; B200 and B300 agree within
    3%, Rubin differs; K=1024 was re-measured after the table was built and
    confirmed every pick -- its only off-cell is Rubin's L <= 2048 class at 16K
    rows, 5% behind the slab, inside the noise band, so the 16K gate stands):
      * L <= 2048: B200/B300 VPT=1/MINB=4, 1.6-2.0x the slab at every row
        count (K = 2048 never reaches this class: a <= 2048 window is an
        identity row, ``n_hint <= k`` above). Rubin: VPT=1/MINB=4 only
        while the launch is small (K <= 1024: <= 16K rows) — above that
        Rubin's slab is as fast or faster (up to 1.4x).
      * L <= 4096: B200/B300 VPT=2/MINB=4 (the decode rung), 1.6-1.7x the
        slab. Rubin: VPT=2/MINB=2 — the MINB=4 rung is register-starved there
        (1.4-2x slower than the slab) while MINB=2 is 1.4x faster; bounds
        <= 2560 (one empty float4 per thread) keep the small-launch gate of
        the class below (6-40% behind the slab at 8K-32K rows otherwise).
      * L <= 8192: K = 2048 -> VPT=4/MINB=2 (5-8% ahead of the slab on every
        part); K <= 1024 -> slab (the register rungs lose 10-25%).
      * above: slab (streaming main).
    Occupancy variants of the same rung (MINB 1/2/4, BLK 1024) were measured
    and lose everywhere else."""
    # Capacity: the register batch holds BLK*VPT float4 = cap elements and the
    # kernel's scalar tail lane covers up to 3 more (ntail = n - 4*n4 < 4), so a
    # rung ranks scan extents nv + lead <= cap + 3 exactly, i.e. any window
    # bound <= cap (the <= 3 lead lanes ride in the tail). Classes are keyed on
    # the bound itself so 2048 / 4096 / 8192-token prompts keep their rung.
    n_hint = int(n_hint)
    if n_hint > _PREFILL_REG_MAX_N4 * 4 or n_hint <= k:
        return None
    rows = int(rows)
    rubin = "107" in _arch_token()
    # Rubin's slab is the faster engine for short windows at high row counts
    # (K <= 1024: above 16K rows; K = 2048: above 4K rows). That gate covers
    # the VPT=1 class and the low half of the VPT=2 class (a 2049-2560 bound
    # lands on VPT=2 with one empty float4 per thread: measured 6-40% behind
    # the slab at 8K-32K rows on Rubin, ahead of it below).
    rubin_short_gate = rubin and ((k <= 1024 and rows > 16384) or (k > 1024 and rows > 4096))
    if n_hint <= 2048:
        if rubin:
            if rubin_short_gate:
                return None
            rung = (512, 1, 4)
        else:
            rung = (512, 1, 4)  # k == 2048 never reaches this class (n_hint <= k above)
    elif n_hint <= 4096:
        if rubin:
            if rubin_short_gate and n_hint <= 2560:
                return None
            rung = (512, 2, 2)
        else:
            rung = (512, 2, 4)
    else:
        if k != 2048:
            return None
        rung = (512, 4, 2)
    blk, vpt, minb = rung
    assert blk * vpt * 4 >= n_hint
    return _reg_plan(blk, vpt, minb, blk * vpt * 4, k, rows)


_PREFILL_CLASS_BOUNDS = (2048, 4096, 8192)  # register capacity classes (VPT 1 / 2 / 4)
# Every windowed call routes by ONE window-length bound: the caller's
# max_window_len when given, else the logits width (always a valid bound: no
# window is wider than its row). The bound picks a register rung whose capacity
# covers bound + 3 lead lanes, else the slab tier. A per-row split across two
# launches (register rung + slab, each skipping the other's rows) was measured
# and rejected: the slab launch that only skips still schedules one CTA per row
# at its 1-2 CTA/SM smem occupancy (~35 us per 32K rows on B200), and without a
# bound the register launch must take the largest class, which is the wrong
# kernel for short windows (VPT=4 on 2K windows: slower than the slab itself at
# K = 2048). The width is a loose bound for a packed multi-request batch (it
# sums every request's keys), and the tight one is host knowledge in every
# serving framework, so the API also takes it as max_window_len.


def _prefill_cache_key(fam: str, tier_or_tpl, k: int, n_bucket: int, abs_out: bool = False):
    # main: tiers 1/2 fix U, so the bucket does not change their engine —
    # collapse it to one key so warmup covers them with a single launch.
    # reg: the compile tuple + envelope constants identify the launcher.
    if fam == "main":
        tier = tier_or_tpl
        return (
            "main",
            tier,
            k,
            n_bucket if tier == 0 else 0,
            bool(abs_out),
            _arch_token(),
            _sm_count(),
        )
    return ("reg", tier_or_tpl, k, bool(abs_out), _arch_token(), _sm_count())


def _prefill_reg_launcher(plan: dict, k: int, abs_out: bool = False) -> tuple:
    """Windowed register-family launcher for a route() plan (hint-free varlen
    compile with the prefill flag). ABI: ``fn(logits, row_starts, kv_lens=window
    lengths, out, n_clamp, CMP, QC, smem)``."""
    rt = plan["rt"]
    tpl = tuple(plan["tpl"])
    key = _prefill_cache_key("reg", (tpl, rt["CMP"], rt["QC"], plan["smem"]), k, 0, abs_out)
    hit = _PREFILL_CACHE.get(key)
    if hit is not None:
        return hit
    dev = _device()
    fn = _gate_first_call(
        dev.get_compiled__reg(
            tpl,
            varlen=True,
            next_n=1,
            cr_shift=0,
            hint_free=True,
            prefill=True,
            prefill_abs=abs_out,
            stage_out=False,  # prefill slabs are many waves deep: direct emit
        )
    )
    lc = ("reg", fn, (rt["CMP"], rt["QC"], dev.STATIC_BYTES + plan["smem"]))
    _PREFILL_CACHE[key] = lc
    return lc


def _prefill_get(
    rows: int, k: int, n_hint: int, n_bucket: int, compile_ok: bool, abs_out: bool = False
) -> tuple:
    """Launcher for one row slab: register rung when the window bound admits
    it, else the slab tier. ``compile_ok=False`` (capture) returns None on a
    cache miss instead of compiling."""
    plan = _prefill_reg_route(rows, k, n_hint)
    if plan is not None:
        key = _prefill_cache_key(
            "reg",
            (tuple(plan["tpl"]), plan["rt"]["CMP"], plan["rt"]["QC"], plan["smem"]),
            k,
            0,
            abs_out,
        )
        lc = _PREFILL_CACHE.get(key)
        if lc is None and compile_ok:
            lc = _prefill_reg_launcher(plan, k, abs_out)
        return lc
    tier = _prefill_tier(rows, n_hint, k)
    lc = _PREFILL_CACHE.get(_prefill_cache_key("main", tier, k, n_bucket, abs_out))
    if lc is None and compile_ok:
        lc = _prefill_launcher(tier, k, n_bucket, abs_out)
    return lc


def _prefill_launcher(tier: int, k: int, n_bucket: int, abs_out: bool = False) -> tuple:
    """Windowed slab plan + compiled launcher: ``_varlen_launcher``'s main branch
    with r_const=1, split=False, hint_free=True and the prefill compile flag.
    SCAP_/CMP_ are envelope upper bounds; the `n` slot is filled per call."""
    key = _prefill_cache_key("main", tier, k, n_bucket, abs_out)
    hit = _PREFILL_CACHE.get(key)
    if hit is not None:
        return hit
    b_route = _PREFILL_TIER_ROWS[tier]
    n_route = max(n_bucket, k + 1)
    plan = route_streaming(b_route, n_route, n_route, k, force_main=True)
    if plan["kernel"] != "main":
        raise RuntimeError(f"prefill route did not land on gvr_main: {plan['kernel']}")
    rt = plan["rt"]
    if rt["R"] != 1:
        raise RuntimeError(f"prefill requires R==1 (got {rt['R']})")
    tpl = tuple(plan["tpl"])
    dev = _device()
    fn = _gate_first_call(
        dev.get_compiled(
            tpl[:6] + (False,) + (1, 0, 1), hint_free=True, prefill=True, prefill_abs=abs_out
        )
    )
    big = tier == 0
    # r_const==1 branch of the _varlen_launcher tuning scalars
    aim_base = (
        (4 * k if k >= 1024 else 2 * k) if big else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = 64 if k >= 1024 else 32
    amin = (7 * k) // 2
    sd_en = 1 if (k > 1024 and not big) else 0
    tail = (aim_base, sfac, amin, sd_en, 0)  # tsh_en=0 (split=False)
    lc = ("main", fn, (rt["SCAP_"], rt["CMP_"]), tail)
    _PREFILL_CACHE[key] = lc
    return lc


def _launch_prefill(
    lg, row_starts, kv_lens, idx, ws, k, n_clamp, n_hint, npad, abs_out=False
) -> None:
    """Launch the windowed engine over ``lg`` in row slabs. ABI (main varlen):
    ``fn(logits, pre_idx_slot=row_starts, out, ws, n, npad, k, SCAP_, CMP_, R,
    dead x5, kv_lens_slot=window lengths, tuning tail)``; the prefill compile
    reads ks from the pre_idx slot and the window length from the kv_lens slot
    and clamps ke to ``n_clamp`` (the logits width) in the ``n`` slot.
    ``n_hint`` (the window-length bound: the caller's ``max_window_len`` or, when
    none was given, the logits width; capture-stable either way) selects the
    engine: a register rung whose capacity covers it, else the slab tier and
    envelope bucket. Every engine clamps ``ke`` to the logits width; the slab
    ranks any window exactly, a register rung ranks exactly every row whose
    extent (window + lead lanes) fits its capacity and reports a longer row as
    all -1, never as a truncated ranking (``_prefill_reg_route``); a
    width-derived bound cannot be exceeded. ``abs_out`` selects the compiled
    variants that emit absolute logits columns (window-local + row start)
    instead of window-local indices."""
    num_rows = lg.shape[0]
    n_hint = min(max(int(n_hint), 1), int(n_clamp))
    n_bucket = _prefill_bucket_for(n_hint, k)
    capture = _is_capturing()
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        r1 = min(r0 + _PREFILL_ROW_SLAB, num_rows)
        rows = r1 - r0
        # the bound picks ONE engine for the slab: a register rung whose
        # capacity covers bound + 3 lead lanes, else the streaming main tier
        lc = _prefill_get(rows, k, n_hint, n_bucket, compile_ok=not capture, abs_out=abs_out)
        if lc is None:
            raise RuntimeError(
                "prefill launcher not compiled for this shape — warm up "
                "(warmup_prefill or one eager windowed call) before CUDA graph capture"
            )
        if lc[0] == "reg":
            # register rung ABI: (logits, row_starts, window lengths, out, n_clamp, CMP, QC, smem)
            lc[1](lg[r0:r1], row_starts[r0:r1], kv_lens[r0:r1], idx[r0:r1], n_clamp, *lc[2])
            continue
        if ws is None:
            # only the streaming main family takes the per-(device, stream)
            # slab; resolved here so register-rung launches never need one
            ws = _ws_hot.get(_ws_key(lg.get_device()))
            if ws is None:
                ws = default_workspace(lg)  # eager only: raises under capture
        _, fn, (scap, cmp_), tail = lc
        pre = (n_clamp, npad, k, scap, cmp_, 1, 0, 0, 0, 0, 0)
        fn(lg[r0:r1], row_starts[r0:r1], idx[r0:r1], ws, *pre, kv_lens[r0:r1], *tail)


_PREFILL_WARMUP_DONE: set = set()
_PREFILL_WARMUP_LOCK = threading.Lock()


def _prefill_warm_launch(lc: tuple, k: int, n_h: int, row_stride: int | None) -> None:
    """Launch a compiled windowed engine once, eagerly, on a one-row window of
    ``n_h`` columns at ``ks = 0``. The engines compile with symbolic shapes, so
    this exercises exactly the object a full-size slab launches: its first-call
    init (``_gate_first_call``) and, for the slab family, the CURRENT stream's
    default workspace slab both happen here instead of inside a later
    CUDA-graph capture."""
    dev = torch.cuda.current_device()
    n_w = (n_h + 3) // 4 * 4  # launch width: a 1-row view keys npad on shape[1]
    stride = row_stride if row_stride is not None else ((n_w + 256 + 255) // 256 * 256)
    if stride < n_w or stride % 4:
        stride = (max(stride, n_w) + 256 + 255) // 256 * 256
    logits = torch.zeros((1, stride), dtype=torch.float32, device=dev)
    lg = logits[:, :n_w]
    ks = torch.zeros((1,), dtype=_I32, device=dev)
    lens = torch.full((1,), n_h, dtype=_I32, device=dev)
    out = torch.empty((1, k), dtype=_I32, device=dev)
    if lc[0] == "reg":
        # register rung ABI (see _launch_prefill)
        lc[1](lg, ks, lens, out, n_w, *lc[2])
        return
    ws = default_workspace(lg)  # eager: creates this stream's slab when missing
    _, fn, (scap, cmp_), tail = lc
    pre = (n_w, stride, k, scap, cmp_, 1, 0, 0, 0, 0, 0)
    fn(lg, ks, out, ws, *pre, lens, *tail)


def warmup_prefill(
    top_k: int,
    max_cols: int,
    num_rows_list: Sequence[int] = (1, 149, 297),
    row_stride: int | None = None,
    absolute_indices: bool = False,
) -> None:
    """Compile, and launch once, every windowed (prefill) engine that a
    ``top_k_varlen(..., row_starts=...)`` call with this geometry can select,
    so that a later CUDA-graph capture compiles and allocates nothing. Call
    it before serving with the serving device current and on the stream that
    will capture: each engine is launched once on a one-row window, which
    runs its first-call initialisation, and the current stream's default
    workspace slab is created when missing (the streaming engine needs it; a
    register-resident engine does not). The first call for a key returns
    after a device synchronization; a repeat call for the same key returns
    at once, re-creating the slab if it was released in between.

    Coverage: a windowed call routes on its row count, ``top_k``, output
    frame and window-length bound (``max_window_len``, else the logits width),
    never on the exact width or row stride, and a call of more than 32768
    rows is issued in launches of at most 32768 rows, each routed on its own
    count. For every row count in ``num_rows_list`` and every power-of-two
    bucket of the bound from the first above ``top_k`` up to the one
    covering ``max_cols`` (buckets are capped at 32768 columns; wider bounds
    share that bucket), at both edges of the bucket, this compiles the
    streaming engine that bound takes and the register-resident engine
    (2048-, 4096- or 8192-column class) the host admits for it. A later call
    is covered when its ``top_k`` and ``absolute_indices`` match, its bound
    (or width) lies above ``top_k`` and at most ``max_cols``, and its row
    count falls in a band warmed by ``num_rows_list`` (the default list
    covers every row count); ``prefill_ready`` answers for one geometry.

    Parameters
    ----------
    top_k : int
        The call's ``top_k`` (512, 1024 or 2048).
    max_cols : int
        The largest window-length bound a covered call will route by: its
        ``max_window_len``, or its logits width when it passes none.
    num_rows_list : Sequence[int], optional
        Row counts to warm. The engines depend on the count only through a
        few bands, which the default covers for every count; pass the
        serving batch's counts to narrow or extend it.
    row_stride : int, optional
        Row stride of the one-row warm-up windows, in elements (a multiple
        of 4, at least the bound); ``None`` picks a padded stride. The
        windowed launchers do not key on the stride, so the default is right
        for any producer layout.
    absolute_indices : bool, optional
        Warm the absolute-column output variants, a distinct compiled set;
        call once per output frame the server uses. Default ``False``.

    Raises
    ------
    RuntimeError
        If the windowed engines cannot run on the current device: no CuTe
        DSL, no CUDA, or a compute capability outside SM100 / SM103 / SM107.
    """
    err = _prefill_env_error()
    if err is not None:
        raise RuntimeError(f"gvr_2 windowed (prefill) engines unavailable: {err}")
    dev = torch.cuda.current_device()
    k = int(top_k)
    max_cols = int(max_cols)
    lo = _prefill_bucket_for(k + 1, k)
    hi = _prefill_bucket_for(max_cols, k)
    buckets = []
    b = lo
    while b <= hi:
        buckets.append(b)
        b <<= 1
    if not buckets:
        buckets = [hi]
    # the launch slabs a requested row count is cut into (_launch_prefill)
    slabs = set()
    for rows in num_rows_list:
        rows = max(int(rows), 1)
        for r0 in range(0, rows, _PREFILL_ROW_SLAB):
            slabs.add(min(_PREFILL_ROW_SLAB, rows - r0))
    # one representative (family, plan, hint) per launcher key
    reps: dict = {}
    for rows in sorted(slabs):
        for bk in buckets:
            for n_h in sorted({max(bk // 2 + 1, k + 1), bk}):
                plan = _prefill_reg_route(rows, k, n_h)
                if plan is not None:
                    key = _prefill_cache_key(
                        "reg",
                        (tuple(plan["tpl"]), plan["rt"]["CMP"], plan["rt"]["QC"], plan["smem"]),
                        k,
                        0,
                        absolute_indices,
                    )
                    reps.setdefault(key, ("reg", plan, n_h))
                # the slab tier of this width: taken wherever the table admits
                # no rung for the bound (max_window_len, or the width itself)
                tier = _prefill_tier(rows, n_h, k)
                key = _prefill_cache_key("main", tier, k, bk, absolute_indices)
                reps.setdefault(key, ("main", (tier, bk), n_h))
    done_key = (
        dev,
        torch.cuda.current_stream(dev).cuda_stream,
        k,
        max_cols,
        tuple(sorted(int(r) for r in num_rows_list)),
        row_stride,
        bool(absolute_indices),
    )
    # this stream's default workspace slab, before the done-key short-circuit:
    # only an eager launch creates it and release_cached_resources() drops it
    # without touching the done set, so a re-warm after a release must bring
    # the capturing stream back to ready although every engine is already
    # compiled (same rule as warmup_varlen)
    default_workspace(torch.empty(0, dtype=torch.uint8, device=torch.device("cuda", dev)))
    with _PREFILL_WARMUP_LOCK:
        if done_key in _PREFILL_WARMUP_DONE:
            return
    for fam, spec, n_h in reps.values():
        if fam == "reg":
            lc = _prefill_reg_launcher(spec, k, absolute_indices)
        else:
            tier, bk = spec
            lc = _prefill_launcher(tier, k, bk, absolute_indices)
        _prefill_warm_launch(lc, k, n_h, row_stride)
    torch.cuda.synchronize()
    with _PREFILL_WARMUP_LOCK:
        _PREFILL_WARMUP_DONE.add(done_key)


def prefill_ready(
    num_rows: int, k: int, n_env: int, width: int | None = None, absolute_indices: bool = False
) -> bool:
    """Whether a windowed ``top_k_varlen`` call (``row_starts`` given) with
    this geometry would launch under CUDA-graph capture on the current device
    and stream without compiling or allocating: every engine the call would
    select is compiled and, where it selects the streaming engine, the
    current stream's default workspace slab exists. Lets a serving framework
    route around the engine under capture instead of catching the
    capture-time ``RuntimeError``. Compiles and launches nothing and never
    raises for an unusable environment.

    Parameters
    ----------
    num_rows : int
        Rows of the call (``logits.shape[0]``); ``0`` is always ready.
    k : int
        Its ``top_k``.
    n_env : int
        The window-length bound the call routes by: the ``max_window_len`` it
        will pass, or the logits width for a call without ``max_window_len``.
    width : int, optional
        The logits width when ``n_env`` is a caller-given ``max_window_len``
        (the bound is clamped to it, as the call does); leave ``None`` when
        ``n_env`` is the width itself.
    absolute_indices : bool, optional
        The call's ``absolute_indices`` (a distinct compiled set). Default
        ``False``.

    Returns
    -------
    bool
        ``True`` when every launch the call makes (one per slab of at most
        32768 rows) finds its compiled engine and, if that engine is the
        streaming one, the current stream already has its default workspace
        slab; ``False`` otherwise, and whenever the current device cannot run
        the windowed engines (no CuTe DSL, no CUDA, or a compute capability
        outside SM100 / SM103 / SM107). ``warmup_prefill`` covering the
        geometry, or one eager windowed call of this geometry on this
        stream, turns it ``True``.
    """
    if num_rows == 0:
        return True
    if _prefill_env_error() is not None:
        return False
    width = int(n_env) if width is None else int(width)
    n_env = min(max(int(n_env), 1), width)
    n_bucket = _prefill_bucket_for(n_env, k)
    # a slab (streaming main) launch also needs this (device, CURRENT stream)'s
    # default workspace slab, which only an eager launch on this stream creates
    # (warmup_prefill or one eager windowed call); register rungs never need it
    slab_ok = _ws_hot.get(_ws_key(torch.cuda.current_device())) is not None
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        rows = min(r0 + _PREFILL_ROW_SLAB, num_rows) - r0
        lc = _prefill_get(rows, k, n_env, n_bucket, compile_ok=False, abs_out=absolute_indices)
        if lc is None or (lc[0] == "main" and not slab_ok):
            return False
    return True


def route_bands(
    b: int,
    npad: int,
    k: int,
    n_lo: int | None = None,
    n_hi: int | None = None,
    sms: int = 148,
) -> list[tuple[int, int, dict[str, object]]]:
    """Enumerate maximal n-intervals on which route_static is constant.
    Dense O(n_hi - n_lo) scan of the pure host dispatch — an offline /
    engine-init tool (seconds for the 262144-token envelope), NOT a hot
    path. Returns [(n_lo, n_hi, static_plan), ...]. Pass the target part's
    SM count as ``sms`` (``_sm_count()`` on the device) or the bands describe
    the B200 dispatch."""
    lo = k + 1 if n_lo is None else max(n_lo, k + 1)
    hi = npad if n_hi is None else min(n_hi, npad)
    bands = []
    cur_key, cur_lo, cur_plan = None, lo, None
    for n in range(lo, hi + 1):
        st = route_static(b, n, npad, k, sms=sms)
        key = repr(st)
        if key != cur_key:
            if cur_key is not None:
                bands.append((cur_lo, n - 1, cur_plan))
            cur_key, cur_lo, cur_plan = key, n, st
    if cur_key is not None:
        bands.append((cur_lo, hi, cur_plan))
    return bands


# ===========================================================================
# ==== workspace ============================================================
# ===========================================================================
"""Default workspace slabs for the multi-CTA SPLIT path.

Semantics:
  * ONE zero-initialised slab per (device index, raw CUDA stream handle),
    lazily allocated through the torch caching allocator by the first EAGER
    slab-using launch on that stream (never under CUDA-graph capture);
  * keep-alive store: module dict `_ws_keep` (tensor refcount = keep-alive);
  * double-checked locking: lock-free hot-path load (a GIL-atomic dict get
    plays an acquire load), slow path re-checks under a mutex;
  * device index bounds `0 <= d < GVR_MAX_DEV` -- checked BEFORE the
    CUDA-ness of the tensor (run() resolves the default workspace before the
    input checks, so a CPU logits tensor dies here with "device index out of
    range: -1").

Concurrent streams on one device each get their own slab, so launches that
overlap in time (distinct streams, graphs captured on different streams and
replayed together) never share mutable scratch; a caller-provided workspace
(run_ws() / `workspace=`) is optional, not required for concurrency. Two
graphs captured on the SAME stream share that stream's slab and are correct
as long as they replay in stream order. The number of slabs equals the number
of distinct stream handles that have run an eager slab-using launch:
`torch.cuda.Stream()` recycles handles from a fixed pool (32 per device and
priority), but externally created / foreign stream handles are not bounded by
that pool; `release_cached_resources()` frees them all.

Size: workspace_bytes() = GVR_WS_BUF_OFF + MAXC*GCAP*sizeof(int2)
    = 2048 + 160*16384*8 = 20,973,568 B.

Kernel-facing view: the compiled main-family signature takes the workspace
as a 1-D contiguous int32 tensor (fake tensor dtype Int32, assumed_align=16
-- torch caching-allocator bases are 256B-aligned so the default slab always
satisfies it).  `kernel_view()` reproduces raw `workspace.data_ptr()`
semantics for arbitrary user tensors by aliasing the underlying storage at
the tensor's byte offset.
"""


# workspace geometry constants -- must match the device kernels
GVR_MAX_DEV = 64
_MAXC = 160
_GCAP = 16384
_GVR_WS_BUF_OFF = 2048
WS_BYTES = _GVR_WS_BUF_OFF + _MAXC * _GCAP * 8  # 20,973,568
assert WS_BYTES == 20_973_568

_mu = threading.Lock()  # slow-path mutex
# FlashInfer-local: one slab per (device index, CUDA stream) instead of one
# per device. The SPLIT slab is mutable scratch (publish counters + candidate
# buffer, restored by the kernel), so two launches that can overlap in time —
# concurrent streams, or graphs captured on different streams and replayed
# together — must not share it (measured 17/5120 corrupted rows through a
# device-wide slab; see test_topk_varlen_serving). Keyed by the raw stream
# handle; the tensor refcount is the keep-alive.
_ws_keep = {}  # (device index, cuda stream handle) -> keep-alive int32 view


def _ws_key(d: int) -> tuple[int, int]:
    return (d, torch.cuda.current_stream(d).cuda_stream)


def workspace_bytes() -> int:
    """Workspace bytes required by the multi-CTA SPLIT path."""
    return WS_BYTES


def default_workspace(ref: torch.Tensor) -> torch.Tensor:
    """Cached workspace slab for (ref's device, the CURRENT stream).

    Returns the kernel-facing 1-D int32 view (zero-initialised on first use;
    the kernel restores the zeros it consumes, so one zeroing suffices for
    the lifetime of the cache entry). Allocation is eager-only: under
    CUDA-graph capture a slab would come from the graph's private pool and
    its zero-fill would be part of the graph, so a capture on a stream that
    has never run an eager gvr_2 call raises instead (same warm-up rule as
    the launcher cache)."""
    d = ref.get_device()
    if not (0 <= d < GVR_MAX_DEV):
        raise RuntimeError(f"device index out of range: {d}")
    key = _ws_key(d)
    ws = _ws_keep.get(key)  # hot path: one (GIL-atomic) load
    if ws is not None:
        return ws
    with _mu:  # slow path: double-checked
        ws = _ws_keep.get(key)
        if ws is not None:
            return ws
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "gvr_2 default workspace: no slab for this stream yet; run one "
                "eager gvr_2 call on the stream that captures the graph (or pass "
                "workspace={'gvr2_workspace': ...}) before CUDA-graph capture"
            )
        # lazy zeros via the torch caching allocator, viewed int32 for the
        # DSL launch signature.
        buf = torch.zeros(WS_BYTES, dtype=torch.uint8, device=ref.device)
        ws = buf.view(torch.int32)
        _ws_keep[key] = ws  # keep-alive
        return ws


def validate_run_ws(workspace: torch.Tensor, logits: torch.Tensor) -> None:
    """run_ws() workspace hardening, in a fixed predicate order:
    CUDA + same device as logits; numel*element_size >= workspace_bytes();
    base 16-byte aligned (the compiled ABI's assumed_align)."""
    if not (workspace.is_cuda and workspace.get_device() == logits.get_device()):
        raise RuntimeError("workspace must be a CUDA tensor on the same device")
    if workspace.numel() * workspace.element_size() < WS_BYTES:
        raise RuntimeError(f"workspace too small: need {WS_BYTES} bytes")
    if workspace.data_ptr() & 15:
        # the compiled ws_fake declares assumed_align=16 (FlashInfer-local
        # tightening from 8; an 8-but-not-16 pointer used to fail at launch)
        raise RuntimeError("workspace must be 16-byte aligned")


def kernel_view(workspace: torch.Tensor) -> torch.Tensor:
    """Raw-pointer view of a user workspace tensor: alias the first WS_BYTES
    bytes at the tensor's data_ptr() as int32[WS_BYTES/4], ignoring
    dtype/shape.

    NOTE: the DSL-side fake tensor declares assumed_align=16, which
    validate_run_ws enforces up front (an 8-but-not-16-byte pointer used to
    pass the host check and be rejected by the DSL at conversion)."""
    if (
        workspace.dtype is torch.int32
        and workspace.dim() == 1
        and workspace.is_contiguous()
        and workspace.storage_offset() == 0
        and workspace.numel() == WS_BYTES // 4
    ):
        return workspace  # already the canonical view
    off_bytes = workspace.storage_offset() * workspace.element_size()
    if off_bytes & 3:
        # unreachable past the 8B-alignment check for allocator-backed
        # storages; kept as a hard error rather than silent misalias.
        raise RuntimeError("workspace storage offset must be 4-byte aligned")
    t = torch.empty(0, dtype=torch.int32, device=workspace.device)
    t.set_(workspace.untyped_storage(), off_bytes // 4, (WS_BYTES // 4,))
    return t


def _reset_for_tests() -> None:
    """Drop cached slabs (tests only; NOT part of the C contract)."""
    with _mu:
        _ws_keep.clear()


def release_cached_resources(device=None) -> int:
    """FlashInfer-local: release the lazily created per-device caches — every
    default workspace slab (one per (device, stream), 20,973,568 B each) —
    for ``device`` (default: the current device; an int, a ``torch.device`` or
    a device string, CUDA only). Synchronizes the device, then drops the
    references under the cache lock; the memory returns to the torch caching
    allocator (``torch.cuda.empty_cache()`` hands it back to the driver).
    Returns the number of bytes released.

    CONTRACT — caller quiescence. The launch hot paths read the caches
    without taking the locks (a lock there would cost every launch), so this
    call cannot exclude a launch that is being issued concurrently: while it
    runs, no gvr_2 call on ``device`` may be in flight or issued — no eager
    call from any thread and no CUDA-graph replay — exactly like
    ``torch.cuda.empty_cache()`` versus live tensors. Under that contract the
    device sync retires every launch that could still address a cached
    object, and the locks make the clear atomic with respect to the slow
    paths that create objects (a creation either completes before the clear
    or starts after it; nothing is left half-published).

    INVALIDATION: a CUDA graph captured against a released slab replays on
    freed memory. Callers release only when no such graph will be replayed
    again, and re-capture after a fresh eager warm-up on the capturing stream
    (the same rule as the first capture)."""
    if device is None:
        d = torch.cuda.current_device()
    else:
        dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        if dev.type != "cuda":
            raise ValueError(
                f"release_cached_resources: expected a CUDA device, got {dev!r}"
            )
        d = dev.index if dev.index is not None else torch.cuda.current_device()
    freed = 0
    with _mu:
        torch.cuda.synchronize(d)
        for key in [k for k in _ws_keep if k[0] == d]:
            freed += _ws_keep.pop(key).numel() * 4
    return freed


# ===========================================================================
# ==== operator entry =======================================================
# ===========================================================================
"""Operator entry: input hardening, dispatch, and bind-once launch cache.

Hardening checks run in a fixed order with fixed predicates:
  1. all three tensors CUDA
  2. dtypes: logits f32, pre_idx i32, indices i32
  3. all 2-D
  4. all contiguous
  5. n_valid unwrap: python-int fast path (strict integral cast); Tensor
     path checks torch.cuda.is_current_stream_capturing() FIRST and fails
     loudly, else .item() (the D2H sync)
  6. b/npad from logits, k = pre_idx.size(1)
  7. b == 0 -> early no-op
  8. npad % 4 == 0 (float4 row loads)
  9. logits base 16-byte aligned
 10. pre_idx/indices batch dims match
 11. indices width >= k
 12. n_valid >= 0
 13. n = min(nv, npad) clamped in unbounded ints BEFORE any narrowing

Dispatch: route(b, n, npad, k) -> compile cache keyed on (kernel family,
constexpr tuple) in the device module -> bind-once launch cache keyed on the
shape key (b, n, npad, k): caches the compiled callable + the prebuilt
runtime-scalar arg pack as plain Python ints (never pre-wrapped
cutlass.Int32 -- the FFI per-argument cost is paid every call regardless;
pre-binding removes only route()/marshal-prep work).

Error contract: launch failures surface as exceptions WITH
(b, n, npad, k) context.

The device module is imported LAZILY (first shape that routes to it), so a
missing/broken module only fails when actually reached, with (b, n, npad, k)
context.  The per-family compiled ABIs are documented at each launcher
builder in _build_launcher; only the main family takes the workspace.
"""


# shape key (b, n, npad, k) -> (fn, args tuple of python ints, needs_ws)
_LAUNCH_CACHE = {}
_DUMMY_KV = {}


def _dummy_kv(dev_index, device):
    """Cached 1-element int32 tensor per device — the dead kv_lens slot of
    the extended gvr_main ABI in legacy (batch-uniform) mode."""
    t = _DUMMY_KV.get(dev_index)
    if t is None:
        t = torch.zeros(1, dtype=_I32, device=device)
        _DUMMY_KV[dev_index] = t
    return t


# hot-path local bindings: each torch.<attr> lookup costs ~0.1 us and the
# validation battery runs on EVERY call
_F32 = torch.float32
_I32 = torch.int32
_TENSOR = torch.Tensor
_is_capturing = torch.cuda.is_current_stream_capturing
_index = operator.index
_ws_hot = _ws_keep  # shared dict object (hot-path load)
_GVR_MAX_DEV = GVR_MAX_DEV


# ---------------------------------------------------------------------------
# per-family launcher builders (cold path: once per distinct shape key)
# ---------------------------------------------------------------------------
def _build_launcher(b, n, npad, k):
    rd = route(b, n, npad, k, sms=_sm_count())
    fam = rd["kernel"]
    tpl = tuple(rd["tpl"])
    rt = rd["rt"]
    if fam in ("reg", "regimg"):
        dev = _device()
        raw = _gate_first_call(dev.get_compiled__reg(tpl))

        # compiled ABI: (logits, pre_idx, kv_lens, out, n, CMP, QC,
        # smem_total) -- kv_lens is the dead varlen slot in batch-uniform
        # mode (cached dummy tensor)
        def fn(lg, pi, o, *a, _raw=raw):
            _raw(lg, pi, _dummy_kv(lg.get_device(), lg.device), o, *a)

        # + 4*k: the register kernel stages its k outputs in smem (stage_out default)
        args = (rt["n"], rt["CMP"], rt["QC"], dev.STATIC_BYTES + rd["smem"] + 4 * k)
        return (fn, args, False)
    if fam == "main":
        dev = _device()
        raw = _gate_first_call(dev.get_compiled(tpl))

        # compiled ABI: (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_,
        #                R, SMP, TGT, Q, SS2, TGT2,
        #                kv_lens, aim_base, sfac, amin, sd_en, tsh_en)
        # [SCAP_/CMP_ dead, ABI parity; the trailing varlen block is dead in
        #  legacy mode — a cached dummy kv_lens tensor + five zeros]
        def fn(lg, pi, o, w, *a, _raw=raw):
            _raw(lg, pi, o, w, *a, _dummy_kv(lg.get_device(), lg.device), 0, 0, 0, 0, 0)

        args = (
            rt["n"],
            rt["npad"],
            rt["k"],
            rt["SCAP_"],
            rt["CMP_"],
            rt["R"],
            rt["SMP"],
            rt["TGT"],
            rt["Q"],
            rt["SS2"],
            rt["TGT2"],
        )
        return (fn, args, True)
    if fam == "clus":
        dev = _device()
        # compile key carries the smem-extent scalars (scap/cmp_); compiled
        # ABI: (logits, pre_idx, kv_lens, out, n, npad, k, SCAP, CMP, SMP,
        #       TGT, Q, SS2, TGT2) -- NO workspace; kv_lens is the dead
        # varlen slot in batch-uniform mode (cached dummy tensor)
        fn = _gate_first_call(
            dev.get_compiled__clus(tpl, scap=rt["SCAP"], cmp_=rt["CMP"])
        )
        args = (
            rt["n"],
            rt["npad"],
            rt["k"],
            rt["SCAP"],
            rt["CMP"],
            rt["SMP"],
            rt["TGT"],
            rt["Q"],
            rt["SS2"],
            rt["TGT2"],
        )

        def _call(lg, pi, idx, _fn=fn, _args=args):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, *_args)

        return (_call, (), False)
    if fam == "reg_clus":
        dev = _device()
        # compiled ABI: (logits, pre_idx, kv_lens, out, n) -- kv_lens is the
        # dead varlen slot in batch-uniform mode (cached dummy tensor);
        # smem/k derived in-module
        fn = _gate_first_call(dev.get_compiled__regclus(tpl))
        n_arg = rt["n"]

        def _call(lg, pi, idx, _fn=fn, _n=n_arg):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, _n)

        return (_call, (), False)
    # unreachable: route() only emits the five families above
    raise RuntimeError(f"unknown dispatch family {fam!r}")


# ---------------------------------------------------------------------------
# shared implementation of the batch-uniform entries
# ---------------------------------------------------------------------------
def _run_impl(logits, pre_idx, n_valid, indices, ws, values=None):
    if not (logits.is_cuda and pre_idx.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if logits.dtype is not _F32:
        raise RuntimeError("logits must be float32")
    if pre_idx.dtype is not _I32:
        raise RuntimeError("pre_idx must be int32")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    lsh, psh, ish = logits.shape, pre_idx.shape, indices.shape
    if not (len(lsh) == 2 and len(psh) == 2 and len(ish) == 2):
        raise RuntimeError("logits/pre_idx/indices must be 2-D")
    if not (logits.is_contiguous() and pre_idx.is_contiguous() and indices.is_contiguous()):
        raise RuntimeError("tensors must be contiguous")

    # n_valid unwrap: tensor path = D2H sync, illegal under CUDA graph
    # capture -- fail loudly instead of crashing the capture.
    if isinstance(n_valid, _TENSOR):
        if _is_capturing():
            raise RuntimeError(
                "tensor n_valid requires a D2H sync, illegal under CUDA "
                "graph capture — pass n_valid as a python int"
            )
        nv = int(n_valid.item())
    else:
        # strict integral cast (rejects floats/strings)
        nv = _index(n_valid)

    b, npad = lsh
    k = psh[1]
    if b == 0:  # empty batch: no-op
        return
    if npad & 3:
        raise RuntimeError(f"npad (logits stride) must be a multiple of 4, got {npad}")
    if logits.data_ptr() & 15:
        raise RuntimeError(
            "logits base must be 16-byte aligned (storage-offset views break the float4 row loads)"
        )
    if psh[0] != b or ish[0] != b:
        raise RuntimeError(f"batch dims must match: logits {b} pre_idx {psh[0]} indices {ish[0]}")
    if ish[1] < k:
        raise RuntimeError(f"indices width {ish[1]} < k={k} (k is pre_idx.size(1))")
    if nv < 0:
        raise RuntimeError(f"n_valid must be non-negative, got {nv}")
    # clamp BEFORE any narrowing (python ints are unbounded, so min() is the
    # exact 64-bit clamp)
    n = nv if nv < npad else npad

    # CUDA out-indexing mirror: every kernel derives O = out + row*k --
    # flat PACKED rows, ignoring the actual indices width.  The DSL kernels
    # index out[row, :] with the tensor's own row stride, so a wider
    # `indices` must be re-viewed packed (pure view, no copy; contiguity
    # already checked).
    if ish[1] != k:
        indices = indices.reshape(-1)[: b * k].view(b, k)

    # ---- optional values output (production parity, default OFF) ------------
    # dsa.py allocates the values scratch only for the non-CuTeDSL path, so
    # values stay opt-in. The indices are exact, so a gather epilogue
    # reproduces the in-kernel writeback bit-for-bit; the constexpr in-kernel
    # form rides the CUDA-graph per-row rewrite.
    if values is not None:
        if not values.is_cuda:
            raise RuntimeError("values must be CUDA")
        if values.dtype is not _F32:
            raise RuntimeError("values must be float32")
        vsh = values.shape
        if len(vsh) != 2 or not values.is_contiguous():
            raise RuntimeError("values must be 2-D contiguous")
        if vsh[0] != b:
            raise RuntimeError(f"batch dims must match: logits {b} values {vsh[0]}")
        if vsh[1] < k:
            raise RuntimeError(f"values width {vsh[1]} < k={k}")
        if vsh[1] != k:
            values = values.reshape(-1)[: b * k].view(b, k)

    # ---- n <= k short path (heuristicTopKDecode.cu parity) ------------------
    # Every valid position is in the top-K: emit identity indices and pad the
    # tail with -1 (the production pad convention; downstream treats -1 as
    # invalid). Order is contract-irrelevant — exactness is tie-interchangeable
    # SET semantics. Torch-op path for now; the CUDA-graph-safe per-row rewrite
    # moves this branch in-kernel (it cannot fall back per row inside a graph).
    if n <= k:
        if n > 0:
            indices[:, :n] = torch.arange(n, dtype=_I32, device=indices.device)
            if values is not None:
                values[:, :n] = logits[:, :n]
        if n < k:
            indices[:, n:] = -1
            if values is not None:
                values[:, n:] = torch.finfo(_F32).min  # -FLT_MAX pad
        return

    key = (b, n, npad, k, _arch_token(), _sm_count())
    lc = _LAUNCH_CACHE.get(key)
    if lc is None:
        lc = _build_launcher(b, n, npad, k)
        _LAUNCH_CACHE[key] = lc
    fn, args, needs_ws = lc
    try:
        if needs_ws:
            fn(logits, pre_idx, indices, ws, *args)
        else:
            fn(logits, pre_idx, indices, *args)
    except Exception as e:
        raise RuntimeError(f"gvr_topk launch failed (b={b} n={n} npad={npad} k={k}): {e}") from e
    if values is not None:
        # same epilogue as run_varlen: a (never-expected) negative index
        # degrades to -FLT_MAX instead of a context-poisoning device assert
        idx64 = indices.to(torch.int64)
        values.copy_(logits.gather(1, idx64.clamp_min(0)))
        values.masked_fill_(indices < 0, torch.finfo(_F32).min)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------
def run(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    n_valid: int,
    indices: torch.Tensor,
    values: torch.Tensor | None = None,
) -> None:
    """TESTING/BENCH ONLY — production callers must use ``run_varlen`` (per-request
    device kv_lens; this entry assumes one batch-uniform host ``n_valid``,
    which real serving batches do not satisfy).

    Fast 4-arg form.  ``values`` (optional DPS output, default None = OFF)
    mirrors the production values writeback; see _run_impl.
    The default per-device slab workspace is resolved FIRST (a CPU logits
    tensor therefore dies with 'device index out of range').
    Hot path inlines the device check + atomic load + cache hit; the slow
    path allocates under the workspace lock."""
    if _on_other_device(logits):
        with torch.cuda.device(logits.get_device()):
            return run(logits, pre_idx, n_valid, indices, values)
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:  # checked on EVERY call
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_hot.get(_ws_key(d))
    if ws is None:
        ws = default_workspace(logits)
    _run_impl(logits, pre_idx, n_valid, indices, ws, values)


def run_ws(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    n_valid: int,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    values: torch.Tensor | None = None,
) -> None:
    """TESTING/BENCH ONLY — production callers must use ``run_varlen(workspace=...)``.

    Explicit-workspace form for multi-stream callers."""
    if _on_other_device(logits):
        with torch.cuda.device(logits.get_device()):
            return run_ws(logits, pre_idx, n_valid, indices, workspace, values)
    validate_run_ws(workspace, logits)
    _run_impl(logits, pre_idx, n_valid, indices, kernel_view(workspace), values)


def run_varlen(
    logits: torch.Tensor,
    pre_idx: torch.Tensor | None,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    values: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    engine: str = "auto",
    workspace: torch.Tensor | None = None,
    top_k: int | None = None,
    row_starts: torch.Tensor | None = None,
    absolute_indices: bool = False,
    page_table: torch.Tensor | None = None,
    page_size: int = 1,
) -> None:
    """Production-contract varlen entry (per-row device kv_lens).

    HINT-FREE: ``pre_idx=None`` runs the hint-free compiled engines (identity
    sample in place of the hint, no hint loads; TRT-LLM #18410) and then
    REQUIRES ``top_k`` (normally ``k = pre_idx.shape[1]``). Exact like the
    hinted call; only the sampling anchor is weaker. When both are given they
    must agree. Hint mode is a warm-up dimension: hinted and hint-free
    launchers are distinct compiled objects (``warmup_varlen`` compiles both).

    WINDOWED / PREFILL (TRT-LLM #18702 port): ``row_starts`` int32 ``[num_rows]``
    makes row ``r``'s candidates ``logits[r, ks : ks + kv_lens[r])`` with
    ``ks = row_starts[r]`` (compressed column units) and the output indices
    LOCAL to the window (``column - ks``; the absolute column with
    ``absolute_indices=True``), ``-1`` padded, identity for short
    windows — the DSA prefill indexer contract (packed requests along the
    column axis, causal per-row windows). Hint-free only, ``next_n == 1`` and
    ``compress_ratio == 1``; columns outside a window are never read. Runs the
    dedicated prefill engines: a register rung when ``max_seq_len`` (or the
    logits width) fits one, else the streaming ``main`` family (one CTA per
    row); ``warmup_prefill`` compiles them ahead of CUDA-graph capture.

    Row semantics (mirror of ``heuristicTopKDecode.cu`` and the in-tree
    ``cute_dsl_gvr_topk_decode`` runner):

      ``num_rows = logits.shape[0]``, ``batch = num_rows // next_n``;
      ``kv_lens`` int32 ``[batch]`` — per-request TOTAL cache length in
      UNCOMPRESSED token space (dsa.py ``metadata.kv_lens_cuda_runtime``,
      not new-token seq_lens); row ``r`` uses
      ``n_r = (kv_lens[r // next_n] - next_n + (r % next_n) + 1) //
      compress_ratio`` valid entries (cr 1 = DSv3.2, 4 = DSv4 Flash/Pro);
      ``pre_idx`` ``[batch, k]`` is REQUEST-level raw prev-step top-K,
      shared by a request's ``next_n`` rows (offset-free hint contract);
      per-row ``n_r <= k`` takes the short path (identity + ``-1`` tail).

    ENGINES: ``engine="auto"`` (default) launches the per-row IN-KERNEL
    gvr_main varlen port — ONE launch for the whole batch; each CTA reads its
    row's kv_len on device and re-derives the sampling ladder (route_dynamic
    formula mirror), so with ``max_seq_len`` given (a capture-stable engine
    constant, e.g. dsa.py's ``indexer_max_seq_len``) the call performs NO
    host reads.  Without ``max_seq_len`` the envelope comes from ONE
    ``kv_lens.max()`` host read (documented sync, refused under capture).
    With ``row_starts`` (windowed mode) the engine is chosen by a bound on the
    WINDOW lengths: ``max_seq_len`` when given, else the logits width (always
    a valid bound, no host read). A register rung whose capacity covers the
    bound serves the launch, else the streaming slab. Every engine clamps
    windows to the logits width; the slab ranks any window exactly, while a
    register rung reports a row whose extent exceeds its capacity as all -1
    (never a truncated ranking), so a caller-given bound must hold at every
    graph replay; the width-derived default cannot be exceeded.
    ``engine="reference"`` keeps the b=1 host-loop reference implementation —
    the differential oracle the in-kernel engine is validated against.

    KNOWN LIMITATION: on rows containing NaN logits the selected index SET
    can differ from ``heuristicTopKDecode.cu`` (both kernels order NaNs
    implementation-specifically). Finite inputs — including +/-inf and
    denormals — are tie-aware exact.

    CONTRACT: correct and dispatched for any ``num_rows``
    (BS 1..1024+ x next_n) and any envelope up to 1M kv tokens.  Family
    selection (streaming main / clustered register-resident) is a pure
    function of the capture-stable launcher key.
    """
    if _on_other_device(logits):
        with torch.cuda.device(logits.get_device()):
            return run_varlen(
                logits,
                pre_idx,
                kv_lens,
                indices,
                next_n=next_n,
                compress_ratio=compress_ratio,
                values=values,
                max_seq_len=max_seq_len,
                engine=engine,
                workspace=workspace,
                top_k=top_k,
                row_starts=row_starts,
                absolute_indices=absolute_indices,
                page_table=page_table,
                page_size=page_size,
            )
    if logits.dtype is not torch.float32:
        raise RuntimeError(
            f"logits must be float32 (got {logits.dtype}); bf16/fp16 paths "
            "are a follow-up — see the PR roadmap"
        )
    if not (isinstance(kv_lens, _TENSOR) and kv_lens.is_cuda):
        raise RuntimeError("kv_lens must be a CUDA tensor")
    if kv_lens.dtype is not _I32:
        raise RuntimeError("kv_lens must be int32")
    if kv_lens.dim() != 1:
        raise RuntimeError("kv_lens must be 1-D")
    nn = _index(next_n)
    cr = _index(compress_ratio)
    if nn < 1:
        raise RuntimeError(f"next_n must be >= 1, got {nn}")
    if cr not in (1, 4):
        raise RuntimeError(f"compress_ratio must be 1 (DSv3.2) or 4 (DSv4), got {cr}")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    # Hint-free (pre_idx=None): the in-kernel engines are compiled with the
    # identity sample (row positions 0..k-1, computed in registers) in place
    # of the hint loads (upstream PR NVIDIA/TensorRT-LLM#18410, refined per
    # family in the device module), so the pre_idx ABI slot is never read and
    # ``k`` comes from ``top_k``. The reference engine below builds the same
    # identity hint as a tensor.
    hint_free = pre_idx is None
    if hint_free:
        if top_k is None:
            raise RuntimeError("run_varlen: top_k is required when pre_idx is None (hint-free)")
        k = _index(top_k)
        if k < 1:
            raise RuntimeError(f"top_k must be >= 1, got {k}")
    else:
        if top_k is not None and len(pre_idx.shape) == 2 and pre_idx.shape[1] != _index(top_k):
            raise RuntimeError(f"top_k={top_k} != pre_idx.shape[1]={pre_idx.shape[1]}")
        if len(pre_idx.shape) != 2 or pre_idx.shape[0] != batch:
            raise RuntimeError(
                f"pre_idx must be [batch={batch}, k] REQUEST-level, got {tuple(pre_idx.shape)}"
            )
        k = pre_idx.shape[1]
    windowed = row_starts is not None
    if windowed:
        # prefill scope (upstream #18702): hint-free, one row per request, no
        # compression shift — kv_lens IS the window length in column units
        if not hint_free:
            raise RuntimeError("row_starts (windowed / prefill mode) requires pre_idx=None")
        if k < 4:
            # the window engines mask the <= 3 lead lanes of the float4-aligned
            # read base in the vector path; the scalar tail lane is not lane-
            # masked, which only matters when the whole scan extent is shorter
            # than one float4 (nv + lead < 4 with nv > k, i.e. k <= 2). The
            # public API admits k in {512, 1024, 2048}; keep direct callers out.
            raise RuntimeError(
                f"row_starts (windowed / prefill mode) requires top_k >= 4, got {k}"
            )
        if nn != 1 or cr != 1:
            raise RuntimeError(
                f"row_starts requires next_n == 1 and compress_ratio == 1, got {nn} / {cr}"
            )
        if not (isinstance(row_starts, _TENSOR) and row_starts.is_cuda):
            raise RuntimeError("row_starts must be a CUDA tensor")
        if row_starts.dtype is not _I32:
            raise RuntimeError("row_starts must be int32")
        if row_starts.dim() != 1 or row_starts.shape[0] != num_rows:
            raise RuntimeError(
                f"row_starts must be 1-D of length num_rows={num_rows}, "
                f"got {tuple(row_starts.shape)}"
            )
        if not row_starts.is_contiguous():
            raise RuntimeError("row_starts must be contiguous")
    elif absolute_indices:
        raise RuntimeError("absolute_indices requires row_starts (windowed / prefill mode)")
    paged = page_table is not None
    page_shift = None
    if paged:
        # PAGED output (decode): every selected column c of request q leaves the
        # kernel as page_table[q, c // page_size] * page_size + c % page_size, the
        # physical KV slot the sparse-attention kernel reads (the SGLang
        # DSA decode contract). Hint-free only (the page table rides in the
        # pre_idx ABI slot), no windows, no values (the raw indices are gone).
        if windowed:
            raise RuntimeError("page_table is a decode-mode option (no row_starts)")
        if not hint_free:
            raise RuntimeError("page_table requires pre_idx=None (hint-free engines)")
        if values is not None:
            raise RuntimeError("page_table cannot be combined with values (raw indices are not kept)")
        ps = _index(page_size)
        if ps < 1 or ps > (1 << 30) or ps & (ps - 1):
            raise RuntimeError(f"page_size must be a power of two in [1, 2**30], got {page_size}")
        page_shift = ps.bit_length() - 1
        if not (isinstance(page_table, _TENSOR) and page_table.is_cuda):
            raise RuntimeError("page_table must be a CUDA tensor")
        if page_table.dtype is not _I32:
            raise RuntimeError("page_table must be int32")
        if page_table.dim() != 2 or page_table.shape[0] != batch:
            raise RuntimeError(
                f"page_table must be [batch={batch}, max_pages] (one row per request), "
                f"got {tuple(page_table.shape)}"
            )
        if not page_table.is_contiguous():
            raise RuntimeError("page_table must be contiguous")
        # every column the kernels can emit (< min(envelope, width)) needs a page
        need_pages = (min(_index(max_seq_len) // cr if max_seq_len is not None else logits.shape[1], logits.shape[1]) + ps - 1) // ps
        if page_table.shape[1] < need_pages:
            raise RuntimeError(
                f"page_table has {page_table.shape[1]} pages per request; the logits width "
                f"needs {need_pages} at page_size={ps}"
            )
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    if workspace is not None:
        # multi-stream escape hatch (run_ws parity): concurrent varlen
        # launches on one device must not share the SPLIT publish slab
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        # resolved lazily: only the streaming `main` family takes the slab, so
        # register/cluster plans never need one (and can be captured on any
        # stream without an eager launch there)
        ws = None

    if engine == "auto":
        # ---- per-row in-kernel engine (gvr_main varlen port) ----------------
        # Full validation battery (the engine bypasses _run_impl — every
        # check the batch-uniform path enforces is replayed here; the
        # batch-dim check is CRITICAL: the kernel grid comes from
        # logits.shape[0], so a short indices/values tensor would be written
        # out of bounds).
        if not (logits.is_cuda and indices.is_cuda and (hint_free or pre_idx.is_cuda)):
            raise RuntimeError("all tensors must be CUDA")
        if (
            logits.dtype is not _F32
            or indices.dtype is not _I32
            or (not hint_free and pre_idx.dtype is not _I32)
        ):
            raise RuntimeError("logits must be float32; pre_idx/indices int32")
        if len(indices.shape) != 2 or indices.shape[0] != num_rows:
            raise RuntimeError(
                f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
            )
        if indices.shape[1] < k:
            raise RuntimeError(f"indices width {indices.shape[1]} < k={k}")
        if not (indices.is_contiguous() and kv_lens.is_contiguous()):
            raise RuntimeError("indices/kv_lens must be contiguous")
        if not hint_free and not pre_idx.is_contiguous():
            raise RuntimeError("pre_idx must be contiguous")
        if (indices.data_ptr() | (0 if hint_free else pre_idx.data_ptr())) & 15:
            # the compiled fakes declare assumed_align=16 for pre_idx/indices
            # (FlashInfer-local check; the DSL used to refuse at conversion)
            raise RuntimeError(
                "pre_idx/indices base must be 16-byte aligned (shifted views are "
                "not supported)"
            )
        # logits: accept row-major views with a wider row stride (the DSL
        # paged-MQA logits arena is 256-aligned and column-sliced — a legal
        # NON-contiguous view). The kernel only needs (base, row stride):
        # widen back to a compact [rows, stride] view over the same storage;
        # the tail columns are never classified (per-row n gates all reads).
        if logits.stride(1) != 1:
            raise RuntimeError("logits inner stride must be 1")
        if num_rows > 1 and logits.stride(0) < logits.shape[1]:
            # the row pitch is derived from stride(0); overlapping rows (an
            # expand view, stride 0) would shrink the search domain and drive
            # reads past the storage (FlashInfer-local check)
            raise RuntimeError(
                "logits rows overlap (stride(0) < shape[1]); pass a row-major "
                "view with stride(0) >= shape[1]"
            )
        npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
        lg = logits
        if not logits.is_contiguous():
            need = logits.storage_offset() + num_rows * npad
            if logits.untyped_storage().size() // 4 < need:
                raise RuntimeError("logits view storage too small to widen to its row stride")
            lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
        if npad & 3:
            raise RuntimeError(f"npad (logits row stride) must be a multiple of 4, got {npad}")
        if lg.data_ptr() & 15:
            raise RuntimeError("logits base must be 16-byte aligned")
        if values is not None:
            if not values.is_cuda or values.dtype is not _F32:
                raise RuntimeError("values must be CUDA float32")
            if (
                len(values.shape) != 2
                or values.shape[0] != num_rows
                or values.shape[1] < k
                or not values.is_contiguous()
            ):
                raise RuntimeError(
                    f"values must be contiguous [num_rows={num_rows}, >=k], "
                    f"got {tuple(values.shape)}"
                )
        cshift = 0 if cr == 1 else 2
        if windowed and max_seq_len is None:
            # windowed call without a bound: the logits width is the bound
            # (always valid: no window is wider than its row; no host read of
            # device data) and routes the engine exactly like max_seq_len
            n_env = min(logits.shape[1], npad)
        elif max_seq_len is not None:
            n_env = int(max_seq_len) >> cshift
        else:
            if _is_capturing():
                raise RuntimeError(
                    "run_varlen without max_seq_len reads kv_lens.max() on "
                    "host — pass max_seq_len (a capture-stable engine "
                    "constant) under CUDA graph capture"
                )
            n_env = int(kv_lens.max().item()) >> cshift
            # eager mode: quantize the data-dependent envelope up to the next
            # power of two so a growing decode does not recompile at every
            # R increment (bounded plans, bounded _VARLEN_CACHE)
            n_env = 1 << max(n_env - 1, 1).bit_length()
        n_env = min(max(n_env, 1), npad)
        if windowed:
            idx = indices
            if idx.shape[1] != k:
                idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
            vals = values
            if vals is not None and vals.shape[1] != k:
                vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
            # ws stays None here: _launch_prefill resolves the default slab only
            # for a slab (streaming main) launch, so a register-rung capture
            # needs no eager call on the capturing stream (prefill_ready
            # mirrors the same rule). Every engine clamps windows to the logits
            # WIDTH (memory safety); n_env (max_seq_len, else the width) picks
            # the engine (register rung or slab tier / envelope bucket); a
            # register rung reports rows beyond its capacity as all -1
            _launch_prefill(
                lg,
                row_starts,
                kv_lens,
                idx,
                ws,
                k,
                min(logits.shape[1], npad),
                n_env,
                npad,
                abs_out=bool(absolute_indices),
            )
            if vals is not None:
                # absolute columns for the gather (window-local + row start
                # unless the kernel already emitted absolute columns)
                idx64 = idx.to(torch.int64)
                if not absolute_indices:
                    # the kernels clamp a negative start to 0 (the window keeps
                    # its length), so the gather adds the clamped start as well
                    idx64 = idx64 + row_starts.to(torch.int64).clamp_(min=0).unsqueeze(1)
                vals.copy_(lg.gather(1, idx64.clamp_min(0).clamp_max(npad - 1)))
                vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
            return
        key = (num_rows, npad, k, n_env, nn, cr, hint_free, page_shift, _arch_token(), _sm_count())
        lc = _VARLEN_CACHE.get(key)
        if lc is None:
            if _is_capturing():
                raise RuntimeError(
                    "varlen launcher not compiled for this shape — warm up "
                    "before CUDA graph capture"
                )
            lc = _varlen_launcher(num_rows, npad, k, n_env, nn, cr, hint_free, page_shift)
        idx = indices
        if idx.shape[1] != k:
            idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
        vals = values
        if vals is not None and vals.shape[1] != k:
            vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
        # hint-free engines never read the pre_idx ABI slot: pass the output
        # buffer (int32, 16-byte aligned) to satisfy the compiled fake; the
        # paged engines read the page table from that slot instead
        pre_arg = page_table if paged else (idx if hint_free else pre_idx)
        if lc[0] == "reg_clus":
            # compiled ABI: (logits, pre_idx, kv_lens, out, n_envelope)
            lc[1](lg, pre_arg, kv_lens, idx, lc[2])
        elif lc[0] == "reg":
            # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, CMP, QC, smem)
            lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
        elif lc[0] == "clus":
            # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, npad, k,
            #                SCAP, CMP, dead DYN x5)
            lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
        else:
            _, fn, pre, tail = lc
            if ws is None:
                ws = _ws_hot.get(_ws_key(d))
                if ws is None:
                    ws = default_workspace(logits)  # raises under capture
            fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
        if vals is not None:
            idx64 = idx.to(torch.int64)
            vals.copy_(lg.gather(1, idx64.clamp_min(0)))
            vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
        return
    if engine != "reference":
        raise RuntimeError(f"engine must be 'auto' or 'reference', got {engine!r}")

    # ---- reference engine (differential oracle): b=1 host loop --------------
    if _is_capturing():
        raise RuntimeError(
            "run_varlen reference engine reads kv_lens on host, illegal under CUDA graph capture"
        )
    # match the engine's flat-packed output convention for wider-than-k
    # buffers (pack ONCE from the tensor base, then slice per row)
    if hint_free:
        # the batch-uniform kernels read a hint: the identity sample, the same
        # positions the hint-free engines compute in registers
        pre_idx = (
            torch.arange(k, dtype=_I32, device=logits.device)
            .unsqueeze(0)
            .expand(batch, k)
            .contiguous()
        )
    idx = indices
    if len(idx.shape) != 2 or idx.shape[0] != num_rows:
        raise RuntimeError(
            f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
        )
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    vals = values
    if vals is not None and vals.shape[1] != k:
        vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
    kl = kv_lens.tolist()  # the ONE documented D2H sync of this engine
    ksl = row_starts.tolist() if windowed else None
    if ws is None:
        ws = default_workspace(logits)
    for r in range(num_rows):
        # production graph slots can carry kv_len < next_n (padded / evicted
        # requests): clamp to the empty row, emitting all -1 — the same
        # contract the in-kernel engine implements
        actual = max(kl[r // nn] - nn + (r % nn) + 1, 0)
        req = r // nn
        row_logits = logits[r : r + 1]
        if windowed:
            # window [ks, ks + len) clamped to the row: copy it to a fresh
            # 16-byte-aligned buffer (ks need not be a multiple of 4) so the
            # batch-uniform kernel sees a plain row; indices come out local
            ks = max(min(ksl[r], logits.shape[1]), 0)
            actual = max(min(actual, logits.shape[1] - ks), 0)
            wlen = max((actual + 3) // 4 * 4, 4)
            row_logits = torch.full(
                (1, wlen), float("-inf"), dtype=_F32, device=logits.device
            )
            row_logits[0, :actual] = logits[r, ks : ks + actual]
        _run_impl(
            row_logits,
            pre_idx[req : req + 1],
            actual // cr,
            idx[r : r + 1],
            ws,
            None if vals is None else vals[r : r + 1],
        )
        if windowed and absolute_indices:
            # reference frame is window-local: shift the hits (not the pad)
            rr = idx[r]
            idx[r] = torch.where(rr >= 0, rr + ks, rr)
        if paged:
            rr = idx[r].to(torch.int64)
            pt = page_table[req].to(torch.int64)
            phys = (pt[(rr >> page_shift).clamp_min(0)] << page_shift) | (rr & ((1 << page_shift) - 1))
            idx[r] = torch.where(rr >= 0, phys, rr).to(_I32)


__all__ = [
    "route",
    "route_static",
    "route_dynamic",
    "route_split",
    "route_bands",
    "run",
    "run_ws",
    "run_varlen",
    "warmup_varlen",
    "workspace_bytes",
    "WS_BYTES",
    "default_workspace",
    "validate_run_ws",
    "kernel_view",
]


# --------------------------------------------------------------------------
# warmup: pre-compile the varlen engine for an engine envelope so no live
# request pays the first-touch DSL JIT (mirrors warmup_heuristic_topk_decode
# and warmup_cute_dsl_radix_topk). Idempotent per (device, geometry) key.
# CUDA-graph capture warmup naturally compiles the captured batch sizes;
# this covers the eager/first-touch path (num_rows defaults to (1,)).
_VARLEN_WARMUP_DONE: set = set()
_VARLEN_WARMUP_LOCK = threading.Lock()


def warmup_varlen(
    top_k: int,
    max_seq_len: int,
    compress_ratio: int = 1,
    next_n: int = 1,
    num_rows_list: Sequence[int] = (1,),
    row_stride: int | None = None,
    page_size: int | None = None,
) -> None:
    """TESTING/INIT ONLY — compile the varlen engine's envelope tuples.

    One tiny real launch per requested ``num_rows`` and hint mode (hinted and
    hint-free engines are distinct compiled objects; compile keys do not
    depend on tensor contents). Uses the current CUDA device. The done-key
    is recorded only after every launch succeeds, so a failed or interrupted
    warmup is retried on the next call instead of short-circuiting to an
    uncompiled engine.

    ``row_stride`` must be the logits row stride the serving producer will
    emit: the launcher key includes it, so a warmup at a different stride
    compiles a variant dispatch never looks up. Callers that know the
    producer layout (e.g. the DSL paged-MQA arena's 256-element rounding)
    must pass it; the 64-element default only matches producers that round
    the same way.

    ``page_size`` additionally compiles the PAGED hint-free engines and
    launchers for that page size (``run_varlen(page_table=..., page_size=)``
    is a distinct compiled set per page size).
    """
    page_shift = None
    if page_size is not None:
        ps = int(page_size)
        if ps < 1 or ps > (1 << 30) or ps & (ps - 1):
            raise RuntimeError(f"page_size must be a power of two in [1, 2**30], got {page_size}")
        page_shift = ps.bit_length() - 1
    dev = torch.cuda.current_device()
    nn = max(1, int(next_n))
    # round each request down to a next_n multiple (min next_n) and dedup
    req_rows = sorted({max(int(r) - int(r) % nn, nn) for r in num_rows_list})
    if not req_rows:
        return
    # BAND-AWARE enumeration: the engine compile key depends on the plan's
    # constexpr tuple (+ r_const family axis) and, for the register families,
    # on the staged-output mode (a row-count threshold: _reg_stage_out /
    # _regclus_stage_out), NOT on the exact row count, so warming ONE
    # representative row per distinct engine key covers every row count up to
    # the largest request. Representatives are the first row of
    # each band, which keeps the warmup allocation bounded (~a few hundred
    # rows) even when CUDA-graph batch lists reach thousands of rows.
    n_env_c = max(1, int(max_seq_len) // int(compress_ratio))
    npad_c = (n_env_c + 63) // 64 * 64 if row_stride is None else int(row_stride)
    # create THIS stream's default workspace slab (the band launches below may
    # all land on register/cluster plans while a requested row count routes
    # to the slab-using `main` family; refused under capture)
    default_workspace(torch.empty(0, dtype=torch.uint8, device=torch.device("cuda", dev)))
    seen_keys = set()
    rows_list = []
    r = nn
    r_max = req_rows[-1]
    sms = _sm_count()
    while r <= r_max:
        plan_free = route(
            r, max(min(n_env_c, npad_c), int(top_k) + 1), npad_c, int(top_k), sms=sms
        )
        if plan_free["kernel"] == "reg_clus":
            ekey = (
                "reg_clus",
                tuple(plan_free["tpl"]),
                _regclus_stage_out(r, int(top_k), page_shift is not None),
            )
        elif plan_free["kernel"] in ("reg", "regimg"):
            ekey = ("reg", tuple(plan_free["tpl"]), _reg_stage_out(r, sms))
        else:
            p = route_streaming(
                r,
                max(min(n_env_c, npad_c), int(top_k) + 1),
                npad_c,
                int(top_k),
                force_main=True,
            )
            ekey = ("main", tuple(p["tpl"][:6]), p["rt"]["R"])
        if ekey not in seen_keys:
            seen_keys.add(ekey)
            rows_list.append(r)
        r += nn
    if not rows_list:
        return
    n_env = max(1, int(max_seq_len) // int(compress_ratio))
    if row_stride is None:
        npad = (n_env + 63) // 64 * 64
    else:
        npad = int(row_stride)
        if npad < n_env or npad % 4:
            raise RuntimeError(
                f"row_stride must be a float4-multiple >= n_env={n_env}, got {row_stride}"
            )
    key = (
        dev,
        int(top_k),
        int(max_seq_len),
        int(compress_ratio),
        nn,
        tuple(rows_list),
        npad,
        page_shift,
    )
    n_env_l = min(max(int(max_seq_len) >> (0 if int(compress_ratio) == 1 else 2), 1), npad)
    with _VARLEN_WARMUP_LOCK:
        bands_done = key in _VARLEN_WARMUP_DONE
    if bands_done:
        # FlashInfer-local (DKG #60): the done key covers the ENGINE band
        # launches only (they depend on the representative rows, not on the
        # exact request), so the pure-host exact-row launcher population must
        # still run for a later warmup that adds a row count mapping to an
        # already-warmed engine — otherwise a CUDA-graph capture at that row
        # count misses the launcher cache.
        for r in req_rows:
            for hf in (False, True):
                _varlen_launcher(r, npad, int(top_k), n_env_l, nn, int(compress_ratio), hf)
            if page_shift is not None:
                _varlen_launcher(
                    r, npad, int(top_k), n_env_l, nn, int(compress_ratio), True, page_shift
                )
        return
    rows_max = rows_list[-1]
    # one allocation at the largest geometry; smaller row counts run on
    # contiguous prefix views (compile keys depend on shapes only). Both the
    # hinted and the hint-free engines are compiled: they are distinct
    # compiled objects and a serving stack may use either (``auto`` picks
    # hint-free whenever the caller passes no previous-step indices).
    logits = torch.zeros((rows_max, npad), dtype=torch.float32, device=dev)
    kv_lens = torch.full((rows_max // nn,), int(max_seq_len), dtype=torch.int32, device=dev)
    pre_idx = torch.zeros((rows_max // nn, int(top_k)), dtype=torch.int32, device=dev)
    out = torch.empty((rows_max, int(top_k)), dtype=torch.int32, device=dev)
    page_tab = None
    if page_shift is not None:
        page_tab = torch.zeros(
            (rows_max // nn, (npad + (1 << page_shift) - 1) >> page_shift), dtype=torch.int32, device=dev
        )
    for rows in rows_list:
        batch = rows // nn
        if page_tab is not None:
            run_varlen(
                logits[:rows],
                None,
                kv_lens[:batch],
                out[:rows],
                next_n=nn,
                compress_ratio=int(compress_ratio),
                max_seq_len=int(max_seq_len),
                top_k=int(top_k),
                page_table=page_tab[:batch],
                page_size=1 << page_shift,
            )
        for hint in (pre_idx[:batch], None):
            run_varlen(
                logits[:rows],
                hint,
                kv_lens[:batch],
                out[:rows],
                next_n=nn,
                compress_ratio=int(compress_ratio),
                max_seq_len=int(max_seq_len),
                top_k=int(top_k),
            )
    del logits, kv_lens, pre_idx, out, page_tab
    torch.cuda.synchronize()
    # band launches compiled every ENGINE; now populate the per-row-count
    # LAUNCHER cache entries for the exact requested row counts (pure host
    # work, zero allocation/launch — engines hit the compile cache), so a
    # CUDA-graph capture at any requested geometry finds its key immediately.
    for r in req_rows:
        for hf in (False, True):
            _varlen_launcher(r, npad, int(top_k), n_env_l, nn, int(compress_ratio), hf)
        if page_shift is not None:
            _varlen_launcher(
                r, npad, int(top_k), n_env_l, nn, int(compress_ratio), True, page_shift
            )
    with _VARLEN_WARMUP_LOCK:
        _VARLEN_WARMUP_DONE.add(key)


# --------------------------------------------------------------------------
# SGLang: K31HT_DSA_SELECT=gvrmf: DSA DCP candidate output from the streaming
# main family. Not upstream; the index-output paths above are unchanged.
# --------------------------------------------------------------------------
_NOCAND = ("nocand",)


def _varlen_cand_launcher(num_rows, npad, k, n_env, world, rank, p5b=0):
    """Main-family launcher compiled with candidate output for hint-free decode
    rows (next_n 1, compress ratio 1, unpaged), or ``_NOCAND`` when the varlen
    route of this capture key does not land on the non-SPLIT main family (the
    register / cluster families keep the index-output path; the caller then
    runs its own epilogue). A pure function of the key, so replay-safe."""
    sms = _sm_count()
    key = ("cand", num_rows, npad, k, n_env, int(world), int(rank), _arch_token(), sms, int(p5b))
    hit = _VARLEN_CACHE.get(key)
    if hit is not None:
        return hit
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    plan_free = route(num_rows, n_route, npad, k, sms=sms)
    lc = _NOCAND
    if plan_free["kernel"] not in ("reg_clus", "reg", "regimg", "clus"):
        # identical plan and scalars to _varlen_launcher's main tail (hint-free, next_n 1, cr 1)
        plan = route_streaming(num_rows, n_route, npad, k, force_main=True)
        tpl = tuple(plan["tpl"])
        rt = plan["rt"]
        r_const = rt["R"]
        if not tpl[5] and r_const == 1:
            fn = _gate_first_call(
                _device().get_compiled(
                    tpl[:6] + (False,) + (1, 0, r_const),
                    hint_free=True,
                    pt_smem=0,
                    pdl=_DECODE_PDL,
                    cand_world=int(world),
                    cand_rank=int(rank),
                    p5b=int(p5b),
                )
            )
            big = _big_regime(num_rows, r_const, n_route >> 2)
            aim_base = ((4 * k if k >= 1024 else 2 * k) if big else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2))
            sfac = 64 if k >= 1024 else 32
            amin = (7 * k) // 2
            sd_en = 1 if (k > 1024 and not big) else 0
            tsh_en = 0
            pre = (n_kernel, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
            lc = ("main", fn, pre, (aim_base, sfac, amin, sd_en, tsh_en))
    _VARLEN_CACHE[key] = lc
    return lc


def run_varlen_cand(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    world: int,
    rank: int,
    max_seq_len: int,
) -> bool:
    """Hint-free decode top-k (next_n 1, compress ratio 1) written as DSA DCP
    candidates: ``out`` int32 [num_rows, 2 * top_k] (a view of [rows, top_k, 2]),
    slot j of row r = (fp32 bits of the selected logit, column * world + rank),
    (0, -1) padded, slot order unspecified. Selection = run_varlen's (same kernel
    body, same plan). Returns False WITHOUT launching when the capture key routes
    to a family other than the non-SPLIT main family; True after the launch."""
    if logits.dtype is not _F32 or logits.dim() != 2 or logits.stride(1) != 1:
        raise RuntimeError("run_varlen_cand: logits must be 2-D float32 with inner stride 1")
    if not (isinstance(kv_lens, _TENSOR) and kv_lens.is_cuda and kv_lens.dtype is _I32 and kv_lens.dim() == 1):
        raise RuntimeError("run_varlen_cand: kv_lens must be a 1-D int32 CUDA tensor")
    num_rows = logits.shape[0]
    k = _index(top_k)
    world, rank = _index(world), _index(rank)
    if not (1 <= world and 0 <= rank < world):
        raise RuntimeError(f"run_varlen_cand: bad world / rank {world} / {rank}")
    if num_rows == 0:
        return True
    if kv_lens.shape[0] != num_rows or not kv_lens.is_contiguous():
        raise RuntimeError("run_varlen_cand: kv_lens must be contiguous [num_rows]")
    if (
        out.dtype is not _I32
        or tuple(out.shape) != (num_rows, 2 * k)
        or not out.is_contiguous()
        or out.data_ptr() & 15
        or not out.is_cuda
    ):
        raise RuntimeError("run_varlen_cand: out must be contiguous 16-byte aligned int32 [num_rows, 2 * top_k]")
    if num_rows > 1 and logits.stride(0) < logits.shape[1]:
        raise RuntimeError("run_varlen_cand: logits rows overlap")
    npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
    lg = logits
    if not logits.is_contiguous():
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 4 < need:
            raise RuntimeError("run_varlen_cand: logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 3 or lg.data_ptr() & 15:
        raise RuntimeError("run_varlen_cand: logits row stride must be a multiple of 4 and the base 16-byte aligned")
    n_env = min(max(int(max_seq_len), 1), npad)
    global p5b_calls
    sms = _sm_count()
    p5 = p5b_for(num_rows, sms)
    key = ("cand", num_rows, npad, k, n_env, world, rank, _arch_token(), sms, p5)
    lc = _VARLEN_CACHE.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError("run_varlen_cand: launcher not compiled for this shape - warm up before CUDA graph capture")
        lc = _varlen_cand_launcher(num_rows, npad, k, n_env, world, rank, p5)
    if lc is _NOCAND:
        return False
    _, fn, pre, tail = lc
    d = logits.get_device()
    ws = _ws_hot.get(_ws_key(d))
    if ws is None:
        ws = default_workspace(logits)  # raises under capture
    # hint-free: the pre_idx ABI slot is never read; pass the output buffer (int32, 16-byte aligned)
    fn(lg, out, out, ws, *pre, kv_lens, *tail)
    p5b_calls += p5
    return True
