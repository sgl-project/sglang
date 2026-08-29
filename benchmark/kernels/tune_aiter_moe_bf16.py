#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tune aiter's bf16 (a16w16) CK 2-stage fused-MoE for one MoE shape, end to end.

WHY THIS EXISTS
---------------
aiter picks its fused-MoE kernels from a CSV keyed on the *exact* shape --

    aiter/fused_moe.py:2114  _INDEX_COLS = [gfx, cu_num, token, model_dim,
                                            inter_dim, expert, topk, act_type,
                                            dtype, q_dtype_a, q_dtype_w, q_type,
                                            use_g1u1, doweight_stage1]

-- with no nearest-neighbour fallback on any key but ``token`` (and even that
only walks _PADDED_M_TIERS downward, i.e. only for token > 32768). A shape whose
``expert`` count is absent from the table therefore gets NOTHING, and
``get_2stage_cfgs`` drops into its ``cfg is None`` branch at
aiter/fused_moe.py:2320, which hands the CK dispatcher an empty kernel name.
The C++ side then falls back to a hard-coded heuristic:

    csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu:44-61
        if (kernelName != "") { ...lookup... }
        return moe_stage{1,2}_heuristic_dispatch(block_m, inter_dim, ...);

That heuristic is a three-branch ``if`` on block_m and ``inter_dim <= 192``. It
is not tuned for anything; it is a floor.

ZAYA1-74B at tp8 has expert=24, and NO aiter tuned CSV -- neither
``aiter/configs/tuned_fmoe.csv`` nor any of the 40+ ``model_configs/*.csv`` --
contains a single row with expert=24. The model's MoE therefore runs the
untuned CK default on every decode step. This script closes that gap.

WHAT IT SWEEPS
--------------
The knob space for this path is NOT free-form (no BLOCK_SIZE_M/num_warps knobs
like a Triton autotuner). aiter's CK 2-stage MoE is code-generated from a fixed
instance list, and the tuner picks among compiled instances:

  * ``block_m``  -- the moe_sorting block size. The tuner sweeps [16, 32, 64,
    128] (gemm_moe_tune.py:4847). 16 has no a16w16 stage1 instance, so the
    effective space is {32, 64, 128}. At runtime an untuned shape gets block_m
    from ``get_block_size_M`` (aiter/fused_moe.py:1178).
  * ``kernelName1`` -- one a16w16 stage1 instance whose MPerBlock == block_m.
    On gfx950 the list is ``a16w16_gemm1_kernels_list_gfx950``
    (gemm_moe_ck2stages_common.py:117): BLOCK_SIZE is always 256, MWaves x
    NWaves always 1x4, so the live axes are (MPerBlock, NPerBlock, KPerBlock,
    GemmPipelineVersion) over 8 enabled instances.
  * ``kernelName2`` -- likewise from ``a16w16_gemm2_kernels_list_gfx950``
    (:233), 12 enabled instances.
  * ``ksplit`` -- 0 for QuantType.No (``get_ksplit`` is only consulted for
    per_1x128 / per_1x32, aiter/fused_moe.py:2359).
  * ``run_1stage`` -- unavailable here: ``fused_moe_1stage_dict["gfx950"]``
    (aiter/fused_moe.py:1247) has no (Silu, QuantType.No, bf16, bf16, bf16)
    entry, and no g1u1 bf16 entry at all. gfx942 does; gfx950 does not.

Stage 1 and stage 2 are timed independently and the best of each is kept, so
the cost is |block_m| x (|stage1| + |stage2|) kernel launches, not the product.
For this shape that is 15 measured candidates per token tier.

MEASURED (MI355X, gfx950, ROCm 7.2, aiter c16d44b9, 2026-08-29)
---------------------------------------------------------------
The tune is real but small, and the shape of the win is instructive.

Tuning cost: 14 token tiers in 14.8 s -- this is not an expensive sweep.

What it changed: at every decode tier the tuner moved stage1 from the
heuristic's KPerBlock=64 to KPerBlock=128 (``256x32x64x64/v1`` ->
``256x32x64x128/v1``), and at a few tiers narrowed stage2 to ``256x32x64x64/v1``.

Per-MoE-layer, CUDA-graph captured (reps 2-4 agree to <0.15%):

    token     1      2      4      8     16     32     64    128   1024
    default  27.7   27.6   38.7   40.6   43.6   52.7   71.2   77.1   94.0 us
    tuned    26.9   27.1   31.4   33.6   37.3   47.9   63.2   71.3   87.9 us
    speedup  1.03x  1.02x  1.23x  1.21x  1.17x  1.10x  1.13x  1.08x  1.07x

The default heuristic has a cliff between token=2 and token=4 (27.6 -> 38.7 us)
that the KPerBlock=128 kernel does not, which is where the 20%+ wins live.

Projected onto ZAYA1-74B (60 MoE layers, tp8/dp4 reference TPOT):
    C=1   MoE = 1.66 ms of 15.00 ms TPOT (11%) -> saves 0.05 ms = 0.3%
    C=32  MoE = 3.16 ms of 16.45 ms TPOT (19%) -> saves 0.29 ms = 1.8%
    C=128 MoE = 4.62 ms of 20.98 ms TPOT (22%) -> saves 0.35 ms = 1.7%

So: worth landing, not a headline. The gap this kernel leaves on the table is
NOT tile selection. At token=1, topk=1-of-24 needs one expert's weights --
12.6 MB, ~1.6 us at HBM speed -- and the kernel takes 27 us. The CK 2-stage
MoE launches a grid over (all experts x N tiles) and lets the unrouted blocks
retire, so a 1-token decode pays close to a dense-over-experts launch. Re-tiling
that grid buys 3%; not launching it buys the other 17x.

Also note the eager-mode row: ~91 us flat from token=1 to token=512, i.e. ~60 us
of pure CPU launch overhead per call. Benchmark this path WITHOUT CUDA graphs
and every config measures identical, which reads as "the tune does nothing".

USAGE
-----
    # 0. what will be swept, and what the untuned default resolves to
    python tune_aiter_moe_bf16.py plan

    # 1. kernel tune (resumable; ~10s per token tier on MI355X)
    python tune_aiter_moe_bf16.py tune

    # 2. kernel-level A/B: untuned CK default vs the tuned row, per token tier.
    #    Verifies via aiter's own log line which path each arm actually took.
    python tune_aiter_moe_bf16.py micro

    # 3. end-to-end serving A/B (needs a served model)
    python tune_aiter_moe_bf16.py e2e --server-cmd ./zaya_run.sh

    # 4. summary
    python tune_aiter_moe_bf16.py report

Every phase writes into ``--outdir`` (default ./aiter_moe_tune) and is
resumable: re-running skips token tiers already present in the tuned CSV and
benchmark cells already present in results.jsonl.

The tuned CSV is consumed by aiter through ``AITER_CONFIG_FMOE=<path>``.
NOTE: that env var *replaces* the config set rather than adding to it
(aiter/jit/core.py:376 ``get_config_file`` -- when the env var is set, the
model_configs/ merge is skipped entirely). ``tune --merge-base`` therefore
concatenates the tuned rows onto a copy of the merged default table so a server
launched with AITER_CONFIG_FMOE does not silently lose every other model's
tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Shape under test
# --------------------------------------------------------------------------


@dataclass
class MoeShape:
    """One aiter fused-MoE tuning key, minus the (gfx, cu_num) the tuner fills in.

    Defaults are ZAYA1-74B at tp8: 24 experts, top-1, hidden 4096, and
    ffn_hidden_size 8192 -> per-side intermediate 4096 -> 512 per rank at tp8.
    """

    model_dim: int = 4096
    inter_dim: int = 512
    expert: int = 24
    topk: int = 1
    act_type: str = "ActivationType.Silu"
    dtype: str = "torch.bfloat16"
    q_dtype_a: str = "torch.bfloat16"
    q_dtype_w: str = "torch.bfloat16"
    q_type: str = "QuantType.No"
    use_g1u1: int = 1
    doweight_stage1: int = 0

    def untuned_row(self, token: int) -> str:
        return ",".join(
            str(x)
            for x in (
                token,
                self.model_dim,
                self.inter_dim,
                self.expert,
                self.topk,
                self.act_type,
                self.dtype,
                self.q_dtype_a,
                self.q_dtype_w,
                self.q_type,
                self.use_g1u1,
                self.doweight_stage1,
            )
        )


UNTUNED_HEADER = (
    "token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
)

# nextPow2 tiers, matching aiter's get_padded_M (aiter/fused_moe.py:1272).
# 1..64 is the decode range at the concurrencies ZAYA1 is served at;
# 1024..8192 covers prefill.
DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


# --------------------------------------------------------------------------
# Mirrors of aiter's untuned-path decisions, so `plan` and `report` can name
# the kernel the model gets *today* without running anything.
# --------------------------------------------------------------------------


def get_block_size_M(
    token: int, topk: int, expert: int, inter_dim: int, cu_num: int
) -> int:
    """Verbatim port of aiter/fused_moe.py:1178 get_block_size_M.

    Kept as a local copy (rather than imported) so `plan` runs on a host with
    no ROCm and no aiter installed.
    """
    tileN = 128
    tgN = (inter_dim + tileN - 1) // tileN
    out = []
    for el in (32, 64, 128):
        max_num_tokens = token * topk + expert * el - topk
        tg_num = tgN * (max_num_tokens + el - 1) // el
        rnd = (tg_num + cu_num - 1) // cu_num
        empty = cu_num - tg_num % cu_num
        out.append((rnd, empty, el))
    return min(out, key=lambda x: x[:2])[-1]


def default_heuristic_kernels(block_m: int, inter_dim: int) -> tuple[str, str]:
    """What ``moe_stage{1,2}_heuristic_dispatch`` returns for a16w16 on gfx950.

    Ported from the codegen templates
    ``A16W16_A8W8_gemm1_gfx950_heuristic_dispatch`` (gen_instances.py:74) and
    ``A16W16_gemm2_gfx950_heuristic_dispatch`` (:369). ``128/sizeof(bf16)`` is
    64 and ``256/sizeof(bf16)`` is 128. Returned as "MxNxK/vP" tile summaries
    (the C++ heuristic instantiates the template directly, so these instances
    have no lookup-table name of their own).
    """
    if block_m == 32:
        s1 = "256x32x64x64/v1"
    elif block_m == 64:
        s1 = "256x64x64x64/v1"
    elif block_m == 128:
        s1 = "256x128x64x64/v1" if inter_dim <= 192 else "256x128x128x64/v3"
    elif block_m == 256:
        s1 = "256x256x64x64/v1" if inter_dim <= 192 else "256x256x128x64/v3"
    else:
        s1 = f"<unsupported block_m={block_m}>"

    if block_m == 32:
        s2 = "256x32x64x64/v1" if inter_dim <= 192 else "256x32x128x128/v1"
    elif block_m == 64:
        s2 = "256x64x128x64/v1" if inter_dim <= 192 else "256x64x128x128/v1"
    elif block_m == 128:
        s2 = "256x128x64x64/v3" if inter_dim <= 192 else "256x128x128x64/v3"
    elif block_m == 256:
        s2 = "256x256x128x64/v3" if inter_dim <= 192 else "256x256x128x128/v3"
    else:
        s2 = f"<unsupported block_m={block_m}>"
    return s1, s2


def tile_summary(kernel_name: str) -> str:
    """'moe_ck2stages_gemm1_256x32x64x128_1x4_..._v1_...' -> '256x32x64x128/v1'."""
    tiles = re.search(r"_(\d+x\d+x\d+x\d+)_", kernel_name)
    ver = re.search(r"_v(\d+)_", kernel_name)
    if not tiles:
        return kernel_name
    return f"{tiles.group(1)}/v{ver.group(1) if ver else '?'}"


# --------------------------------------------------------------------------
# State / resumability
# --------------------------------------------------------------------------


@dataclass
class Paths:
    outdir: Path
    untuned: Path = field(init=False)
    tuned: Path = field(init=False)
    profile: Path = field(init=False)
    merged: Path = field(init=False)
    results: Path = field(init=False)
    logs: Path = field(init=False)

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.untuned = self.outdir / "untuned_fmoe.csv"
        self.tuned = self.outdir / "tuned_fmoe.csv"
        self.profile = self.outdir / "profile_fmoe.csv"
        self.merged = self.outdir / "tuned_fmoe_merged.csv"
        self.results = self.outdir / "results.jsonl"
        self.logs = self.outdir / "logs"
        self.logs.mkdir(exist_ok=True)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    import csv as _csv

    with path.open() as f:
        r = _csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def append_result(paths: Paths, rec: dict) -> None:
    with paths.results.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def load_results(paths: Paths) -> list[dict]:
    if not paths.results.exists():
        return []
    out = []
    for line in paths.results.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def done_cells(paths: Paths, phase: str) -> set:
    return {
        (r.get("phase"), r.get("arm"), r.get("token"), r.get("conc"), r.get("rep"))
        for r in load_results(paths)
        if r.get("phase") == phase
    }


# --------------------------------------------------------------------------
# phase: plan
# --------------------------------------------------------------------------


def phase_plan(args, paths: Paths, shape: MoeShape) -> int:
    cu = args.cu_num
    print(f"shape: {asdict(shape)}")
    print(f"gfx={args.gfx} cu_num={cu}")
    print()
    print("Per token tier, this is what the UNTUNED path resolves to today")
    print("(block_m from get_block_size_M; kernels from the C++ heuristic):")
    print()
    print(
        f"{'token':>7}  {'block_m':>7}  {'stage1 (default)':>20}  {'stage2 (default)':>20}"
    )
    for t in args.tokens:
        bm = get_block_size_M(t, shape.topk, shape.expert, shape.inter_dim, cu)
        s1, s2 = default_heuristic_kernels(bm, shape.inter_dim)
        print(f"{t:>7}  {bm:>7}  {s1:>20}  {s2:>20}")
    print()
    print("Sweep space per token tier (aiter gemm_moe_tune.py, CK a16w16 path):")
    print("  block_m in {32, 64, 128}   (16 has no a16w16 stage1 instance)")
    print("  stage1: instances with MPerBlock == block_m from")
    print("          a16w16_gemm1_kernels_list_gfx950  -> 2 per block_m, 6 total")
    print("  stage2: instances with MPerBlock == block_m from")
    print("          a16w16_gemm2_kernels_list_gfx950  -> 2/3/4 per block_m, 9 total")
    print("  ksplit: fixed 0 (QuantType.No never consults get_ksplit)")
    print("  1-stage asm: unavailable (no gfx950 bf16 g1u1 entry)")
    print(f"  => 15 timed candidates per tier, {15 * len(args.tokens)} total")
    return 0


# --------------------------------------------------------------------------
# phase: tune
# --------------------------------------------------------------------------


def find_aiter_tuner() -> Path:
    """Locate aiter's own MoE tuner. Prefer extending it over reimplementing."""
    cands = []
    try:
        import aiter  # noqa: F401
        from aiter.jit.core import AITER_CSRC_DIR

        cands.append(Path(AITER_CSRC_DIR))
    except Exception:
        pass
    for env in ("AITER_ROOT_DIR", "AITER_META_DIR"):
        v = os.environ.get(env)
        if v:
            cands.append(Path(v) / "csrc")
    cands.append(Path("/sgl-workspace/aiter/csrc"))
    for c in cands:
        p = c / "ck_gemm_moe_2stages_codegen" / "gemm_moe_tune.py"
        if p.exists():
            return p
    raise SystemExit(
        "could not find aiter's gemm_moe_tune.py; set AITER_ROOT_DIR or install aiter"
    )


def tuned_tokens(paths: Paths) -> set[int]:
    _, rows = read_csv(paths.tuned)
    return {int(r["token"]) for r in rows if r.get("token")}


def phase_tune(args, paths: Paths, shape: MoeShape) -> int:
    tuner = find_aiter_tuner()
    have = tuned_tokens(paths)
    todo = [t for t in args.tokens if t not in have]
    if not todo:
        print(f"[tune] nothing to do; {sorted(have)} already in {paths.tuned}")
    else:
        print(f"[tune] resuming: have {sorted(have)}, tuning {todo}")
        paths.untuned.write_text(
            UNTUNED_HEADER + "\n" + "\n".join(shape.untuned_row(t) for t in todo) + "\n"
        )
        log = paths.logs / f"tune-{int(time.time())}.log"
        cmd = [
            sys.executable,
            str(tuner),
            "-i",
            str(paths.untuned),
            "-o",
            str(paths.tuned),
            "-o2",
            str(paths.profile),
        ]
        print("[tune] " + " ".join(cmd))
        print(f"[tune] log -> {log}")
        t0 = time.time()
        with log.open("w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[tune] rc={rc} in {time.time() - t0:.1f}s")
        if rc != 0:
            print(f"[tune] FAILED -- see {log}")
            return rc

    _, rows = read_csv(paths.tuned)
    print(f"\n[tune] {len(rows)} tuned rows in {paths.tuned}")
    print(
        f"{'token':>7} {'bm':>4}  {'stage1 tuned':>18} {'stage2 tuned':>18}  {'us':>9}"
    )
    for r in sorted(rows, key=lambda r: int(r["token"])):
        print(
            f"{r['token']:>7} {r['block_m']:>4}  "
            f"{tile_summary(r['kernelName1']):>18} {tile_summary(r['kernelName2']):>18}  "
            f"{float(r['us']):>9.3f}"
        )

    if args.merge_base:
        merge_with_base(paths)
    return 0


def merge_with_base(paths: Paths) -> None:
    """Concatenate the tuned rows onto aiter's merged default table.

    AITER_CONFIG_FMOE replaces rather than augments the config set
    (aiter/jit/core.py:376), so a server pointed at a bare per-model CSV loses
    every other tuned shape. Build a superset instead.
    """
    try:
        from aiter.jit.core import AITER_CONFIGS

        base = Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    except Exception as e:  # noqa: BLE001
        print(f"[merge] cannot resolve aiter's base config ({e}); skipping merge")
        return
    if not base.exists():
        print(f"[merge] base {base} missing; skipping merge")
        return
    import csv as _csv

    bfields, brows = read_csv(base)
    tfields, trows = read_csv(paths.tuned)
    fields = list(dict.fromkeys(bfields + tfields))
    with paths.merged.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in brows + trows:
            w.writerow(r)
    print(
        f"[merge] {len(brows)} base + {len(trows)} tuned -> {paths.merged}\n"
        f"[merge] launch with AITER_CONFIG_FMOE={paths.merged}"
    )


# --------------------------------------------------------------------------
# phase: micro  (kernel-level A/B with path verification)
# --------------------------------------------------------------------------

_PATH_RE = re.compile(
    r"\[fused_moe\] using (\S+)(?: xbf16)? (default|\(kernelName1=.*)"
)


class PathSpy(logging.Handler):
    """Capture aiter's own '[fused_moe] using ...' dispatch line.

    aiter/fused_moe.py:2404 logs, once per distinct tuning key (the function is
    lru_cached), exactly which path it took:

        [fused_moe] using 2stage default for (gfx950, 256, 1, 4096, 512, 24, ...)
        [fused_moe] using 2stage (kernelName1='moe_ck2stages_gemm1_256x32x64x128...

    Without this, an arm whose config silently failed to load looks like a
    null result rather than the no-op A/B it is.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "[fused_moe] using" in msg:
            self.lines.append(msg.strip())

    def verdict(self) -> str:
        if not self.lines:
            return "UNKNOWN(no dispatch log seen)"
        last = self.lines[-1]
        if " default " in last:
            return "default"
        m = re.search(r"kernelName1='([^']*)'.*kernelName2='([^']*)'", last)
        if m:
            return f"tuned[{tile_summary(m.group(1))}+{tile_summary(m.group(2))}]"
        return "tuned[?]"


def _bench_once(fn, iters: int) -> float:
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def phase_micro(args, paths: Paths, shape: MoeShape) -> int:
    """A/B the untuned CK default against the tuned row, per token tier.

    Two arms, distinguished only by how aiter resolves the config:
      default : AITER_BYPASS_TUNE_CONFIG=1 -- forces the cfg-is-None branch at
                aiter/fused_moe.py:2320 even if a tuned row exists.
      tuned   : AITER_CONFIG_FMOE=<tuned csv>.
    Each arm is re-launched as a subprocess because aiter caches both the
    config table (module global ``cfg_2stages``) and the resolved metadata
    (``get_2stage_cfgs`` is lru_cached), so the two arms cannot coexist in one
    process.
    """
    if args._child_arm:
        return _micro_child(args, shape)

    done = {
        (r["arm"], r["token"]) for r in load_results(paths) if r.get("phase") == "micro"
    }
    for token in args.tokens:
        for arm in ("default", "tuned"):
            if (arm, token) in done and not args.force:
                print(f"[micro] skip {arm} token={token} (already done)")
                continue
            env = dict(os.environ)
            if arm == "default":
                env["AITER_BYPASS_TUNE_CONFIG"] = "1"
                env.pop("AITER_CONFIG_FMOE", None)
            else:
                env.pop("AITER_BYPASS_TUNE_CONFIG", None)
                env["AITER_CONFIG_FMOE"] = str(
                    paths.merged if paths.merged.exists() else paths.tuned
                )
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "micro",
                "--outdir",
                str(paths.outdir),
                "--tokens",
                str(token),
                "--reps",
                str(args.reps),
                "--iters",
                str(args.iters),
                "--_child-arm",
                arm,
            ]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            out = (p.stdout or "") + (p.stderr or "")
            rec = None
            for line in out.splitlines():
                if line.startswith("RESULT "):
                    rec = json.loads(line[len("RESULT ") :])
            if rec is None:
                print(f"[micro] {arm} token={token} FAILED\n{out[-2500:]}")
                continue
            append_result(paths, rec)
            g = rec.get("graph_us")
            print(
                f"[micro] token={token:>5} arm={arm:<7} eager={rec['us']:8.3f} "
                f"graph={g if g is None else format(g, '8.3f')} path={rec['path']}"
            )
    return _micro_report(paths)


def _micro_child(args, shape: MoeShape) -> int:
    import torch

    logging.basicConfig(level=logging.INFO)
    spy = PathSpy()
    logging.getLogger("aiter").addHandler(spy)
    logging.getLogger("aiter").setLevel(logging.INFO)

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    token = args.tokens[0]
    E, D, I, topk = shape.expert, shape.model_dim, shape.inter_dim, shape.topk
    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)

    hs = torch.randn(token, D, dtype=dt, device=dev) / 10
    # aiter layout: w1 [E, 2*inter, dim] (gate|up), w2 [E, dim, inter].
    w1 = (torch.randn(E, 2 * I, D, dtype=dt, device=dev) / 30).contiguous()
    w2 = (torch.randn(E, D, I, dtype=dt, device=dev) / 30).contiguous()
    # sglang shuffles both with (16, 16) before handing them to aiter
    # (srt/layers/quantization/unquant.py:495).
    w1s = shuffle_weight(w1, (16, 16))
    w2s = shuffle_weight(w2, (16, 16))
    ids = torch.randint(0, E, (token, topk), dtype=torch.int32, device=dev)
    wts = torch.rand(token, topk, dtype=torch.float32, device=dev)

    def run():
        fused_moe(
            hs,
            w1s,
            w2s,
            wts,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
            doweight_stage1=False,
        )

    run()  # populate the config lookup -> emits the dispatch log line
    torch.cuda.synchronize()

    # rep 1 is discarded everywhere: the first timed rep runs consistently slow
    # (~7% at the serving level; the same effect shows up here). reps 2..n then
    # agree to well under 1%.
    eager = [round(_bench_once(run, args.iters), 4) for _ in range(args.reps)]

    # The graph number is the one that matters. Eagerly, this call is dominated
    # by CPU launch overhead -- it measures ~91us flat from token=1 to token=512
    # on MI355X, which buries the GEMM entirely and makes every config look
    # identical. sglang decodes under CUDA graphs, where that overhead is gone
    # and the kernel choice is what is left.
    graph, gerr = [], ""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        torch.cuda.synchronize()
        graph = [round(_bench_once(g.replay, args.iters), 4) for _ in range(args.reps)]
    except Exception as e:  # noqa: BLE001
        gerr = repr(e)[:200]

    def mean_after_first(xs):
        kept = xs[1:] or xs
        return round(sum(kept) / len(kept), 4)

    rec = {
        "phase": "micro",
        "arm": args._child_arm,
        "token": token,
        "us": mean_after_first(eager),
        "reps_us": eager,
        "graph_us": mean_after_first(graph) if graph else None,
        "graph_reps_us": graph,
        "graph_err": gerr,
        "path": spy.verdict(),
        "dispatch_log": spy.lines[-1] if spy.lines else "",
    }
    print("RESULT " + json.dumps(rec))
    return 0


def _micro_report(paths: Paths) -> int:
    rows = [r for r in load_results(paths) if r.get("phase") == "micro"]
    by = {}
    for r in rows:
        by[(r["token"], r["arm"])] = r
    tokens = sorted({t for t, _ in by})
    print()
    print(
        f"{'token':>7} {'dflt eager':>11} {'tuned eag':>10} {'spd':>7} | "
        f"{'dflt graph':>11} {'tuned grph':>10} {'spd':>7}   paths"
    )
    for t in tokens:
        d, u = by.get((t, "default")), by.get((t, "tuned"))
        if not (d and u):
            continue
        sp = d["us"] / u["us"] if u["us"] else float("nan")
        flag = ""
        if d["path"] != "default" or not u["path"].startswith("tuned"):
            flag = "  <-- ARMS DID NOT DIVERGE, result is a no-op A/B"
        dg, ug = d.get("graph_us"), u.get("graph_us")
        gcol = (
            f"{dg:>11.3f} {ug:>10.3f} {dg / ug:>6.3f}x"
            if dg and ug
            else f"{'-':>11} {'-':>10} {'-':>7}"
        )
        print(
            f"{t:>7} {d['us']:>11.3f} {u['us']:>10.3f} {sp:>6.3f}x | {gcol}   "
            f"{d['path']} / {u['path']}{flag}"
        )
    print(
        "\nThe graph columns are the ones to read: eagerly this call is CPU-launch "
        "bound\nand flat across token, which hides the kernel entirely."
    )
    return 0


# --------------------------------------------------------------------------
# phase: e2e
# --------------------------------------------------------------------------


def phase_e2e(args, paths: Paths, shape: MoeShape) -> int:
    """Serving A/B. Rep 1 of every cell is discarded (see --reps)."""
    runner = Path(args.server_cmd)
    if not runner.exists():
        print(f"[e2e] --server-cmd {runner} not found")
        return 2
    cfg = paths.merged if paths.merged.exists() else paths.tuned
    if not cfg.exists():
        print(f"[e2e] no tuned config at {cfg}; run `tune` first")
        return 2

    done = {
        (r["arm"], r["conc"], r["rep"])
        for r in load_results(paths)
        if r.get("phase") == "e2e"
    }
    for arm in args.arms:
        env = dict(os.environ)
        if arm == "tuned":
            env["AITER_CONFIG_FMOE"] = str(cfg)
        else:
            env.pop("AITER_CONFIG_FMOE", None)
        needed = [
            (c, rep)
            for c in args.conc
            for rep in range(1, args.reps + 1)
            if (arm, c, rep) not in done or args.force
        ]
        if not needed:
            print(f"[e2e] arm={arm} already complete")
            continue

        slog = paths.logs / f"server-{arm}.log"
        print(
            f"[e2e] arm={arm}: serving (AITER_CONFIG_FMOE={env.get('AITER_CONFIG_FMOE', '<unset>')})"
        )
        subprocess.run([str(runner), "stop"], env=env)
        rc = subprocess.run(
            [str(runner), "serve", f"moetune-{arm}"], env=env
        ).returncode
        if rc != 0:
            print(f"[e2e] serve failed for {arm}")
            return rc
        if (
            subprocess.run([str(runner), "wait", f"moetune-{arm}"], env=env).returncode
            != 0
        ):
            print(f"[e2e] server never became ready for {arm}")
            return 1

        for c, rep in needed:
            tag = f"moetune-{arm}-c{c}-rep{rep}"
            subprocess.run(
                [
                    str(runner),
                    "bench",
                    f"moetune-{arm}",
                    str(args.isl),
                    str(args.osl),
                    str(c),
                    "1",
                ],
                env=env,
            )
            rec = {
                "phase": "e2e",
                "arm": arm,
                "conc": c,
                "rep": rep,
                "tag": tag,
                "discard": rep == 1,
            }
            rec.update(_scrape_bench(Path(args.results_dir), arm, c))
            rec["path"] = _server_path_verdict(paths, arm, args.server_log_dir)
            append_result(paths, rec)
            print(f"[e2e] {tag}: tpot={rec.get('median_tpot_ms')} path={rec['path']}")
        subprocess.run([str(runner), "stop"], env=env)
    return phase_report(args, paths, shape)


def _scrape_bench(resdir: Path, arm: str, conc: int) -> dict:
    """Pull the newest bench_serving jsonl for this cell."""
    if not resdir.exists():
        return {}
    cands = sorted(
        resdir.glob(f"*{arm}*c{conc}*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        return {}
    try:
        last = [json.loads(l) for l in cands[0].read_text().splitlines() if l.strip()][
            -1
        ]
    except Exception:  # noqa: BLE001
        return {}
    return {
        k: last.get(k)
        for k in (
            "median_tpot_ms",
            "median_ttft_ms",
            "output_throughput",
            "total_token_throughput",
        )
    }


def _server_path_verdict(paths: Paths, arm: str, logdir: str | None) -> str:
    """Grep the server log for aiter's dispatch line for the MoE shape.

    A tuned arm whose CSV did not load logs '2stage default' exactly like the
    baseline; without this check the two arms are indistinguishable and a null
    result is unfalsifiable.
    """
    cands: list[Path] = []
    if logdir:
        cands += sorted(Path(logdir).glob(f"server-*{arm}*.log"))
    cands += sorted(paths.logs.glob(f"server-*{arm}*.log"))
    for p in reversed(cands):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        hits = [l for l in txt.splitlines() if "[fused_moe] using" in l]
        if not hits:
            continue
        moe = [l for l in hits if ", 24, " in l] or hits
        last = moe[-1]
        if " default " in last:
            return "default"
        m = re.search(r"kernelName1='([^']*)'.*kernelName2='([^']*)'", last)
        if m:
            return f"tuned[{tile_summary(m.group(1))}+{tile_summary(m.group(2))}]"
        return "tuned[?]"
    return "UNKNOWN(no dispatch log in server output)"


# --------------------------------------------------------------------------
# phase: report
# --------------------------------------------------------------------------


def phase_report(args, paths: Paths, shape: MoeShape) -> int:
    _, tuned = read_csv(paths.tuned)
    if tuned:
        print("=== tuned rows ===")
        cu = args.cu_num
        print(
            f"{'token':>7} {'bm':>4}  {'stage1 default':>18} -> {'stage1 tuned':<18}"
            f"  {'stage2 default':>18} -> {'stage2 tuned':<18} {'us':>9}"
        )
        for r in sorted(tuned, key=lambda r: int(r["token"])):
            t = int(r["token"])
            dbm = get_block_size_M(t, shape.topk, shape.expert, shape.inter_dim, cu)
            d1, d2 = default_heuristic_kernels(dbm, shape.inter_dim)
            print(
                f"{t:>7} {r['block_m']:>4}  {d1:>18} -> {tile_summary(r['kernelName1']):<18}"
                f"  {d2:>18} -> {tile_summary(r['kernelName2']):<18} {float(r['us']):>9.3f}"
            )
    micro = [r for r in load_results(paths) if r.get("phase") == "micro"]
    if micro:
        print("\n=== kernel-level A/B ===")
        _micro_report(paths)
        if args.moe_layers:
            tmap = {}
            for pair in (args.tpot_map or "").split(","):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    tmap[int(k)] = float(v)
            print(
                f"\n=== projected e2e, {args.moe_layers} MoE layers ===\n"
                "Under DP attention the MoE sees the DP-gathered batch, so token == "
                "concurrency."
            )
            by = {(r["token"], r["arm"]): r for r in micro}
            print(
                f"{'token':>7} {'TPOT ms':>8} {'MoE/step ms':>12} {'% of TPOT':>10} "
                f"{'saved ms':>9} {'% TPOT':>8}"
            )
            for t in sorted({k[0] for k in by}):
                d, u = by.get((t, "default")), by.get((t, "tuned"))
                if not (d and u and d.get("graph_us") and u.get("graph_us")):
                    continue
                tpot = tmap.get(t, args.tpot_ms)
                star = "" if t in tmap else "*"
                moe_ms = d["graph_us"] * args.moe_layers / 1000.0
                saved_ms = (d["graph_us"] - u["graph_us"]) * args.moe_layers / 1000.0
                print(
                    f"{t:>7} {tpot:>7.2f}{star} {moe_ms:>12.3f} "
                    f"{100 * moe_ms / tpot:>9.1f}% {saved_ms:>9.3f} "
                    f"{100 * saved_ms / tpot:>7.2f}%"
                )
            print(
                "\n* = no measured TPOT for that concurrency; --tpot-ms used, so the\n"
                "  percentages on those rows are over-stated (TPOT grows with C).\n"
                "The '% of TPOT' column is the ceiling: it is what deleting the MoE\n"
                "entirely would buy. A tune can only ever take a slice of it."
            )
    e2e = [
        r
        for r in load_results(paths)
        if r.get("phase") == "e2e" and not r.get("discard")
    ]
    if e2e:
        print("\n=== e2e (rep 1 discarded) ===")
        agg: dict = {}
        for r in e2e:
            agg.setdefault((r["conc"], r["arm"]), []).append(r)
        for conc in sorted({c for c, _ in agg}):
            line = [f"C={conc:<5}"]
            for arm in ("baseline", "tuned"):
                rs = agg.get((conc, arm), [])
                vals = [r["median_tpot_ms"] for r in rs if r.get("median_tpot_ms")]
                if vals:
                    line.append(
                        f"{arm}: tpot={sum(vals) / len(vals):.3f}ms n={len(vals)} "
                        f"path={rs[-1].get('path')}"
                    )
            print("  " + "   ".join(line))
    return 0


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("phase", choices=["plan", "tune", "micro", "e2e", "report"])
    p.add_argument("--outdir", default="./aiter_moe_tune", type=Path)
    p.add_argument(
        "--tokens",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=DEFAULT_TOKENS,
        help="padded-M tiers to tune (aiter keys on nextPow2 of the batch)",
    )
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=512, help="per-rank intermediate")
    p.add_argument("--expert", type=int, default=24)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--gfx", default="gfx950")
    p.add_argument("--cu-num", type=int, default=256)
    p.add_argument("--force", action="store_true", help="redo cells already recorded")
    # tune
    p.add_argument(
        "--merge-base",
        action="store_true",
        default=True,
        help="also emit a CSV of aiter's defaults + these rows (AITER_CONFIG_FMOE replaces, "
        "it does not augment)",
    )
    p.add_argument("--no-merge-base", dest="merge_base", action="store_false")
    # micro
    p.add_argument(
        "--reps", type=int, default=4, help="benchmark reps; rep 1 is discarded"
    )
    p.add_argument("--iters", type=int, default=200)
    p.add_argument(
        "--_child-arm", dest="_child_arm", default=None, help=argparse.SUPPRESS
    )
    # e2e
    p.add_argument("--server-cmd", default="./zaya_run.sh")
    p.add_argument("--results-dir", default="/persistent/results")
    p.add_argument("--server-log-dir", default="/persistent/logs")
    p.add_argument(
        "--arms", type=lambda s: tuple(s.split(",")), default=("baseline", "tuned")
    )
    p.add_argument(
        "--conc",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(1, 32, 128),
    )
    p.add_argument("--isl", type=int, default=1024)
    p.add_argument("--osl", type=int, default=128)
    # report: turn per-layer kernel deltas into an e2e projection
    p.add_argument(
        "--moe-layers",
        type=int,
        default=60,
        help="MoE layers in the model (ZAYA1-74B: 120 layers, every other one)",
    )
    p.add_argument(
        "--tpot-ms",
        type=float,
        default=15.0,
        help="baseline median TPOT, used for any token tier not in --tpot-map",
    )
    p.add_argument(
        "--tpot-map",
        default="1:15.00,32:16.45,128:20.98",
        help="token:tpot_ms pairs -- TPOT grows with concurrency, so a single "
        "number over-states the saving at large token counts. Default is the "
        "ZAYA1-74B tp8/dp4 reference (C=1/32/128).",
    )
    args = p.parse_args(argv)

    shape = MoeShape(
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        expert=args.expert,
        topk=args.topk,
    )
    paths = Paths(args.outdir)
    return {
        "plan": phase_plan,
        "tune": phase_tune,
        "micro": phase_micro,
        "e2e": phase_e2e,
        "report": phase_report,
    }[args.phase](args, paths, shape)


if __name__ == "__main__":
    raise SystemExit(main())
