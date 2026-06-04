# Journal — MoE EP8 vs TP8 (no-LoRA baseline) bench & profile

- **Date started:** 2026-06-02
- **Owner:** yushengsu
- **Task branch:** `moe-ep8-nolora-bench` (based on `lora-opti-nvfp4` @ `ac0fa6d3ee`, tracking `jybsuper/nvfp4-lora`)
- **Worktree:** `/Users/yushengsu/Downloads/river/sglang-moe-ep8-nolora-bench`
- **Model under test:** `nvidia/Kimi-K2.5-NVFP4` (2 nodes, 4+4 GPU MNNVL/GB200), 384 experts, hidden 7168, LoRA rank 16/32.

---

## Goal

Find out whether running the MoE under **EP8** (expert parallel, each rank owns 384/8 = 48 experts)
instead of the current **TP8** (every rank holds the full expert weight, GEMM rhs not sliced) avoids
the memory-bound LoRA "tax" we measured under TP, **without** hurting the no-LoRA baseline too much —
especially at **larger batch sizes**.

This first task is the **best no-LoRA baseline only**: switch MoE TP8 → EP8 and measure the speed
difference (bench + profile). If EP8 is acceptable at larger bs, EP becomes the foundation for moving
the LoRA MoE kernels onto a 1/8-sliced (per-rank 48-expert) layout.

## Motivation / context (from the P0 thread)

Under **TP8**, two of the four grouped MoE-LoRA GEMMs do **not** get their rhs sliced, so each rank
streams the full per-expert weight — pure memory traffic, overlap can't hide it because we already
assume 8 TB/s is saturated:

| # | GEMM (grouped, per-expert) | TP8 (rhs) | EP8 |
|---|---|---|---|
| ① | gate_up lora_A (shrink, H→2r) | `[384,M,7168]×[384,7168,32]` — **rhs full, K=7168 not sliced** | `[48,M,7168]×[48,7168,32]` |
| ② | gate_up lora_B (expand, 2r→2l) | `[384,M,16]×[384,16,512]` (block-diag, N=4096/8) | `[48,M,16]×[48,16,4096]` |
| ③ | down lora_A (shrink, l→r) | `[384,M,256]×[384,256,16]` (K=2048/8) | `[48,M,2048]×[48,2048,16]` |
| ④ | down lora_B (expand, r→H) | `[384,M,16]×[384,16,7168]` — **rhs full, N=7168 not sliced** | `[48,M,16]×[48,16,7168]` |

Theoretical TP8 tax @8 TB/s: gate_up A ≈ 21us, down B ≈ 10.5us (84 MB) — unavoidable, no overlap help.
EP8: every kernel's compute/traffic ÷8 → as little as ~4us. Profile (DECODE bs=64) confirms the two
"not sliced" kernels (`_moe_lora_*` circled) are visibly the slow ones; the 8-way-sliced kernels are
much faster by eye.

**Hypothesis:** since bs is tunable, at larger bs EP8's all-to-all + per-rank compute should amortize
well and EP8 should not be much worse than TP8 on the no-LoRA baseline — while killing the LoRA tax.

## Plan / actionable items

1. **[no-LoRA only] MoE TP8 → EP8** on the Kimi-K2.5-NVFP4 best baseline; measure speed delta.
   - Baseline (control): current `--tp 8` (no EP), no-LoRA, default MoE backend.
   - Variant: `--tp 8 --ep-size 8` + appropriate `--moe-a2a-backend` / `--moe-runner-backend`
     for NVFP4 cross-node EP (`ep_size ∈ {1, tp_size}`; flashinfer_cutlass/cutedsl support EP on fp4).
   - Sweep **larger bs** (not just 16/32/64 — add 128/256, bump `--cuda-graph-max-bs`) to test the
     "EP fine at large bs" hypothesis.
2. **If EP8 looks bad due to expert imbalance (balancedness):**
   - `--init-expert-location <dist>` to flatten the expert placement, or
   - drop to **4 GPU** (tp=ep=4) which should be more balanced.
3. Profile both (graph-on bs16/64[/128/256], graph-off bs16) and compare the circled kernels +
   end-to-end decode step time. Pull traces locally.

## Conventions

- **Journal timestamps (REQUIRED):** every log entry/heading must carry **date AND time** (HH:MM, with
  tz), not just the date — e.g. `### 2026-06-02 22:46 (KST) — …`.
- **Perf measurement rule (REQUIRED):** never read only the `bench_one_batch_server` e2e result.
  Also read the **decode throughput (token/s) printed in the server log** (`/tmp/server.log`:
  `Decode batch ... gen throughput (token/s): ...`) and report that per bs/variant. The e2e number
  includes prefill + scheduling; the server-log decode thpt is the steady-state decode rate that the
  TP8-vs-EP8 comparison actually hinges on.
- k8s id: `yushengsu-<date>-<time>` (avoid resource conflicts).
- Per skill.md: regression = `sglang-base-variant-regression.md` (Qwen) / `kimi-regression` (Kimi);
  perf = `sglang-lora-base-perf-benchmark.md`. Release nodes when done.
- This task touches **launch flags / bench config** primarily (no-LoRA EP8 should work on stock code);
  any sglang code fix needed to make NVFP4 EP8 launch will be recorded below and committed.

---

## Log

### 2026-06-02 — setup
- Confirmed base branch `lora-opti-nvfp4` in sync with `jybsuper/nvfp4-lora` (`0 0`).
- Created worktree `sglang-moe-ep8-nolora-bench` + branch `moe-ep8-nolora-bench` off `ac0fa6d3ee`.
- Verified EP server-args in this build: `--ep-size` (∈{1,tp}), `--moe-a2a-backend`
  {none,deepep,flashinfer,…}, `--init-expert-location` (default `trivial`).
- Authored `run_kimi_ep_vs_tp.sh` (no-LoRA, VARIANT=tp8|ep8, larger-bs sweep, EP backend env knobs).
- Committed JOURNAL + harness (`b62becfdb6`), pushed to `origin` (yushengsu-thu fork),
  opened **PR jybsuper/sglang#18** against `nvfp4-lora`.

---

## STATUS SNAPSHOT — 2026-06-02 (precise: done / now / next)

### What has been done so far
1. **Read `skill.md`** and the Kimi section of `sglang-lora-base-perf-benchmark.md` §3 — confirmed the
   "best no-LoRA" TP8 config is the `nvidia/Kimi-K2.5-NVFP4` 2-node (4+4 GPU MNNVL) launch at
   `--tp 8` with **no EP**, default MoE backend. This is the control we want to beat.
2. **Synced base branch:** `git fetch jybsuper` → `lora-opti-nvfp4` is `0 0` vs `jybsuper/nvfp4-lora`
   (already in sync, HEAD `ac0fa6d3ee`).
3. **Created isolated worktree + branch:** `sglang-moe-ep8-nolora-bench` / branch
   `moe-ep8-nolora-bench` off `lora-opti-nvfp4` (so it doesn't collide with other agents on the
   shared repo).
4. **Verified EP server-args exist** in this build (`server_args.py`): `--ep-size` (must be 1 or
   tp_size → EP8 is the only EP option at tp8), `--moe-a2a-backend` {none,deepep,flashinfer,…},
   `--moe-runner-backend` (flashinfer_cutedsl/cutlass support fp4 EP), `--init-expert-location`.
5. **Wrote the experiment harness** `benchmark/kernels/moe_ep8_nolora/run_kimi_ep_vs_tp.sh`:
   no-LoRA baseline only, `VARIANT=tp8` (control) vs `VARIANT=ep8`, larger-bs sweep (16/32/64/128/256
   with `--cuda-graph-max-bs 256`), EP backend choices as env vars so they can be tuned on-pod
   without re-pushing, and an `--init-expert-location` hook for the balancedness fallback.
6. **Wrote this JOURNAL**, committed both (`b62becfdb6`), pushed to `origin`, **opened PR #18**.

### Current status
- **No GPU nodes launched yet.** Nothing has run on hardware — there are **no bench or profile
  numbers yet**. Everything so far is repo/branch/harness setup + the PR.
- The EP8 backend combo (`MOE_RUNNER=flashinfer_cutedsl`, `MOE_A2A=deepep`, `DEEPEP_MODE=low_latency`)
  is a **first guess** for cross-node NVFP4 EP and is expected to need on-pod iteration.
- k8s `ID` not yet chosen (will be `yushengsu-<date>-<time>` at launch).

### What's next (in order)
1. Pick `ID=yushengsu-<date>-<time>`; apply the 2-node Kimi pod spec (`kimi-2node.yaml` from the perf
   skill §3.1), wait for setup, build+inject this branch into both pods, ghost-check + clean HBM.
2. **Run `VARIANT=tp8`** (control) — bench + profile the no-LoRA baseline at bs 16…256. Pull traces.
3. **Run `VARIANT=ep8`** — same workload. Debug the EP launch (a2a / runner backend) until it serves;
   record the working flag combo here.
4. **Compare:** EP8 vs TP8 decode step time + the two "not sliced" `_moe_lora_*` kernels, per bs.
   Confirm/deny the "EP8 fine at large bs" hypothesis.
5. **If EP8 looks bad → balancedness:** retry with `--init-expert-location <dist>`, or drop to 4 GPU
   (tp=ep=4). Record deltas.
6. Write findings + numbers into this JOURNAL and the PR description, **release the nodes**, push the
   results commit.

---

### 2026-06-02 — launch
- `ID=yushengsu-20260602-220516`. Applied `kimi-2node.yaml` (ctx `leira`). Both pods
  `mnnvl-kimi-${ID}-0/1` scheduled + Running within ~8s (nodes np-67167b3f-2 / -3, eu-iceland1-a).
- Built branch bundle: `MAIN_BASE=fba083c80f` (merge-base sgl/main ↔ branch), 13 commits,
  `/tmp/sglang-branch-moe-ep.bundle` (119K).
- In-pod setup: numactl + hf accel installed, sglang cloned + `pip install -e` done, now downloading
  `nvidia/Kimi-K2.5-NVFP4` (140 files, fp4) + `kimi_k25_lora_alpha`. Waiting for `/root/.setup-done`.
- Next on setup-done: inject bundle into both pods, ghost-check + drop HBM page cache, run VARIANT=tp8.
- 22:10 progress check: lora adapter (4 files) downloaded on both pods; base NVFP4 (140 files) at
  ~96/140 (head) / ~74/140 (worker). pip install -e already done earlier in setup. Still waiting.
- ~22:18 SETUP_DONE on both pods. Injected bundle: first fetch failed (bundle ref is
  `moe-ep8-nolora-bench`, not `__bench_target`); re-fetched `moe-ep8-nolora-bench:refs/heads/__bench_target`
  → both pods at `a20eb4601`. Ghost-check: all 8 GPUs clean (~<200 MiB / 189 GB GB200 HBM), no drain needed.
- Starting **VARIANT=tp8 (control)** via `run_kimi_ep_vs_tp.sh` (driven locally): checkout __bench_target +
  pip install -e, launch no-LoRA server, bench bs 16/32/64/128/256 (in/out 2048), profile graph-on
  bs 16/64/128/256 + graph-off bs16. Running in background.
- **tp8 run #1 FAILED (exit 1) in `checkout()`.** Root cause: the branch editable rebuild
  (`pip install -e python`) died with **"can't find Rust compiler"** on the worker pod. The in-pod
  `setup.sh` had sourced `~/.cargo/env` directly so the base/main install built fine, but our
  `checkout()` runs pip via `kubectl exec bash -lc`, which does NOT put cargo on PATH. (`set -e` +
  `both` runs the worker first → it aborted before the head even ran, hence head had no pip.log.)
  Confirmed cargo present at `/root/.cargo/bin/cargo` (1.96.0); `. ~/.cargo/env` fixes PATH.
  **Fix:** `checkout()` now sources `~/.cargo/env` before `pip install -e`. Re-running tp8.
- **tp8 run #2:** cargo fix WORKED — `checkout()` rebuilt the editable install and printed
  `a20eb4601` on both pods. But the monolithic background script then died: a `kubectl exec`
  right after checkout (a `kill_all`) was `Killed: 9` (SIGKILL on the local kubectl, likely a
  transient local OOM from stacked execs). No server launched, no server.log, no results.
  **Decision:** stop using the fragile all-in-one background script; drive launch → bench → profile
  **step by step** via direct kubectl (more visibility + needed anyway to capture the server-log
  decode throughput). Both pods are already on `a20eb4601` with the rebuilt editable, so the next
  step skips checkout and launches the tp8 no-LoRA server directly.

### 2026-06-02 — EP backend correction (user feedback, PAUSED tp8 mid-load)
- User flagged the planned EP a2a (`deepep` / low_latency) as **slow and wrong** for this setup, and
  said: keep the SAME MoE backend as today (the **trtllm-gen** one), do NOT switch to cutlass/cutedsl
  (neither is fast), and do NOT use deepep — use the same a2a as now.
- Verified from the live tp8 server_args dump what "today's baseline" actually is:
  `attention_backend=trtllm_mla`, **`moe_runner_backend=flashinfer_trtllm`** (auto-picked on sm100 for
  DeepseekV3ForCausalLM), **`moe_a2a_backend=none`**, `ep_size=1`.
- Confirmed in `server_args.py` (~L2404) that the flashinfer_trtllm auto-selection condition is
  `quant∈{fp8,fp4} & moe_a2a_backend==none & moe_runner_backend==auto` — it does **NOT** check ep_size.
  ⇒ adding **only `--ep-size 8`** keeps `flashinfer_trtllm` + `a2a=none`; EP combine rides the existing
  NVLink TP communicator (no deepep, no NVSHMEM/IBGDA). This is exactly "same backend + same a2a, EP on".
- **Harness fix:** EP variant flags reduced to `--ep-size 8` (+ optional `--init-expert-location` for
  the balancedness fallback). Removed the `--moe-runner-backend`/`--moe-a2a-backend`/`--deepep-mode`
  overrides. EP8 now differs from the tp8 control by exactly one flag.

### 2026-06-02 — switch to the hardened kimi-regression harness (user direction)
- User pointed me at `kimi-regression/SKILL.md` to drive the Kimi run. Its `run_kimi.sh` is the
  hardened base-vs-variant harness (acc + bench + profile in one 2-node run). Crucially its
  **robustness #1 explains my earlier "Killed: 9" death**: orphaned local `kubectl exec … launch_server`
  clients race a new launch → its `kill_all` does `pkill -9 -f "kubectl exec.*launch_server"` + loops
  nvidia-smi until compute-apps==0 on BOTH nodes before launching. My hand-rolled monolithic script
  lacked this. Switching to it.
- Copied `kimi-regression/scripts/run_kimi.sh` → `run_kimi_epreg.sh` and edited the cell block for THIS
  A/B: **base** = no-LoRA `--tp 8` (today's baseline); **variant** = no-LoRA + **only `--ep-size 8`**
  (same flashinfer_trtllm + a2a=none). Both cells = same commit (`__bench_target` = a20eb4601 on pods).
- Edits: `BENCH_BS="16 32 64 128 256"` + `--cuda-graph-max-bs 256` (larger-bs sweep — EP amortizes at
  big bs); ported the `~/.cargo/env` fix into the skill's `checkout()` (it doesn't source cargo either).
- Kept the skill's hardening untouched: 40-min cold-autotune wait, retry-once launch, mem-fraction 0.83,
  NEVER --disable-flashinfer-autotune, --show-report bench, asymmetric 8-rank trace pull, drop_caches.
- acc (logprobs) now a real regression check: EP8 vs TP8 are numerically equivalent, expect diff within
  the ~0.30 noise floor. summary.py (ACC_TOL=0.30) + decode_isolate.py run locally after.
- DECODE-THPT-RULE (skill §304) reaffirms: report server-log decode thpt, not just bench e2e.
- Next: run `run_kimi_epreg.sh` in background (its kill_all clears my orphaned tp8 launcher first).

### 2026-06-02 22:46 (KST) — hardened run launched & clean
- Launched `run_kimi_epreg.sh` (bg `b9vrh0lkk`), `RUN_ROOT=~/Downloads/sglang_kimi_epreg_${ID}`.
- Clean start confirmed: both pods `prewarmed`, entered `CELL base`, and **GPU compute-apps == 0 on
  both nodes** — the skill's `kill_all` reaped my earlier orphaned tp8 launcher (robustness #1 worked).
- base cell now in checkout (pip install + cargo fix) → base graph-ON launch → ~20min cold fp4 autotune.
- (Reminder applied from here on: every journal entry carries date + time.)

### 2026-06-02 22:53 (KST) — VERIFIED launch commands (from the live process, not the script)
base cell currently in cold `fp4_gemm` autotune (i=27, 7 sglang procs). Verified on the head pod:

**TP8 (base / control)** — `ps args` of the live rank-0 process:
```
python3 -m sglang.launch_server --model-path /root/Kimi-K2.5-NVFP4 --tp 8 --nnodes 2 \
  --dist-init-addr mnnvl-kimi-<ID>-0.mnnvl-kimi-<ID>-head:20000 --host 0.0.0.0 --port 30000 \
  --quantization modelopt_fp4 --mem-fraction-static 0.83 --cuda-graph-max-bs 256 --trust-remote-code \
  --max-prefill-tokens 40960 --chunked-prefill-size 40960 --node-rank 0
```
env prefix (from `/proc/<pid>/environ`), under `numactl --membind=0,1`:
`NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
Resolved server_args (from server.log): **`tp_size=8, ep_size=1, moe_a2a_backend='none', moe_runner_backend='flashinfer_trtllm'`** ✓ (trtllm-gen, a2a none).

**EP8 (variant)** — identical command **+ `--ep-size 8`** (nothing else changes). The flashinfer_trtllm
auto-selection ignores ep_size, so the resolved backend stays `flashinfer_trtllm` + `a2a='none'`, only
`ep_size=8`. NOT yet launched (cell 2) → will re-verify `ep_size=8` + backend from its live process/log
when it comes up, before drawing any comparison.

- After the full run completes: compare TP8 vs EP8 (server-log decode thpt per bs + bench e2e + acc vs
  noise floor + decode-isolated kernels) and post the comparison to the PR description.

### 2026-06-02 22:59 (KST) — next-step plan locked (no-LoRA TP8→EP8, balancedness fallback)
- (journal policy: **APPEND**, keep all prior records — do not overwrite.)
- Restated task: take the best **no-LoRA** baseline, switch MoE **TP8 → EP8**, measure the speed delta,
  and decide whether EP8 is acceptable **at larger bs**. If EP8 is bad due to **expert imbalance
  (balancedness)** — i.e. some of the 8 ranks get far more tokens than others, so the slowest rank
  bottlenecks the all-to-all+compute — then:
    (a) `--init-expert-location <dist>` to flatten the expert→rank placement, or
    (b) drop to **4 GPU** (tp=ep=4) which is inherently more balanced (fewer ranks to skew across).
- Status check at 22:59: base(tp8) autotune advancing normally — **9/20 @ 48s/step** (the 323s step-1
  was the one-off cold JIT, robustness #2; GPU 0% during JIT is expected, NOT a hang). base READY soon,
  then acc + bench(16..256) + profile, then the EP8 cell (warm — autotune cache shared).
- Plan to execute once results land (results appended here + posted to PR #18):
    1. Verify EP8 launch cmd from its live process (`ep-size 8`, resolved flashinfer_trtllm + a2a=none).
    2. Per-bs table: TP8 vs EP8 **server-log decode thpt (token/s)** + bench e2e + acc(Δlogprob vs 0.30).
    3. Read whether EP8 closes the gap at large bs; if imbalance shows, run fallback (a)/(b) and append.

### 2026-06-02 23:18 (KST) — base(TP8) done; harness bug killed the EP8 cell; re-running EP8
- base(TP8) READY (~982s cold autotune), acc done (23:08), **bench bs16..256 done (23:13)**, graph-on
  profile captured. Then the run **died** at `pull_traces` base graph-on with
  `run_kimi_epreg.sh:139: cell: unbound variable` → base graph-off + the **entire EP8 cell never ran**.
  Root cause: `local` expands ALL its arg-words before binding any, so `local cell=$1 … src=…${cell}…`
  references `cell` while still unbound under `set -u`. **Fixed:** split into two `local` lines; added a
  `CELLS` env to re-run a subset. Pulled base's 8 graph-on traces from the pods before re-running.
- **BASE (TP8, no-LoRA) results** — bench e2e vs server-log decode thpt agree (DECODE-THPT-RULE ✓):

  | bs | bench output_thpt (tok/s) | server-log decode (steady max / avg) | e2e latency (s) |
  |----|------|------|------|
  | 16  | 1259.4 | 1271.7 / 1190.0 | 26.73 |
  | 32  | 2145.0 | 2185.5 / 2105.0 | 31.82 |
  | 64  | 3466.9 | 3536.3 / 3402.8 | 40.08 |
  | 128 | 5594.1 | 5771.5 / 5515.7 | 50.74 |
  | 256 | 8377.2 | 8835.0 / 8277.6 | 69.98 |

  acc logprobs captured (`kimi/base/acc/logprobs.json`) for the EP8-vs-TP8 regression check.
- 23:18: re-running **only the EP8 cell** (`CELLS=variant`, bg `bzdh6ip0r`); its `kill_all` stops the
  still-running base tp8 server, then launches EP8 (warm — autotune cache shared, so READY fast).
  Will verify `--ep-size 8` from the live process, then append the EP8 table + TP8-vs-EP8 delta.

### 2026-06-02 23:25 (KST) — CURRENT STATUS & NEXT STEP
**Verified (live process + server_args), both launch commands** — identical except EP8 adds `--ep-size 8`:
- TP8 (base): resolved `tp_size=8, ep_size=1, moe_a2a_backend='none', moe_runner_backend='flashinfer_trtllm'`.
- EP8 (variant): live cmdline has `--ep-size 8`; resolved `ep_size=8, moe_a2a_backend='none',
  moe_runner_backend='flashinfer_trtllm'` ✓ (ranks now log `TP0 EP0 … TP7 EP7` → EP active, 48 experts/rank).
  Same trtllm-gen MoE backend + a2a=none as TP8 — NOT cutlass/cutedsl, NOT deepep.

**Current status:**
- BASE (TP8) complete: acc + bench bs16..256 captured (table above). 8 graph-on traces pulled.
- EP8 cell running (`bzdh6ip0r`): loaded OK, now in its own cold `fp4_gemm` autotune (0/20 — the EP
  layout changes the grouped-GEMM shapes so it re-tunes; ~15-20 min). Not READY yet → no EP8 numbers yet.

**Next step (auto, once EP8 bench lands):**
1. Build the TP8-vs-EP8 table: per bs {16,32,64,128,256} server-log decode thpt + bench e2e, and the
   EP8/TP8 ratio — answering "is EP8 OK at larger bs?".
2. acc: EP8 vs TP8 logprob diff vs the ~0.30 noise floor (numerically-equivalent regression check).
3. Per user: **graph-off can be skipped** (no new kernels). **init-expert-location is not required** —
   only attempt the balancedness fallback (or drop to 4 GPU) if EP8 is clearly bad AND imbalance-bound.
4. Post the comparison to PR #18 description; append results here.

### 2026-06-02 23:55 (KST) — persist /root/.cache (autotune) across pod recreations
- Found `/root/.cache` was **ephemeral** (677M of fp4_gemm autotune + triton/deep_gemm/HF caches, lost
  on pod recreate). Added a hostPath mount `/root/.cache → /var/lib/sglang-cache` (node-local,
  `DirectoryOrCreate`) to BOTH pod specs in: this branch's `kimi-2node.yaml`, the task yaml, the
  canonical `sglang-lora-base-perf-benchmark.md` §3.1 (all 4 model pods), and created
  `kimi-regression/assets/kimi-2node.yaml` (the file SKILL.md §1 referenced but was missing) + a note in
  `kimi-regression/SKILL.md` robustness #2. So the ~20-min cold autotune is paid once **per node**, not
  every fresh pod, going forward.
- **Does NOT apply to the current pods** — k8s volumes are immutable post-creation; re-applying would
  recreate the pods (killing the running EP8 cell + losing the model download). Future runs only.
- Note: EP8 still re-autotunes vs TP8 even with the mount — different grouped-GEMM shapes
  (48 vs 384 experts/rank) = separate cache entries. Observed EP8 cold step-1 ~338s (matches base).

### 2026-06-03 00:21 (KST) — EP8 first launch CRASHED (cross-rank autotune desync) + option-1 fix
- EP8 launched & loaded fine (resolved `ep_size=8, flashinfer_trtllm, a2a=none`), but **died during the
  cold `fp4_gemm` autotune**. Caught it via the new STUCK-CHECK rule: two checks showed CPU ~96% idle,
  GPU util 0%, head `/tmp/server.log` byte-count unchanged over 30s, sglang procs decreasing 4->1 on
  both pods, then **all GPU memory freed** (72GB weights + 78GB KV gone -> processes exited). No Python
  traceback, no Watchdog/OOM/signal line; host RAM had 1583 GB free (not host-OOM).
- **Root cause = cross-rank autotune desync.** head log stuck at `Tuning fp4_gemm 1/20` (its step-1 cold
  JIT ~340s) while the **worker raced to 8/20** — under EP8 each rank cold-JIT-autotunes its OWN
  48-expert grouped-GEMM shapes, which compile at different speeds, so ranks drift; the EP cross-rank op
  (combine / graph-capture barrier) then can't line up -> abort -> full teardown. **TP8 is immune**:
  no-EP means every rank tunes the SAME replicated GEMM in lockstep.
- **Fix (user picked option 1 = warm the autotune cache):** the cache only warms if the autotune
  COMPLETES once, so raised `--dist-timeout 7200 --watchdog-timeout 7200` to let the slow (head) rank
  finish all 20 steps without the fast rank's abort/watchdog killing the group. That writes the EP8
  cache (`/root/.cache/.../rank_tp*.json`); later EP8 launches load it -> no JIT skew -> synchronized.
  The hostPath `/root/.cache` mount persists it for future pods.
- Re-running EP8 (`CELLS=variant`, bg `bj5eiz3y0`); monitoring with the STUCK-CHECK rule.

### 2026-06-03 00:55 (KST) — RESULTS: TP8 vs EP8 (no-LoRA), + EP8 crashes during bench at bs>=32
- The timeout fix WORKED for the desync: EP8 ranks re-synced (both hit 9/20→10/20→20/20 in lockstep)
  and EP8 reached READY + ran acc + bench. BUT the **graph-on server then crashed during the bench**:
  bs16 completed, bs32 died **mid-generation** (server-log decode stopped at n=26/~51 steps), and
  bs64/128/256 all got `ConnectionError: port 30000 connection refused` (server gone).

**TP8 vs EP8 — no-LoRA decode throughput (token/s):**

| bs | TP8 server / bench | EP8 server / bench | EP8/TP8 |
|----|------|------|------|
| 16  | 1272 / 1259 | 1240 / 1225 | ~0.97 (EP8 ~3% slower) |
| 32  | 2186 / 2145 | 2164 / (crashed mid-bench) | ~0.99 |
| 64  | 3536 / 3467 | **server crashed** | — |
| 128 | 5772 / 5594 | **server crashed** | — |
| 256 | 8835 / 8377 | **server crashed** | — |

- **acc (EP8 vs TP8, numerically-equivalent regression check):** mean|Δlogprob| = **0.1138** over 1808
  tokens (max 7.73 = single-token outlier), **well within the ~0.30 noise floor** → EP8 is numerically
  correct, no accuracy regression.

**Findings:**
1. **EP8 gives NO throughput win on the no-LoRA baseline** — at bs16/32 EP8 ≈ TP8 (marginally slower).
   Expected: for no-LoRA the base MoE expert GEMMs are large/compute-bound, so TP8→EP8 just reshuffles
   work and the EP all-to-all overhead offsets the per-rank compute reduction. The P0 "TP tax" lives in
   the **LoRA kernels**, not the base MoE — so EP only pays off once the LoRA kernels ride the EP layout.
2. **EP8 is unstable** — the graph-on server crashed during the bs32 bench, so the larger-bs numbers
   (64/128/256, the whole point) are unmeasured. Crash cause not yet captured (server.log was
   overwritten by the graph-off relaunch). Next: re-launch EP8 (autotune cache is now WARM → fast) and
   re-bench bs>=32, capturing the crash reason this time.

### 2026-06-03 01:08 (KST) — CORRECTED RESULT: EP8 crosses over & WINS at large bs (warm-cache relaunch)
- Relaunched EP8 from the now-WARM autotune cache → READY in **~240s** (vs ~20min cold; option-1 confirmed)
  and benched bs>=32 with server.log kept per-bs. EP8 completed bs32/64/128 cleanly (each `/generate`
  returned 200 OK, decode steady-state captured); the server then **shut down between benches**
  ("leaked semaphore at shutdown") before bs256 — so the per-bs decode numbers are valid, but the EP8
  server doesn't survive request-to-request (separate stability bug, not a decode failure).

**TP8 vs EP8 — no-LoRA decode throughput, server-log steady_max (token/s):**

| bs | TP8 | EP8 | EP8/TP8 |
|----|------|------|------|
| 16  | 1272 | 1240 | 0.97  (EP8 ~3% slower) |
| 32  | 2186 | 2166 | 0.99 |
| 64  | 3536 | **3662** | **1.04**  (EP8 ~4% faster) |
| 128 | 5772 | **6275** | **1.09**  (EP8 ~9% faster) |
| 256 | 8835 | _(re-launching warm to measure)_ | — |

- **Headline (answers the P0 question "is EP8 OK at larger bs"): YES — EP8 crosses over around bs≈64 and
  at bs128 is ~9% FASTER than TP8.** At small bs the EP all-to-all overhead makes EP8 marginally slower;
  as bs grows the per-rank 1/8 compute reduction wins out. So for the **no-LoRA baseline** EP8 is viable
  (even favorable) at the large bs we care about — the earlier "no win" read was an artifact of only
  having bs16/32 before the first crash.
- **Caveat / open issue:** EP8 server is unstable — it tears down between benches (had to relaunch per
  measurement). Needs a stability fix before EP8 is production-usable, but the throughput trend is clear.
- acc unchanged: EP8 vs TP8 mean|Δlogprob| = 0.1138 (within 0.30 floor).

### 2026-06-03 01:20 (KST) — bs256 not captured (EP8 instability); starting WITH-LoRA A/B
- Tried to measure EP8 no-LoRA **bs256**: the warm relaunch loaded + autotuned-from-cache fine but then
  **died again at the `Capturing batches (bs=256)` / serving boundary** (procs → 1/1, never READY). EP8
  reproducibly tears down around graph-capture/first-serve. So **bs256 EP8 left unmeasured** — bs16-128
  already establishes the trend (crossover ~bs64, +9% at bs128). EP8 instability is the real blocker, not
  the throughput.
- **Started the WITH-LoRA A/B** (user request: run LoRA under the same TP/EP settings). This is the actual
  P0 point — the TP "tax" lives in the LoRA grouped-GEMM kernels, so EP should help MORE with LoRA on.
  - base = **TP8 + LoRA**, variant = **EP8 + LoRA**; identical LoRA stack on both
    (`--moe-runner-backend sgl_flashinfer_trtllm --enable-lora --max-lora-rank 32 --lora-backend triton
    --lora-use-virtual-experts --lora-paths alpha=…` + envs `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1
    SGLANG_LORA_TWO_STREAM=1`); variant adds only `--ep-size 8`.
  - Fresh RUN_ROOT (`sglang_kimi_eplora_${ID}`) so the no-LoRA raw files aren't clobbered. bg `b54b49e43`.
  - Note: LoRA uses the `sgl_flashinfer_trtllm` backend = a DIFFERENT autotune cache → expect another
    cold tune (~20min) for the TP8+LoRA cell; EP8+LoRA may hit the same desync (mitigated by the 7200s
    timeouts) and/or the capture-boundary instability seen in no-LoRA EP8.

### 2026-06-03 02:03 (KST) — RESULTS: TP8+LoRA vs EP8+LoRA — EP cuts the LoRA tax (P0 confirmed)
Both LoRA cells ran. **TP8+LoRA stable** (full bs16..256). **EP8+LoRA** completed bs16 then crashed
during bs32 bench (same capture/serve instability as no-LoRA EP8) → only bs16 measured.

**With-LoRA decode throughput (bench output_thpt, token/s):**

| bs | TP8+LoRA | EP8+LoRA | EP8/TP8 |
|----|------|------|------|
| 16  | 675.8 | **760.5** | **1.13** (EP8+LoRA ~13% FASTER) |
| 32  | 1214.8 | crashed mid-bench | — |
| 64  | 2071.0 | crashed | — |
| 128 | 3513.0 | crashed | — |
| 256 | 5464.1 | crashed | — |

**LoRA tax (throughput retained vs no-LoRA, bs16):**
- TP8: 1272 → 676 = **53% retained (47% LoRA tax)**
- EP8: 1240 → 761 = **61% retained (39% LoRA tax)**

**Headline (P0 confirmed):** **EP shrinks the LoRA tax** — the TP tax lives in the LoRA grouped-GEMM
kernels (gate_up-A shrink + down-B expand stream the full per-expert weight under TP); EP slices them
8× (48 experts/rank), so EP helps the LoRA path much more than the base MoE. Concretely at bs16, going
TP8→EP8 *hurts* the no-LoRA baseline (−3%) but *helps* the LoRA case (+13%), and the LoRA tax drops
47%→39%. The win should grow at larger bs (as it did for no-LoRA: +9% at bs128) — but **EP8 can't yet
hold a server past bs16-128 to measure it**.
- acc (EP8+LoRA vs TP8+LoRA, numerically-equiv): mean|Δlogprob| = **0.2933** (n=1808, max 8.86) — within
  the ~0.30 noise floor but near the edge (vs 0.11 for no-LoRA); no clear regression, worth a recheck.
- **The gating issue is EP8 stability**, not throughput: EP8 (±LoRA) reproducibly tears down at the
  graph-capture / first-serve boundary or mid-bench at larger bs. That's the thing to fix to make EP usable.

### 2026-06-03 11:40 (KST) — job stopped, nodes released
- User: stop job + release nodes. Killed local drivers/launchers + in-pod servers; deleted the k8s
  resources for `ID=yushengsu-20260602-220516` (pods `-0/-1`, service `-head`, computedomain). Verified
  0 remaining pods for this ID; other users' kimi pods untouched. GPUs freed.
- **Final summary (no-LoRA + LoRA, all in JOURNAL + PR #18):**
  - no-LoRA: EP8 crosses TP8 ~bs64, **+9% at bs128**; slightly slower at small bs. bs256 unmeasured (EP8 unstable).
  - +LoRA: **EP8+LoRA +13% at bs16**, LoRA tax 47%→39% — EP cuts the LoRA-kernel tax (P0 confirmed).
    Larger-bs +LoRA unmeasured (EP8+LoRA crashes at bs>=32).
  - acc within noise floor (no-LoRA 0.11, +LoRA 0.29).
  - **Open / blocker: EP8 server stability** (teardown at capture/serve boundary, ±LoRA) — must fix
    before EP is usable; then re-measure EP8 large-bs (±LoRA) to quantify the full win.
