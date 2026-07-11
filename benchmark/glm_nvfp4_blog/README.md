# GLM NVFP4 on SGLang — blog figure reproduction

Everything needed to reproduce the two figures from the GLM NVFP4 GB300/B300
blog post, using only **SGLang** (this branch, based on `release/v0.5.15`) and
**[evalscope](https://github.com/modelscope/evalscope)** (pinned by commit) as
the benchmark client.

- `expected_main_figure.png` — the main Pareto figure (2×2: GB300 TP4/TEP4,
  B300 TP8/TEP8; day-0 vs v0.5.15 vs GLM-5.1)
- `isl_ablation/expected_isl_ablation.png` — the c=1 ISL-ablation bar chart

Workload: OpenHands multi-turn agentic replay — mean input ≈ 80k tokens/request,
220 output tokens/turn, 13 turns/conversation, ~92% aggregate prefix-cache hit
rate, real EAGLE speculative acceptance (nothing simulated).

## Hardware / software

- Main figure: one 4×GB300 node (top row) and/or one 8×B300 node (bottom row).
  Rows are independent — reproduce either.
- ISL ablation: the 4×GB300 node.
- SGLang installed from **this branch** (standard install; the day-0 curves are
  verified against `sgl-kernel 0.4.4`, `flashinfer 0.6.12`, `torch 2.11`).
- ~470 GB of HF cache per model (`nvidia/GLM-5.2-NVFP4`, `nvidia/GLM-5.1-NVFP4`).
  If your HF cache is on slow network storage, copy the snapshot to local disk
  and point `--model-path` at it (add `--served-model-name nvidia/GLM-5.2-NVFP4`).

## One-time setup

```bash
# 1. evalscope, pinned, without letting its dep tree downgrade the image
#    (see evalscope-deps/README.md for why)
evalscope-deps/scripts/install_evalscope_deps.sh
PIP_NO_DEPS=1 pip install "evalscope[all] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9"

# 2. the day-0 SGLang snapshot (needed only for the day-0 curves); run from
#    the repo root — the fetch makes the commit available even on
#    single-branch clones (it lives on the public glm-opt branch)
git fetch origin glm-opt
git worktree add ../sglang-day0 22dce572045c277ce46f1a287c4be1112b214368
export DAY0_SGLANG=$(cd ../sglang-day0 && pwd)

# 3. (fresh machine) JIT-warm once so the first measured c=1 point doesn't pay
#    one-time kernel compilation: run any one server+client pair below and
#    discard the result. Compiled-kernel caches persist on disk after that.
```

## Main figure

One curve = one server script + one client invocation. Start the server, wait
for it to load (the client also waits on `/health`), run the client, stop the
server, repeat. ~15–30 min per (server, client) pair.

| platform | curve | server script | client invocation |
|---|---|---|---|
| GB300 | GLM-5.2 v0.5.15 | `gb300/server_glm52_v0515_tp4.sh` / `_tep4.sh` | `./run_client.sh nvidia/GLM-5.2-NVFP4 results/gb300/glm52_v0515 tp4` (resp. `tep4`) |
| GB300 | GLM-5.1 v0.5.15 | `gb300/server_glm51_v0515_tp4.sh` / `_tep4.sh` | `./run_client.sh nvidia/GLM-5.1-NVFP4 results/gb300/glm51_v0515 tp4` (resp. `tep4`) |
| GB300 | GLM-5.2 day-0 | `gb300/server_day0_tp4.sh` / `_tep4.sh` (needs `DAY0_SGLANG`, see setup) | `./run_client.sh nvidia/GLM-5.2-NVFP4 results/gb300/day0 tp4` (resp. `tep4`) |
| B300 | GLM-5.2 v0.5.15 | `b300/server_glm52_v0515_tp8.sh` / `_tep8.sh` | `./run_client.sh nvidia/GLM-5.2-NVFP4 results/b300/glm52_v0515 tp8` (resp. `tep8`) |
| B300 | GLM-5.1 v0.5.15 | `b300/server_glm51_v0515_tp8.sh` / `_tep8.sh` | `./run_client.sh nvidia/GLM-5.1-NVFP4 results/b300/glm51_v0515 tp8` (resp. `tep8`) |
| B300 | GLM-5.2 day-0 | `b300/server_day0_tp8.sh` / `_tep8.sh` (needs `DAY0_SGLANG`, see setup) | `./run_client.sh nvidia/GLM-5.2-NVFP4 results/b300/day0 tp8` (resp. `tep8`) |

The client builds the dataset on first use (~10–20 min per model tokenizer,
cached under `datasets/`), then runs concurrency 1→8 in a single evalscope
invocation (evalscope rotates dataset offsets so every step replays fresh
conversations).

Then:

```bash
python3 plot_main_figure.py        # -> main_figure.png (plots whatever curves exist)
```

## ISL ablation (GB300 only)

```bash
cd isl_ablation
./run_isl_client.sh v0515                            # ~2-3 h (dataset builds dominate, cached after)
./run_isl_client.sh day0                             # needs DAY0_SGLANG exported (see setup)
python3 plot_isl_figure.py         # -> isl_ablation.png
```

The driver boots its own server per rung (context length is a boot-time
setting): 80K→1M mean ISL, scaling only the dataset first-turn budget and
`--context-length`; one c=1 evalscope step per rung.

## What to expect

Run-to-run variance on these curves is ~±2–4% per point (widest at c=1);
your curves should match the expected figures in shape and ordering, not to
the pixel. The 92% cache-hit and ~5.0 acceptance-length invariants are printed
in each `benchmark_summary.json` / server log — if those match, the workload
reproduced faithfully.
