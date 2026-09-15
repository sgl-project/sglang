# MiniMax-M3 on 4x MI350X/MI355X: AgentX reproduction

One script, three data files. Results, config rationale and the optimization catalogue are in `OPTIMIZATIONS.md`.

```bash
export M3_WORK=/scratch            # any writable directory with ~300 GB free (default: /scratch if writable, else ~/m3-agentx)
bash reproduce.sh setup            # sglang (this checkout), aiter 4ad99832 + swizzle fix, tuned MoE rows, both models, SemiAnalysis aiperf fork
bash reproduce.sh real             # recommended config: AIPerf inferencex-agentx-mvp at c=1 8 24 32, 3600 s each, summary per point
bash reproduce.sh lossy 8 24       # ATOM-parity performance-only config (forced acceptance; outputs are not the model's), chosen concurrencies
```

`GPUS=4,5,6,7 PORT=30001 DURATION=1800` override the defaults; `CHECK_ONLY=1` runs only the preflight; `serve real|lossy` and `stop`
manage the server alone. Results land in `$M3_WORK/results/aiperf_<mode>_c<N>/`, server logs in `$M3_WORK/logs/`.

- `reproduce.sh`: setup, server launch (every flag and env var of both configs is in its `serve` function), client, summary.
- `tuned_fmoe_m3_gfx950.csv`, `tuned_a8w8_bpreshuffle_m3_gfx950.csv`: tuned aiter MoE and PTPC-FP8 GEMM rows for this shape.
- `aiter_flydsl_xcd_swizzle_fix.patch`: aiter fix applied by `setup`.
- `tools/`: development helpers used during the tuning work (per-piece launch/config scripts, steady-state and needle probes, chart).
