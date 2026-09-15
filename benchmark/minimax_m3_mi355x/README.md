# MiniMax-M3 on 4x MI350X/MI355X: AgentX reproduction

```bash
git clone -b M3-perf https://github.com/kevin-mii/sglang && cd sglang/benchmark/minimax_m3_mi355x
bash reproduce.sh real      # recommended config: AIPerf inferencex-agentx-mvp at c=1 8 24 32, 3600 s each, summary per point
bash reproduce.sh lossy     # ATOM-parity performance-only config (forced acceptance; outputs are not the model's)
```

Run inside the sglang ROCm 7.2.4 container for gfx950 (`docker/rocm.Dockerfile`, `GPU_ARCH=gfx950-rocm724`) on a node with 4 free
GPUs. The first run installs sglang from this checkout, aiter 4ad99832 with the swizzle fix, the tuned MoE rows, both models
and the SemiAnalysis aiperf fork under `M3_WORK` (default `/scratch` if writable, else `~/m3-agentx`; needs ~300 GB).
Optional: `M3_WORK=/data GPUS=4,5,6,7 PORT=30001 DURATION=1800`, concurrencies as arguments (`lossy 8 24`), `CHECK_ONLY=1`,
`setup` / `serve real|lossy` / `stop` subcommands. Results: `$M3_WORK/results/aiperf_<mode>_c<N>/`; server logs: `$M3_WORK/logs/`.

Results, config rationale and the optimization catalogue: `OPTIMIZATIONS.md`. Every flag and env var of both server configs is in
the `serve` function of `reproduce.sh`; ATOM's client environment and flags are in `bench`. `tools/` holds the development helpers.
