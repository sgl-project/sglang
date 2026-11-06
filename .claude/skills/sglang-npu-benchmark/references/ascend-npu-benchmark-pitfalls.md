# Ascend NPU Benchmark Pitfalls — Field Catalog

Every item below was observed in real Atlas 800T A3 sweeps (Sept 2026, CANN 9.0.0 /
torch_npu 2.10 / SGLang 0.5.x-dev). Symptom -> root cause -> fix.

## P1. Server flag renamed upstream: `--cuda-graph-bs`

- **Symptom**: server rejects `--cuda-graph-bs`; configs written from older docs fail.
- **Cause**: SGLang 0.5.x split it into `--cuda-graph-bs-decode` / `--cuda-graph-bs-prefill`.
  NPU-facing docs may still carry the old spelling.
- **Fix**: validate every config flag against the installed sglang source before sweeping
  (a text-scan test that greps flags in the sglang tree is cheap and catches the whole class).
  Guard all three spellings if you support older versions.

## P2. CUDA-graph capture list smaller than grid concurrency

- **Symptom**: high-concurrency cells run ~10x slower; server log shows `npu graph: False`
  (eager decode). One 35B cell took ~26 min instead of minutes.
- **Cause**: `--cuda-graph-bs-decode [1,2,4,8]` cannot capture a batch of 32/64; runtime
  falls back to eager mode.
- **Fix**: the largest capture entry must be >= the grid's maximum concurrency, e.g.
  `[1,2,4,8,16,32,64]`. Encode this as a config validation rule (axis-level check), not a
  human convention. Note capture-count limits: too many graph sizes can exhaust NPU capture
  resources on some stacks — keep the list bounded.

## P3. KV pool too small for the concurrency axis

- **Symptom**: `KV cache pool is full. Retract requests` in server log; throughput looks
  plausible but is KV-limited; cell duration doubles.
- **Cause**: `--max-total-tokens 32768` while conc=32 x (isl+osl 2048) needs 65536,
  conc=64 needs 131072.
- **Fix**: size the KV pool to cover max concurrency x (isl + osl), or drop the flag and let
  mem-fraction decide. Never mix KV-limited cells into a ranking with unrestricted cells.

## P4. `statistics.stdev` crash on `request_rate: Infinity`

- **Symptom**: all cells finish, then the report rendering crashes with
  `AttributeError: 'float' object has no attribute 'numerator'` (Python 3.11).
- **Cause**: bench output records `request_rate: Infinity` (legal JSON number) when the
  client sends unbounded; aggregation calls `stdev` over values including `inf`.
- **Fix**: filter non-finite values (`math.isfinite`) before aggregation. Include
  Infinity/NaN in unit-test metrics fixtures — JSON-legal but statistics-hostile.

## P5. Offline hosts: `random` dataset still touches the network

- **Symptom**: bench startup hangs or fails on a NAT-ed host even with "no download
  needed" dataset.
- **Cause**: `random` synthesizes requests but still loads a tokenizer via HF hub.
- **Fix**: on offline hosts use `random-ids` (zero-network, synthesizes token ids directly)
  and point the tokenizer/model at local weights. Pre-download the gsm8k eval set;
  `raw.githubusercontent.com` may be unreachable or throttled from NPU clusters.

## P6. Orphan engine workers hold HBM across cells

- **Symptom**: next cell OOMs though the previous server was "killed"; `npu-smi` shows
  residual used HBM (observed 5.3-8.4 GB residue at conc=64).
- **Cause**: engine worker processes survive the parent server kill.
- **Fix**: per-cell HBM release gate — poll `npu-smi` after server shutdown, sweep stragglers
  (`pkill -9 -f` on engine worker process titles), re-verify against the idle baseline.
  This gate is what makes unattended overnight sweeps safe.

## P7. TTFT SLA at saturated concurrency

- **Symptom**: p99 TTFT at conc>=32 always exceeds a small-N TTFT threshold (observed
  2.5-6 s at conc 32-64 vs 2000 ms gate); looks like a failure storm but is expected.
- **Cause**: at saturation, TTFT includes prefill queueing; the SLA measures the queue,
  not the service.
- **Fix**: gate TTFT only on unsaturated cells; at saturation gate TPOT / throughput.
  Alternatively use concurrency-tiered thresholds. Report saturated cells as-is with the
  explanation — do not drop or re-scale them.

## P8. NEXTN / MTP speculative decoding fails to start on NPU

- **Symptom**: base model healthy; with `--speculative-algorithm NEXTN` the server dies
  during startup with a Triton compile error:
  `ub overflow, requires 2097152 bits while 1572864 bits available`
  in `sgl_kernel_npu/mamba/mamba_state_update_triton.py` (MTP draft path).
- **Cause**: sgl-kernel-npu Triton kernel tiling exceeds UB capacity on the draft path.
- **Fix**: none upstream yet — record the cell in the compatibility matrix with the log
  tail, keep the base-model data, and report the kernel failure upstream
  (same UB-overflow family as sgl-project/sglang#25330). Verify the draft path separately
  from the base model so one broken feature does not sink the whole sweep.

## P9. Process cleanup self-kill (shell pattern matching itself)

- **Symptom**: the cleanup command kills its own shell (SSH exit 255) or kills the sweep
  driver, which then relaunches the next cell with stale config.
- **Cause**: `pkill -f "<pattern>"` matches the calling shell's own command line when the
  pattern appears literally in it (grep pipelines are notorious). Killing benchmark
  servers but not the sweep driver lets the driver advance the grid with old settings.
- **Fix**: put patterns in a script file and run that; use a character class
  (`sglang[.]launch_server`) so the pattern text never matches itself; always kill the
  sweep driver first, then servers.

## P10. Report table separators (GFM rendering)

- **Symptom**: the ranking report renders as plain text on GitHub.
- **Cause**: separator row column count differs from the header row (off-by-two after
  adding columns).
- **Fix**: compute separator columns from the header; add a unit test asserting every table's
  separator matches its header. The ranking report is the deliverable — it must render.

## Provenance checklist (record per sweep, not per belief)

- git sha of the sglang tree (env-var injectable when running from an exported tree)
- CANN version and layout (`cann-9.0.0/` vs `ascend-toolkit/latest` differ across images)
- torch / torch_npu versions; sgl-kernel-npu / deep_ep / attentions wheel versions
- driver version (`npu-smi` banner) and full NPU topology output
- complete server args per cell (they are the config that produced the numbers)
- environment allowlist (PATH, LD_LIBRARY_PATH, PYTHONPATH — chroot setups mutate them)
