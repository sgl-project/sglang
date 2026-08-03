# ROCm DWDP current status

Snapshot date: 2026-08-03

This document records the state immediately before the first SGLang and Aiter
commits for the ROCm DWDP port.

## Source state

- SGLang branch: `feature/dwdp-rocm`
- SGLang base: `131bd51b01`
- Aiter branch: `feature/dwdp-multib-rocm`
- Aiter base: `702aacd`
- The SGLang IPC backend depends on the matching Aiter multi-B changes.
- The implementation will be proposed as two new linked PRs. The historical
  SGLang and Aiter DWDP PRs are not force-updated because their bases and
  interfaces are obsolete.

## Validated functionality

- HIP VMM primitives, POSIX-FD exchange, mapping, teardown, and composite
  manager operation pass at small scale. Job `24043` reported 8 manager and
  primitive tests plus 2 raw HSA copy tests passing.
- HIP IPC rank-ordered multi-B prefetch passes 2-, 4-, and 8-rank synthetic
  tests (`23964`, `23967`, and `23972`).
- IPC cleanup followed by setup and prefetch re-initialization passes in job
  `24044`.
- Aiter gfx950 multi-B supports MXFP4, A8W4, MXFP8, and conventional FP8 with
  FP32 128x128 block scales. Job `24031` reported 51 Aiter tests and 28 SGLang
  integration/lifecycle tests passing.
- HSA copy reached 169.2 GiB/s for three concurrent 256 MiB peer copies versus
  167.9 GiB/s for per-peer HIP streams.
- DeepSeek-R1 MXFP4 IPC standalone accuracy passed in `24017`; IPC/NIXL 4P+4D
  PD accuracy passed in `24016`. Both deterministic completions contained
  `Paris`.
- The optimized DeepSeek-R1 long-prefill configuration uses a global chunk of
  262144 at DWDP8, giving 32768 tokens per rank. IPC/DEP throughput was
  0.579x, 0.529x, 1.209x, and 1.637x at 4K, 8K, 16K, and 32K input lengths.
  IPC is therefore useful for long prefill, not short-prefill traffic.
- Slurm submission rejects non-idle, occupied, or VRAM-unbalanced nodes before
  `sbatch`, repeats the check after allocation, and verifies cleanup after the
  container exits.

## Production guidance

- ROCm `auto` selects IPC.
- IPC is the production candidate for gfx950 long-prefill workloads.
- VMM remains experimental. Small-scale mapping is functional after applying
  access permissions to every composite remap, but production-size 58-60-layer
  configurations still hit ROCm generic-allocation/access limits.
- VMM teardown releases resources but does not support setup after cleanup.
- The known-bad node `smci355-ccs-aus-n08-21` is always excluded. No GPU reset
  or unrelated process cleanup is part of this workflow.

## Known blockers and incomplete validation

- MiMo-V2.5 conventional FP8 kernels pass synthetic tests, but the full model
  job `24033` was SIGKILLed while loading checkpoint shard 28/34 with one GPU
  near 267.5 GiB, before DWDP backend setup.
- No GPT-OSS checkpoint is mounted in the Slurm `/models` volume, so its model
  smoke and PD validation are not complete.
- DeepSeek task-level GSM8K and deterministic-logit parity against DEP/EP are
  still pending; the current model-level checks are deterministic completion
  checks.
- HSA raw copies and IPC end-to-end use are validated, but production-scale
  HIP VMM plus HSA overlap remains unvalidated.

## Detailed evidence

See `ROCM_BACKEND_RESULTS.md` for job IDs, node assignments, individual
backend results, failure analysis, and benchmark details.
