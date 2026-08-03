# ROCm DWDP backend validation

All GPU work is submitted through Slurm with
`smci355-ccs-aus-n08-21` explicitly excluded.

## PoC results

- HIP IPC handle export/open and peer copy:
  - Slurm job `23950`
  - node `smci355-ccs-aus-n08-25`
  - result: `1 passed`
- HIP VMM capability, allocation granularity, single-process mapping,
  composite mapping, repeated teardown, and cross-process POSIX-FD import:
  - Slurm jobs `23954`, `23965`, `23985` (4-rank manager),
    `23986` (8-rank manager)
  - node `smci355-ccs-aus-n08-29`
  - result: primitive and full-manager suites passed (`8 passed` in job `23965`)
- HSA pair-specific engine copy with BDF-mapped HIP/HSA agents:
  - Slurm job `23965`
  - node `smci355-ccs-aus-n08-29`
  - result: two-peer-copy tests passed (`2 passed`)
- Three-peer 256 MiB copy benchmark:
  - Slurm job `23984`
  - selected engine masks: GPU1 `0x4000`, GPU2 `0x0004`, GPU3 `0x1000`
  - HSA aggregate: `169.2 GiB/s`; per-peer HIP streams: `167.9 GiB/s`
  - HSA/HIP speed ratio for this transfer size: `1.008x`
- HIP IPC two-slot prefetch with fused shared final partition:
  - Slurm jobs `23964` (2 ranks), `23967` (4 ranks), `23972` (8 ranks)
  - node `smci355-ccs-aus-n08-25`
  - result: all configurations passed
- Aiter gfx950 multi-B contract, boundary routing, MXFP4 stage1/stage2,
  fused-shared uneven partitions, bias, and no-combine:
  - Slurm job `23970`
  - node `smci355-ccs-aus-n08-29`
  - result: `34 passed`
- SGLang DWDP layout, tensor schema, Aiter runner injection, import guard, and
  server-argument tests:
  - Slurm job `23977`
  - node `smci355-ccs-aus-n08-25`
  - result: `14 passed`
- DeepSeek-R1 MXFP4 IPC/multi-B standalone model smoke:
  - Slurm job `23982`
  - node `smci355-ccs-aus-n08-29`
  - result: server became healthy and generated deterministic text beginning
    with `Paris`
- DeepSeek-R1 MXFP4 IPC PD accuracy with NIXL:
  - Slurm job `24016`
  - node `smci355-ccs-aus-n09-21`
  - topology: 4-GPU DWDP prefill plus 4-GPU DEP decode, graph capture disabled
  - result: deterministic completion contained `Paris`; `1 passed` in
    `150.14s`
  - container prerequisites: `memlock=unlimited`, `NIXL_LOG_LEVEL=ERROR`,
    `UCX_LOG_LEVEL=error`
- DeepSeek-R1 MXFP4 IPC standalone accuracy:
  - Slurm job `24017`
  - node `smci355-ccs-aus-n09-21`
  - result: deterministic completion contained `Paris`; `1 passed` in
    `188.63s`
  - postflight: 85-89 GiB remained immediately after server exit and returned
    to 0.30 GiB per GPU on the second check 10 seconds later

## Slurm node hygiene

- Submission now requires Slurm state `idle`, eight readable VRAM counters,
  at most 4 GiB used on any GPU, and at most 2 GiB skew. The allocated job
  repeats the check before launching the container.
- `smci355-ccs-aus-n09-25` was rejected with 27-40 GiB used on GPU0-1;
  `smci355-ccs-aus-n08-29` was rejected with 153-155 GiB used on GPU0-3.
- Standalone job `24013` functionally passed, but its node retained
  76-78 GiB on GPU0-3 after exit. Postflight checks now wait up to 60 seconds
  and fail the job if VRAM remains allocated. No process cleanup or GPU reset
  was attempted.
- IPC cleanup followed by setup/prefetch re-initialization passed in job
  `24044`; postflight returned all GPUs to 0.30 GiB.
- `smci355-ccs-aus-n08-21` remains unconditionally excluded.

## End-to-end performance

Configuration: DeepSeek-R1 MXFP4, 8 GPUs, graph capture disabled, 256 random
prompts, concurrency 128, one output token, radix cache disabled.

- DEP baseline, Slurm job `24015`, node `smci355-ccs-aus-n08-33`:
  - ISL 4096: `52,544.67` input tok/s
  - ISL 8192: `58,295.12` input tok/s
  - ISL 16384: `49,473.02` input tok/s
  - ISL 32768: `39,948.81` input tok/s
- The sequential IPC launch in job `24015` encountered an internal-port
  collision with the just-stopped DEP server before loading weights. The node
  was clean after exit, so this is not an IPC correctness or memory result.
- Isolated IPC with the default global chunk 32768 (4096 per DP8 rank):
  - same-node job `24019`, node `smci355-ccs-aus-n08-33`
  - ISL 4096: `17,839.62` input tok/s, `0.340x` DEP
  - ISL 8192: `18,654.92` input tok/s, `0.320x` DEP
  - ISL 16384: `19,029.53` input tok/s, `0.385x` DEP
  - ISL 32768 at 256 prompts/concurrency 128 exceeded the available request
    capacity before producing a result
- Optimized long-prefill chunk, job `24020`, node
  `smci355-ccs-aus-n09-21`: global chunk 262144 (32768 per DP8 rank), 64
  prompts, concurrency 32, ISL 32768:
  - DEP: `35,258.01` input tok/s
  - IPC: `57,706.92` input tok/s
  - IPC/DEP: `1.637x`
  - both backends returned to 0.30 GiB used per GPU after the 10-second
    postflight grace period
- The matching optimized-chunk short/mid-context matrix, job `24021` on the
  same node:
  - ISL 4096: DEP `33,426.98`, IPC `19,340.84` input tok/s (`0.579x`)
  - ISL 8192: DEP `55,856.05`, IPC `29,565.11` input tok/s (`0.529x`)
  - ISL 16384: DEP `52,257.34`, IPC `63,183.95` input tok/s (`1.209x`)
- IPC DWDP therefore crosses DEP between 8K and 16K for this configuration
  and exceeds the `1.05x` long-context target at 16K and 32K. It should not be
  selected for short-prefill traffic based on these results.

## Open validation

- HIP VMM passes primitive and small 2/4/8-rank manager tests, but
  production-sized 58-60 layer mappings hit ROCm generic-allocation/access
  limits (`hipMemSetAccess: hipErrorInvalidValue`). IPC is therefore the ROCm
  `auto` default; VMM remains experimental.
  Job `24043` additionally verified the composite-remap access fix
  (`8 passed` manager/primitive plus `2 passed` HSA raw copy): every remapped
  local handle now receives `set_access`; postflight VRAM was clean.
- Conventional FP8 multi-B with FP32 128x128 block scales passed the gfx950
  Aiter suite in job `24031` (`51 passed`), together with `28` SGLang
  integration/lifecycle tests. MiMo-V2.5 job `24033` reached checkpoint shard
  28/34, then was SIGKILLed with one GPU at about 267.5 GiB before DWDP backend
  setup; full-model validation remains blocked by loader/VRAM capacity.
- No GPT-OSS checkpoint is mounted in the Slurm container's `/models` volume,
  so its fixed-prompt and PD model validation could not be run.
- HIP/HSA tracing of overlap remains future work; the completed copy-engine
  microbenchmark and end-to-end matrix are the current performance evidence.
- VMM is excluded from production-size comparison because of the
  generic-allocation limit above.
