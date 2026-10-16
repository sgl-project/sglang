# Muse MoE FlyDSL performance plan

## Attribution and current evidence

`GEMM2_BLOCK_N=256` applies only to MXFP4 when `128 < M <= 1024`. Since decode
has `M == concurrency`, concurrency 4--128 is unchanged from `a29522f`. M=128
at 234.9 versus 235.1 us is noise. The attributable microbenchmark changes are
M=256 264.3 -> 259.3 us (-1.9%), M=512 321.9 -> 306.0 (-4.9%), and M=1024
450.9 -> 411.3 (-8.8%). These also represent chunked-prefill batch shapes.

The cancelled end-to-end run stopped in the first discarded concurrency-4
point and left a zero-valued JSON, so it supplies no result. Keep result sets
versioned separately: the pre-pipeline factorial data and the later pipeline
data are different implementations. Full-stack D-versus-A changes also cannot
be attributed to this MoE selector.

One later recorded pipeline end-to-end A/B shows different priorities by format
(the saved JSON does not establish the currently installed kernel revision):
MXFP4 improves throughput versus Triton by +12.0/+3.4/+2.9% at concurrency
4/8/16, while BF16 regresses -3.4/-4.0/-2.7%. Therefore BF16 low concurrency
is the first production target; MXFP4 needs a route-distribution stability gate
so an optimization does not give back its present end-to-end win.

## 2026-09-10 matched BF16 A/B

Using the current installed aiter source and a fresh private FlyDSL cache,
Triton and FlyDSL were served separately with the same Plan B attention,
TP8/EP8, IL8192/OL1024, and matching warmup/request schedules. The BF16
candidate (BN64, GEMM1 K-unroll 2, GEMM2 K-unroll 4) was enabled only for
`8 <= M <= 16`.

| concurrency | Triton tok/s | FlyDSL tok/s | throughput delta | Triton TPOT | FlyDSL TPOT |
|---:|---:|---:|---:|---:|---:|
| 4 | 212.30 | 208.96 | -1.6% | 17.48 ms | 17.66 ms |
| 8 | 321.88 | 337.93 | +5.0% | 23.43 ms | 21.95 ms |
| 16 | 474.26 | 496.86 | +4.8% | 32.12 ms | 30.09 ms |

The M=4 candidate was rejected because it remained below Triton. The M=8/16
results are one matched run per point and need repeat runs before shipping.
MXFP4 was not changed by this selector branch and passed a fresh operator sweep
at M=4/8/16/128/256 with randomized E8M0 scales.

## 2026-09-10 final split-K result

The M=4 gap came from Triton's split-K=4 path rather than its ordinary grouped
GEMM. Disabling Triton split-K moved its one-local-route operator time from about
96 to 105.6 us, behind FlyDSL at 99.4 us. The effect grows on dense EP ranks:
with 8 local routes Triton split-K measured 216.7 us versus 265.9 us without it.

FlyDSL now splits each BF16 M<=4 GEMM into four FP32 K partials. A FlyDSL kernel
reduces gate/up before SiLU, and the existing FlyDSL finalize reduces the down
partials. Split-K changes the best GEMM shape: BN16 and K-unroll 4 create
one-wave workgroups and enough N/K parallelism to fill the device. The finalize
uses separate LDS sum cells for each routing slot, removing the second barrier
that was previously needed before reusing four shared cells. This reduced its
profiled split-K time from 20.0 to 15.9 us.

The strict operator sweep passed against Triton at BF16 M=4 for 1/2/4/8 local
routes. FlyDSL measured 87.58/107.91/141.36/214.52 us versus Triton
95.89/113.51/146.55/217.11 us. Random-scale MXFP4 correctness also passed at
M=4/8/16 for the same four routing distributions.

Fresh matched end-to-end runs used Plan B attention, TP8/EP8, IL8192/OL1024,
and four warmups plus four measured requests per concurrency slot:

| dtype | conc | Triton tok/s | FlyDSL tok/s | throughput | Triton TPOT | FlyDSL TPOT |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 4 | 227.90 | 229.33 | +0.63% | 16.39 ms | 16.16 ms |
| BF16 | 8 | 345.32 | 350.90 | +1.62% | 21.63 ms | 21.43 ms |
| BF16 | 16 | 478.71 | 511.28 | +6.80% | 30.92 ms | 29.58 ms |
| MXFP4 | 4 | 271.93 | 305.98 | +12.52% | 13.53 ms | 12.09 ms |
| MXFP4 | 8 | 470.65 | 483.56 | +2.74% | 15.71 ms | 15.38 ms |
| MXFP4 | 16 | 693.96 | 713.61 | +2.83% | 21.20 ms | 20.89 ms |

All six points completed their full request counts. Exact JSON paths, TTFT/E2E,
source hashes and percentage calculations are recorded in
`docs/muse_moe_flydsl_20260910_results.json`.

## Experiments that preserve numerical behavior

For MXFP4, test direct packed global-to-LDS staging. The old LDS path loads via
VGPR and then stores to LDS, adding register pressure and DS traffic. Reuse the
BF16 path's proven `BufferCopyLDS128b` primitive for packed bytes, then load the
lane-specific dword from LDS and retain the current E8M0 decode, BF16 MFMA and
accumulation order. Decouple LDS BK from register unroll: at BM16, double-buffer
BK64/BK128 uses 21/42 KiB for both GEMM1 BN128 and GEMM2 BN256; BK512 is
168 KiB and exceeds a 160 KiB/CU LDS budget by itself.

For BF16, do not assume that the existing double-buffer pipeline helps decode.
The recorded pipeline probe ran no tests and printed no timing rows; VGPR stayed
166, which establishes neither a speedup nor a regression. Build a stage-specific
harness that varies GEMM1 and GEMM2 independently: register transpose versus
direct LDS, BN32/64/128, 4 versus 8 waves, and one versus two LDS stages. Start
with the current low-M selector as control and preserve BF16 MFMA and accumulation
order. The first goal is to explain the end-to-end regression by stage and worst
EP rank, not to promise a bandwidth or occupancy target.

BF16 has a much larger LDS cost: the same BM16/BN128 gate-up or BN256 down
geometry consumes 68/136/544 KiB at double-buffered BK64/128/512. Its feasible
LDS sweep therefore needs BK16/BK32 or narrower BN, and must include the register
path as a control. Resource arithmetic alone cannot predict occupancy because
VGPRs and allocation granularity can become the binding limit.

The checked-in source proves direct BF16 global-to-LDS API usage, but does not
prove the packed MXFP4 copy tiling and byte offsets. A production kernel change
would be speculative until an isolated compile shows correct address spaces and
ISA. The implementation should remain an experiment until all gates pass.

There are three distinct versions to preserve during comparison: patch 04, the
installed kernel, and each candidate. The installed GEMM has an extra BF16
register software pipeline (`pipelined = (mxfp4_reg or bf16_reg) ...`) that is
absent from patch 04 (`mxfp4_reg` only); their source hashes differ while the
finalize source matches. Do not synchronize or overwrite the live install.
Compile named copies into private caches and record each source hash.

CPU-safe preparation and GPU acceptance gates:

1. Use `tools/validate_muse_moe_flydsl_plan.py` to syntax-check without imports,
   verify MXFP4 and BF16 unchanged boundaries against `a29522f`, check the
   intentional BF16 M=8/16 override and wide MXFP4 override, and calculate stage
   footprints.
   `--emit-manifest` also emits argv arrays for a CPU-generated, stage-specific
   BF16 control/BN sweep over production E32/top-k-8 routing cases; it does not
   execute them.
2. In a private container and compile cache, compile each stage/config and record
   LDS, VGPR, SGPR, waves/SIMD, spills and VMEM/DS ISA. For packed LDS, require a
   direct buffer-to-LDS sequence rather than VMEM-to-VGPR plus DS writes.
3. Compare each stage with fixed routing distributions (1/2/4/8 local slots and
   all-local), M=1/4/8/16/32/64/128/256, randomized E8M0 scales for MXFP4, graph
   replay, warmups, median and tail. Require current correctness tolerances and
   output sentinels; reject regressions outside measured noise at adjacent M.
4. Only then run fixed-length end-to-end A/B with adequate per-slot warmups and
   concurrency 4/8/16/64/128/256. Use a fresh result path and reject incomplete
   or zero-valued JSON. Native activation quantization additionally requires the
   full gsm8k accuracy gate.

GPU work requires an explicit exclusive lease or scheduler job. Zero VRAM or an
empty process list in this namespace does not prove host-idle status. Do not
replace kernels in running containers or shared caches. The existing `e2e_ab.sh`
uses a global `pkill -9 -f sglang::` cleanup and is unsafe to run alongside other
benchmarks; an isolated runner must terminate only PIDs it started.

`ds_read_b64_tr_b4` is a plausible packed-tile consumer, and a CPU-only gfx950
`llvm-mc` probe confirms that the installed assembler accepts the opcode. This
only proves instruction availability. The CDNA4 ISA also requires full EXEC,
aligned addresses and even destination registers, so it does not establish a
correct FlyDSL lane layout. See the [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
section 11.4 and the [FlyDSL tuning guide](https://rocm.docs.amd.com/projects/FlyDSL/en/latest/kernel_tuning_guide.html).

## Native FP4 result already rules out shipping that probe

The native-FP4 probe is neither correct nor a general performance win. Gate/up
slowed 131.9 -> 133.8 and 132.4 -> 137.2 us; down improved 63.1 -> 58.8 and
67.4 -> 61.1 us, but max absolute error was 0.47--2.28 and relative error
1.02--1.54. VGPR increased 394 -> 452. It should remain a research branch until
activation quantization, lane layout and scale semantics pass kernel references;
then it still requires gsm8k because activation quantization changes numerics.
