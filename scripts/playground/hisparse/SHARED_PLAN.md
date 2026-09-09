# Experimental shared-plan Demand

Enable `SGLANG_TEST_HISPARSE_GROUP_PLAN=1` on the validated GLM MTP Demand path. It is off by default. The native FlashMLA extension must be rebuilt with this source; an older binary does not provide the group operator.

An IndexShare-group anchor runs Demand placement and emits an int32 slot plan. Followers acquire-check their own layer's tag and populate their own layer's KV at that slot. Followers do not hash, choose victims, or promote READY-hit generations. FILLING is never overwritten; unassigned entries read Host KV without allocating. The plan must be current, complete, and map each slot to one Host row. TopK selection is unchanged. Execution is ordered, without programmatic dependent launch or two-batch overlap.

The plan buffer is reused across layers: 4 verify rows × 2048 entries × 4 bytes = 32 KiB per preallocated request slot per GPU. It does not duplicate KV. Existing cache capacity and Host storage remain unchanged when only the group flag is toggled.

Related experimental controls retain their defaults:

- `SGLANG_TEST_HISPARSE_DEMAND_CACHE_ROWS=4096` (8192 also supported).
- `SGLANG_TEST_HISPARSE_DEMAND_HOST_STRIDE=656` (768 also supported; padding consumes additional Host memory).
- `SGLANG_DEBUG_HISPARSE_DEMAND_SOURCE_COUNTS=0`. Diagnostic counts are source choices, not measured PCIe bytes. Keep counters off for timing.

## H20 evidence, 2026-09-09

GLM-5.2 W4AFP8, TP8/DP8, MTP, IndexShare=true, B1, 128K input / 1K output, 8192 cache rows and 768-byte Host stride for both Demand modes. Same native binary across all modes, counters off, one warmup plus three timed requests, no profiler. Acceptance simulation is a timing control, not accuracy evidence.

| Mode | Mean TPOT (ms) | Overhead versus HBM |
| --- | ---: | ---: |
| HBM | 22.488238 | baseline |
| Independent Demand | 23.626530 | 5.0617% |
| Shared-plan Demand | 23.473050 | 4.3792% |

Shared plan improves this full-model matrix by 0.6496%; paired gains are 0.6138/0.6950/0.6389%. Verify counts match at 320/328/322, with identical input hashes and zero retractions. Modes ran sequentially, so slow temporal drift is not excluded. The original <=3% HBM-overhead target is not met.

A five-pair interleaved native two-layer screen showed 90.845 us retained control, 91.555 us same-binary independent, and 82.180 us shared. The 9.54% native gain is not a full-model gain. Each output/LSE byte-matched its HBM reference. Native shared-plan regression covers per-layer KV separation, arbitrary physical mappings, changing working sets, FILLING fallback, Host-only plans, epoch wrap, request reset with new contents, and two-group CUDA Graph reuse.

Natural-text acceptance non-regression is **not established**. On three 8K/512 ShareGPT seeds, independent/shared acceptance was 61.30/67.46%, 73.33/58.42%, and 68.66/53.20%; all continuations differed. An independent fresh-process repeat also changed all continuations, with acceptance 60.11/72.43/67.06%. Baseline non-repeatability does not clear the larger shared-mode drops. Root cause is unresolved; do not claim output parity, general quality, or production readiness from this smoke.

## Reproduce correctness checks

After building/installing this checkout's Python package and native kernel:

```sh
python3 -m pytest -q test/registered/kernels/ops/attention/test_flashmla_hisparse_mtp_demand.py
python3 -m pytest -q test/registered/unit/managers/test_hisparse_group_plan.py test/registered/unit/managers/test_hisparse_unit.py
```

Native tests require SM90 and CUDA >=12.4. To reproduce the measured configuration, use the existing `launch_glm52_mtp_release.sh` with the explicit cache/Host-stride overrides above and `INDEX_SHARE_FOR_MTP_ITERATION=true`; toggle only the group flag between independent/shared Demand.
