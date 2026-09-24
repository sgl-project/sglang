Status: implemented

# DCP/L3 phase evidence

**TL;DR:** All five phases are complete. H200 fresh readers at TP/DCP=2/2, 4/2, and 4/4 restored 1,024 tokens on every rank and produced exactly the cold-run output IDs. Missing-shard recovery, runtime attachment, concurrent writers, and CPU failure tests passed.

## Phase 1 reproduction

Base revision: `9f6fc553322e897a22f30320dae750f4a40a0b84`. Run in the SGLang environment from the repository root:

```sh
unshare --cgroup /opt/sglang/bin/python -m pytest \
  test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py \
  test/registered/unit/mem_cache/test_hicache_file_lru_unit.py \
  test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py -q
```

Result: **48 passed, 22 subtests passed, 15 warnings in 11.38 seconds; exit 0.** See the [captured test output](feature-00-dcp-l3-tests.log). Warnings concern pytest's unavailable asyncio plugin configuration and PyTorch JIT deprecation. The cgroup wrapper is specific to the BJ development job; ordinary environments can run the same Python command directly.

The new identity test is the reproducible script for this phase. Its source blob is `d5b77ff098ad65d16044b0470378d3d2279666b1`; the changed `hicache_storage.py` source blob is `79e880939579b4280336d12d192c6ec0710ca626`.

## Phase 1 review result

```text
TP=4 / DCP=2

TP rank       0       1       2       3
key           K0      K1      K0      K1
writer?       yes     yes     no      no
```

The tests verify this table, TP/DCP=2/2 and 4/4, separate identities for different formats/topologies, DCP=1 MLA/GQA keys, and rejection of invalid DCP metadata. Independent file-backend instances find the same two synthetic objects through reads, batch lookup, and metadata scanning.

At phase-1 verification, Ruff lint/format checks, import sorting, registered-test validation, and whitespace checks passed. Controller, file eviction, host-page access, startup compatibility checks, and the existing host-pool test file matched the base revision exactly. The index was empty and the change was uncommitted. The user subsequently approved phase 1 for commit.

The [baseline probe](feature-00-dcp-l3.py) and [captured result](feature-00-dcp-l3.json) reproduce the DCP/L3 startup and page-access guards at phase-1 commit `1cdb6b1`. Run this historical probe at that revision: phase 2 intentionally removes the generic page-access rejection.

```sh
unshare --cgroup /opt/sglang/bin/python spec/evidence/feature-00-dcp-l3.py
```

## Phase 2 reproduction and review result

Base revision: `1cdb6b1`, with the uncommitted phase-2 change. Two runtime/test files change: `pool_host/mla.py` and `test_hicache_dcp_host_pool.py`.

Exact command from the repository root:

```sh
unshare --cgroup /opt/sglang/bin/python -m pytest \
  test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py \
  test/registered/unit/mem_cache/test_hicache_file_lru_unit.py \
  test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py -q
```

Result: **50 passed, 34 subtests passed, 15 warnings in 11.27 seconds; exit 0.** See the [captured phase-2 output](feature-00-dcp-l3-phase-2.txt). Warnings are the same pytest configuration and PyTorch deprecation warnings as phase 1.

The tests use real CPU MLA host pools and temporary files. For every rank at DCP=2 and 4, two nonadjacent pages are saved and restored into different host allocations. The entire destination buffer matches the expected result exactly, including untouched neighboring rows. Each page contains 64 local rows, 2 layers, width 12, and BF16 elements: **3,072 bytes**.

```text
DCP=2, physical page=64 rows, logical page=128 tokens

logical page start 256 -> physical row 128
one logical page      -> physical rows 128..191
```

Negative or misaligned logical page starts and out-of-range reads reject. DCP zero-copy page metadata stays guarded. Existing DCP L1/L2 host-pool tests and DCP=1 file backend tests pass.

At phase-2 verification, Ruff lint/format checks, import sorting, registered-test validation, and whitespace checks passed. Controller, file backend, file eviction, and startup compatibility checks matched the phase-1 commit exactly; the index was empty. The user subsequently approved phase 2, committed as `5ba812b`.

## Phase 3 reproduction and review result

Phase-3 base revision: `5ba812b`; completed commit: `866d936`. The controller passes DCP format information into storage configuration and uses its writer selection for backups. The file evictor uses the same selection. Existing backup-worker and acknowledgment-draining logic is unchanged. Runtime attachment rejects DCP before side effects, preserving the unfinished feature's guard.

Exact commands from the repository root on BJ job 3853948:

```sh
unshare --cgroup /opt/sglang/bin/python -m pytest \
  test/registered/unit/mem_cache/test_hicache_dcp_storage_controller.py \
  test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py \
  test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py \
  test/registered/unit/mem_cache/test_hicache_file_lru_unit.py -q -s

unshare --cgroup /opt/sglang/bin/python -m pytest \
  test/registered/unit/managers/test_scheduler_hicache_attach.py \
  test/registered/unit/mem_cache/test_hicache_pp_sync_drain.py -q

/opt/sglang/bin/python scripts/lint/check_registered_tests.py
```

Results: **54 passed, 44 subtests passed, 15 warnings in 11.62 seconds**, then **6 passed, 15 warnings in 12.31 seconds**. Both pytest commands and registered-test validation exited 0. See the [component test output](feature-00-dcp-l3-phase-3.txt) and [regression output](feature-00-dcp-l3-phase-3-regressions.txt). Warnings concern the same pytest configuration and PyTorch JIT deprecation as prior phases.

The new controller test uses real CPU MLA host pools, temporary files, backup workers, and acknowledgment queue draining. GPU allocation and process placement are mocked; a mocked tree-unlock callback frees a real host allocation. This verifies the component completion path, not GPU transfers, the full radix-tree lock implementation, or distributed reductions.

```text
TP=4 / DCP=2, one 128-token logical page

rank   writer   writes   acknowledgments   remaining host slots
  0      yes       1            1                    0
  1      yes       1            1                    0
  2       no       0            1                    0
  3       no       0            1                    0

two final objects, 3,072 bytes each
```

While rank 1's storage write is held by an event, ranks 0/2/3 acknowledge, rank 1 has no acknowledgment, its tree-unlock callback is not called, and its host allocation remains live. Once released, rank 1 completes. Draining each rank's acknowledgment releases its allocation. All four ranks then read the correct shard through the generic controller page-read path, including replicas 2/3; those read allocations are also freed.

Only ranks 0/1 account for stored bytes. With capacity for one object per shard, rank 0 evicts only its shard of page A when saving page B; rank 1's A remains until rank 1 saves B. Ranks 2/3 neither account for nor evict these files. DCP=1 keeps the previous MLA rank-0 and GQA all-rank writer rules. Runtime attachment rejects DCP=2/4 before starting/stopping storage workers or generating storage configuration.

Ruff lint/format checks, import sorting, registered-test validation, and whitespace checks passed. At phase-3 verification, the index was empty for human review; this phase was subsequently committed as `866d936`.

## Phase 4 reproduction and result

Phase 3 is committed as `866d936`. The user authorized completing the remaining plan without review pauses.

```sh
unshare --cgroup /opt/sglang/bin/python -m pytest test/registered/unit/mem_cache/test_hicache_dcp_storage_failures.py -q -s
unshare --cgroup /opt/sglang/bin/python -m pytest test/registered/unit/mem_cache/test_hicache_dcp_storage_controller.py test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py test/registered/unit/mem_cache/test_hicache_file_lru_unit.py test/registered/unit/managers/test_scheduler_hicache_attach.py test/registered/unit/mem_cache/test_hicache_pp_sync_drain.py -q
/opt/sglang/bin/python scripts/lint/check_registered_tests.py
```

Results: distributed test **1 passed in 24.77s**; regressions **60 passed, 44 subtests in 13.11s**. Both had the same 15 existing warnings. Registered-test validation and whitespace checks passed. [Distributed output](feature-00-dcp-l3-phase-4.txt), [regression output](feature-00-dcp-l3-phase-4-regressions.txt).

Four Gloo processes run the concrete unified-cache controller at TP=4/DCP=2, real storage workers, file I/O, prefix reductions, acknowledgment draining, and host allocation/free. Only GPU construction, the process-group factory wrapper, and tree insertion/unlock are replaced. Every selected group contains ranks 0/1/2/3. Each request spans three 128-token logical pages, with one page per I/O batch to exercise all acknowledgments after failure.

| Case | Reused tokens on every rank | Host slots left |
| --- | --- | --- |
| Healthy | 384 | 0 |
| Middle shard missing | 128 | 0 |
| Middle shard truncated | 128 | 0 |
| Lookup exception on rank 1 | 0 | 0 |
| Read exception after hit | 128 | 0 |
| Eviction after lookup | 128 | 0 |
| Failed/delayed backup | 384 from initial read; write acknowledged after I/O ends | 0 |

The test has collective and parent-process timeouts. Rank 0 reads remain in flight during injected rank 1 read failures. Backup failures acknowledge and keep the worker alive. Short-read detection uses the existing `readinto` count; no file-size check was added.

## Phase 5: enabled configuration and H200 release gate

Phase 4 is committed as `9c67b43`. Phase 5 enables the supported configuration at startup and runtime attachment, rejects unsupported policy changes before side effects, and validates the concrete host pool before starting storage workers.

### CPU checks

From the repository root, using `/opt/sglang/bin/python` under `unshare --cgroup` on this BJ job:

```sh
unshare --cgroup /opt/sglang/bin/python -m pytest test/registered/unit/mem_cache/test_hicache_dcp_storage_failures.py test/registered/unit/mem_cache/test_hicache_dcp_storage_guards.py test/registered/unit/mem_cache/test_hicache_dcp_storage_controller.py test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py test/registered/unit/mem_cache/test_hicache_file_lru_unit.py test/registered/unit/managers/test_scheduler_hicache_attach.py test/registered/unit/mem_cache/test_hicache_pp_sync_drain.py -q
unshare --cgroup /opt/sglang/bin/python -m pytest test/registered/unit/server_args/test_server_args.py -k hicache -q
unshare --cgroup /opt/sglang/bin/python -m pytest test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py -k 'storage or prefetch' -q
unshare --cgroup /opt/sglang/bin/python -m pytest test/registered/unit/mem_cache/test_hicache_dcp_storage_guards.py -q
/opt/sglang/bin/python scripts/lint/check_registered_tests.py
```

Measured results, respectively:

- **64 passed, 63 subtests, 27.36s**: [focused CPU output](feature-00-dcp-l3-phase-5-cpu.txt).
- **9 passed, 20 subtests, 271 deselected, 10.34s**: [argument regressions](feature-00-dcp-l3-phase-5-args.txt).
- **89 passed, 137 skipped, 2,505 deselected, 33.18s**: [unified-cache storage/prefetch regressions](feature-00-dcp-l3-phase-5-unified.txt).
- After adding both accepted BF16 spelling checks: **3 passed, 21 subtests, 10.58s**: [final guard checks](feature-00-dcp-l3-phase-5-guards.txt).

Each pytest invocation above exited 0 and reported the same 15 existing pytest/PyTorch warnings. Registered-test validation, Ruff lint/format checks, import sorting, and whitespace checks passed. An accidental omission of existing disaggregation layout handling was caught by the argument tests and restored before their final passing run.

### H200 reproduction

Environment: BJ job `3853948`, node `slurm-h200-206-089`, H200 GPUs 2-5. Repository: `/mnt/home/byron/sglang-worktree/dcp-l3-dev-3853948`, branch `dcp-hicache-l3`. Model cache revision: `85864749cd611b4353ce1decdb286193298f64c7` of `deepseek-ai/DeepSeek-V2-Lite-Chat`.

The checked-in script `test/manual/hicache/test_dcp_l3.py` uses SGLang's `popen_launch_server` and process-tree cleanup helpers, real `/generate` requests, and real file storage. The input has 1,025 tokens; generation is greedy for eight tokens. It polls complete file publication with a 60-second deadline. Server startup and each request have bounded timeouts. Runtime attachment uses an admin key and waits for scheduler idleness if startup work is still draining.

Reproduce the full matrix with four free H200 GPUs:

```sh
export PATH=/opt/sglang/bin:$PATH
export HF_HOME=/mnt/home/byron/.cache/huggingface
CUDA_VISIBLE_DEVICES=2,3,4,5 DCP_L3_OUTPUT_DIR=.dev/dcp-l3-release \
  unshare --cgroup /opt/sglang/bin/python -m pytest test/manual/hicache/test_dcp_l3.py -q -s
```

Validation was run in subsets during development, with the same Python/pytest command:

```sh
CUDA_VISIBLE_DEVICES=2,3,4,5 DCP_L3_OUTPUT_DIR=.dev/phase5 DCP_L3_TOPOLOGIES=2:2 DCP_L3_SKIP_CONCURRENCY=1 unshare --cgroup /opt/sglang/bin/python -m pytest test/manual/hicache/test_dcp_l3.py -q -s
CUDA_VISIBLE_DEVICES=2,3,4,5 DCP_L3_OUTPUT_DIR=.dev/phase5-rest DCP_L3_TOPOLOGIES=4:2,4:4 unshare --cgroup /opt/sglang/bin/python -m pytest test/manual/hicache/test_dcp_l3.py -q -s
CUDA_VISIBLE_DEVICES=2,3,4,5 DCP_L3_OUTPUT_DIR=.dev/phase5-release DCP_L3_TOPOLOGIES=4:4 unshare --cgroup /opt/sglang/bin/python -m pytest test/manual/hicache/test_dcp_l3.py -q -s
```

The first run passed in **116.47s**. The middle run completed all TP=4/DCP=2 assertions, including missing-shard inference, before its later TP=4/DCP=4 runtime-attach setup failed because the test server lacked an admin key. The final run used the corrected authenticated setup and passed TP=4/DCP=4 plus concurrent writers in **285.75s**, exit 0. This last run showed HTTP 200 for runtime attachment and a positive L3 restore on all four ranks.

| TP / DCP | Logical pages saved | Final objects | Restored logical tokens per rank | Output IDs equal cold |
| --- | --- | --- | --- | --- |
| 2 / 2 | 8 | 16 | 1,024 / 1,024 | yes |
| 4 / 2 | 8 | 16 | 1,024 / 1,024 / 1,024 / 1,024 | yes |
| 4 / 4 | 4 | 16 | 1,024 / 1,024 / 1,024 / 1,024 | yes |

Each object is **1,990,656 bytes**: 64 physical rows × 27 layers × 576 BF16 values × 2 bytes. Every topology therefore stored 31,850,496 bytes for the same 1,024-token prefix. TP=4/DCP=2 stored two objects per logical page, with no additional replica objects.

At TP=4/DCP=2, deleting shard 1 of the second page made every rank restore exactly **128 tokens**. Inference computed the rest and produced the same output IDs as the cold run.

Two TP=2/DCP=2 engines then processed the same full prompt concurrently against empty shared storage. Both logged 1,025 new tokens and zero cached tokens at `23:42:49 UTC`. They left **16 complete objects**. A third, fresh engine restored **1,024 tokens per rank** and produced the same output IDs.

All successful cases produced `[280, 1319, 2135, 13, 338, 8959, 481, 16074]`. See the [measured result table](feature-00-dcp-l3-phase-5.json) and [per-rank H200 trace](feature-00-dcp-l3-phase-5-h200.txt). Complete server logs remain under the `.dev/` directories named above; the committed trace retains the relevant restore counts, prefill counts, and attachment result.

## Boundaries

These results cover the first-stage dense MLA/file contract with matching topology. They do not establish GQA DCP storage, topology conversion, other storage backends, or performance targets.
