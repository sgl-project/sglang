Status: proposed

# DCP/L3 phase evidence

**TL;DR:** Phases 1 and 2 are committed as `1cdb6b1` and `5ba812b`. Phase 3 is uncommitted for review: 54 tests and 44 subtests pass, plus 6 attachment/queue regression tests. At TP=4/DCP=2, writes are `[1, 1, 0, 0]`, every rank acknowledges its operation, and no test host allocations remain. Phases 4-5 have not started.

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

Base revision: `5ba812b`, with the uncommitted phase-3 change. The controller passes DCP format information into storage configuration and uses its writer selection for backups. The file evictor uses the same selection. Existing backup-worker and acknowledgment-draining logic is unchanged. Runtime attachment rejects DCP before side effects, preserving the unfinished feature's guard.

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

Ruff lint/format checks, import sorting, registered-test validation, and whitespace checks passed. The phase-3 index remains empty for human review.

## Boundaries

Writer selection is wired into controller backups and file eviction, but startup and runtime attachment still reject DCP/L3. These component tests do not prove model KV restore or cross-rank failure agreement. Distributed read-failure handling remains for phase 4, and H200 cross-engine restore remains for phase 5. Phase 3 awaits review; neither later phase has started.
