Status: proposed

# DCP/L3 phase evidence

**TL;DR:** Phase 1 is committed as `1cdb6b1`. Phase 2 passes 50 tests and 34 subtests and has been reviewed and approved for commit. Later phases have not started.

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

Ruff lint/format checks, import sorting, registered-test validation, and whitespace checks passed. Controller, file backend, file eviction, and startup compatibility checks match the phase-1 commit exactly; the index is empty. The user subsequently approved phase 2 for commit.

## Boundaries

Writer selection is not wired into controller backups or eviction yet. Phase 2 does not enable DCP/L3 or prove model KV restore. Distributed read-failure handling remains for phase 4. Phase 3 starts only after phase 2 review, approval to commit, and an instruction to continue.
