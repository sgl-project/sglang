Status: proposed

# Phase 1 evidence

**TL;DR:** The restarted phase-1 change passes 48 tests and 22 subtests. The user reviewed and approved it for commit. Later phases have no changes in this working tree.

## Reproduction

Base revision: `9f6fc553322e897a22f30320dae750f4a40a0b84`. Run in the SGLang environment from the repository root:

```sh
unshare --cgroup /opt/sglang/bin/python -m pytest \
  test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py \
  test/registered/unit/mem_cache/test_hicache_file_lru_unit.py \
  test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py -q
```

Result: **48 passed, 22 subtests passed, 15 warnings in 11.38 seconds; exit 0.** See the [captured test output](feature-00-dcp-l3-tests.log). Warnings concern pytest's unavailable asyncio plugin configuration and PyTorch JIT deprecation. The cgroup wrapper is specific to the BJ development job; ordinary environments can run the same Python command directly.

The new identity test is the reproducible script for this phase. Its source blob is `d5b77ff098ad65d16044b0470378d3d2279666b1`; the changed `hicache_storage.py` source blob is `79e880939579b4280336d12d192c6ec0710ca626`.

## Review result

```text
TP=4 / DCP=2

TP rank       0       1       2       3
key           K0      K1      K0      K1
writer?       yes     yes     no      no
```

The tests verify this table, TP/DCP=2/2 and 4/4, separate identities for different formats/topologies, DCP=1 MLA/GQA keys, and rejection of invalid DCP metadata. Independent file-backend instances find the same two synthetic objects through reads, batch lookup, and metadata scanning.

Ruff lint/format checks, import sorting, registered-test validation, and whitespace checks passed. Controller, file eviction, host-page access, startup compatibility checks, and the existing host-pool test file match the base revision exactly. At verification, the index was empty and the change was uncommitted. The user subsequently approved phase 1 for commit.

The [baseline probe](feature-00-dcp-l3.py) and [captured result](feature-00-dcp-l3.json) reproduce the DCP/L3 startup and page-access guards:

```sh
unshare --cgroup /opt/sglang/bin/python spec/evidence/feature-00-dcp-l3.py
```

## Boundaries

Writer selection is defined but is not wired into controller backups or eviction yet. This phase does not enable DCP/L3 or prove model KV restore. Phase 2 starts only after phase 1 review, approval to commit, and an instruction to continue.
