# Mooncake SWA packing validation

For configuration, supported layouts, fallback and failure behavior, see
[Mooncake SWA sender packing](../../../docs/docs/advanced_features/pd_disaggregation.mdx#mooncake-swa-sender-packing).
Historical benchmark tables, tested source revisions and validation limits are
recorded in the [PR description](https://github.com/sgl-project/sglang/pull/47313#historical-benchmark-and-validation-evidence).
Those GPU and model results were not measured on the current PR head.

## CPU regression checks

From the repository root, in an environment with SGLang's test dependencies:

```bash
PYTHONPATH=python python -m unittest discover -s test/registered/unit/disaggregation -p 'test_mooncake_swa_packing.py' -v
PYTHONPATH=python python -m unittest discover -s test/registered/unit/disaggregation -p 'test_mooncake_packed_setup.py' -v
```

Both commands should pass. They cover layout proofs, page bounds, optional peer
metadata compatibility, setup serialization and buffer ownership after uncertain
completion. They do not validate GPU kernels, native RDMA transfers or model outputs.

## GPU and model validation

The historical GPU harness and raw artifacts are not checked into this directory.
The PR description records their methodology and limitations; the CPU checks above
are not a replacement for rerunning GPU and model validation on the target revision.

When validating a new revision:

1. Record both peers' SGLang revisions, Mooncake binary identity, GPU/runtime and
   model revisions, attention TP sizes, page geometry and packing settings.
2. Verify that both peers use RDMA and that `MC_FORCE_TCP` is absent. Check copied
   destination bytes and untouched sentinel regions using independent source and
   destination page indices.
3. Compare model outputs with an independent greedy reference. Verify that the
   intended packed path is exercised; matching output alone does not prove this.
4. For latency comparisons, bracket a packed run with packing-disabled baseline
   runs, hold the workload and other settings fixed, report warmups and sample
   counts, and retain outliers. Report full-KV staging settings separately.
5. Keep GPU/native/model results tied to the revision actually tested. CPU success
   or a rebase does not establish that historical GPU results still hold.
