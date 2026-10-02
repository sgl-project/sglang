# Target-KV Checkpoint Artifact Audit

Use this SGLang API after a trusted fixed-input parity run and before deploying
the resulting immutable checkpoint directory. It checks existing evidence and
does not load SpecForge, a target decoder or a GPU model:

```bash
PYTHONPATH=python python -m \
  sglang.srt.speculative.dspark_components.dspark_target_kv_artifact \
  --checkpoint /models/kv-draft
```

The equivalent Python interface is:

```python
from sglang.srt.speculative.dspark_components.dspark_target_kv_artifact import (
    audit_target_kv_checkpoint,
)

receipt = audit_target_kv_checkpoint("/models/kv-draft")
```

## Required Evidence

The current fixed-input report binds these exact files:

- `config.json`, including the typed target-KV contract.
- `model.safetensors`, the sole root safetensors weight file.
- `validation/inputs.safetensors`, whose digest also matches the contract.
- `validation/parity.json`, produced by the existing numerical validation gate.

Extra root weight files or `model.safetensors.index.json` are rejected because
they could change the serving loader's selected weights without appearing in
this report. Sharded checkpoint evidence needs a corresponding numerical-report
extension; this audit does not alter the serving weight loader.

The report must pass every configured decoder layer, normalized hidden output,
shared-head logits and corrected Markov logits. Element counts must match the
reported anchor count, prediction width and hidden/vocabulary dimensions.
Nonfinite/mismatched values, incomplete stages, empty windows, impossible label
counts, changed dtype, loosened tolerances, missing training-source identities,
or missing/nonfinite/zero gradient groups are rejected.

Metadata reads are bounded. Large artifacts are hashed in 8 MiB chunks, and
file identity/size/timestamps are checked before and after reads and again before
returning. The receipt contains the contract fingerprint and file/report hashes.
The CLI prints JSON with `status=verified` and exits zero, or prints a failure
reason and exits one. It never updates the checkpoint or its reports.

## Acceptance Report

When `acceptance_report_sha256` is set in the contract, the corresponding file
must be `validation/acceptance.json` and match that digest. Add
`--require-acceptance` when the deployment workflow requires this pinned artifact;
without a declared digest the command then fails.

The acceptance report is opaque here. `acceptance_report_bound=true` means only
that its bytes match the declared reference. Actual benchmark thresholds,
trained-model quality and rollout policy remain separate decisions. Updating the
contract changes `config.json` and requires rerunning the fixed-input gate.

## Numerical Gate Integration

```bash
PYTHONPATH=python:/path/to/pinned/SpecForge \
python -m sglang.test.dspark_target_kv_parity \
  --checkpoint /models/kv-draft --target-path /models/target \
  --reference-attention flex_attention
```

The gate compares the actual serving model with the pinned training reference,
checks a cached CE+TV128 backward pass and forbids target decoder forward. Before
successfully returning, it calls the artifact auditor. A post-comparison artifact
failure changes the report to failed and raises an error.

An offline receipt assumes a trusted report and does not rerun its calculations,
authenticate its author or certify a different installed runtime. Startup still
checks exact parameter names/shapes/dtypes and the bound target identity. A
synthetic checkpoint remains a correctness fixture, not a trained draft.

## Verified Results

The final sources pass nine CPU tests in 0.055 seconds with no visible CUDA
devices and the full five-test numerical parity regression in 2.499 seconds.
The regression includes vanilla, gated and RNN heads, an optimizer step followed
by export/reload, prefix isolation and failed-report cleanup.

A fresh run of the retained Qwen3 BF16 fixture passes every layer, hidden and
logits comparison with zero error, and computes cached CE+TV128 over seven valid
labels with finite nonzero trainable gradients. A subsequent standalone CPU CLI
audit returns the corresponding artifact-bound receipt. Final job results,
source/log hashes and the receipt are retained in
[checkpoint-artifact-audit.json](checkpoint-artifact-audit.json).

The resident H100 paused its idle load for these queued jobs and resumed at 60%
after all jobs completed. No extra GPU allocation was used.
