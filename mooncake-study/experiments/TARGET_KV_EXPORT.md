# Target-KV Training Checkpoint Export

The SGLang exporter accepts consolidated KV-input draft weights from a trainer
and emits a new serving directory. It does not build a target decoder or run
prefill. The trainer must stop modifying the state during export and consolidate
TP/FSDP shards before calling this interface.

```python
from sglang.srt.speculative.dspark_components.dspark_target_kv_export import (
    export_target_kv_checkpoint,
)

receipt = export_target_kv_checkpoint(
    draft_config,
    kv_draft.state_dict(),
    golden_fixture="/checkpoints/inputs.safetensors",
    output_dir="/models/kv-draft-version-1",
)
```

`draft_config` is a mapping or Hugging Face config object with the explicit
`DSparkTargetKVDraftModel` architecture, `input_mode=target_kv`, full typed
contract, registered `model_type`, sequence/Markov settings and floating dtype.
The exporter resolves the actual Hugging Face defaults. It rejects conflicting
dtype and sequence/Markov aliases and preserves the teacher/KV contract.

For a checkpoint manager that already writes safetensors:

```bash
PYTHONPATH=python python -m \
  sglang.srt.speculative.dspark_components.dspark_target_kv_export \
  --config /checkpoints/draft-config.json \
  --weights /checkpoints/draft-state.safetensors \
  --golden-fixture /checkpoints/inputs.safetensors \
  --output-dir /models/kv-draft-version-1
```

## Weight And Artifact Rules

Only the complete KV encoder, dense draft backbone/context projections, final
norm and selected Markov head are exported. Target decoder/shared embedding/head,
confidence and legacy hidden-projection parameters are not silently filtered.
Missing or unexpected names fail. The caller can pass a mapping or an iterable
of `(name, tensor)`; duplicate names, including after removing `model.`, fail.

Both complete native QKV/gate-up weights and split Q/K/V/gate/up weights are
accepted. Mixed full/partial representations fail. Output uses split names and
private contiguous CPU tensors in the config's dtype. GQA head geometry and
attention biases are preserved; Markov `gate_proj` is a complete parameter and
is never interpreted as an MLP shard. FP32/FP16/BF16 are supported structurally;
wrong shapes, non-floating tensors, meta tensors, NaN/Inf and destination
overflow are rejected before publication.

The golden fixture must be a nonempty safetensors file with the exact digest in
the contract. Its full numerical/sequence content is checked by the subsequent
parity gate. When the contract pins an acceptance report, supply its path through
`acceptance_report` or `--acceptance-report`; unpinned or mismatched reports fail.
Acceptance bytes are opaque here, not a quality/SLO decision.

Files are written and synced in a private sibling staging directory, then the
complete directory is renamed into place. Existing output paths are rejected;
pre-publication exceptions clean up staging. The receipt lists artifact hashes,
contract fingerprint, tensor count/bytes and `requires_fixed_input_parity=true`.
It is retained as `export.json`. No old numerical pass is copied.

## Validate The New Directory

```bash
PYTHONPATH=python:/path/to/pinned/SpecForge \
python -m sglang.test.dspark_target_kv_parity \
  --checkpoint /models/kv-draft-version-1 --target-path /models/target \
  --reference-attention flex_attention

PYTHONPATH=python python -m \
  sglang.srt.speculative.dspark_components.dspark_target_kv_artifact \
  --checkpoint /models/kv-draft-version-1
```

The numerical gate reloads the actual serving model, compares every layer and
both logits stages, and checks cached CE+TV128 gradients. The artifact auditor
binds that new report to the exported bytes. Neither export nor artifact audit
establishes trained-model quality or serving SLOs; startup still checks loaded
target identity and supported execution capabilities.

The SpecForge reference tests now use this API after a real optimizer step for
vanilla, gated and RNN heads. A separate test exports the native packed serving
parameters and verifies the reloaded model against the training reference.
The production SpecForge checkpoint-manager adapter and full training workflow
remain integration work.

## Verified Results

The final source passes 10 CPU export tests in 0.093 seconds and six GPU parity
tests in 3.257 seconds. The CPU tests check split/packed weights, attention bias,
all Markov heads, malformed/missing/duplicate weights, cast overflow, config
conflicts, artifact mismatches, write failure cleanup and destination preservation.

A retained Qwen3 BF16 checkpoint exported 27 parameters and 80,372,736 tensor
bytes without changing its teacher/KV contract fingerprint. A fresh numerical
run on the exported directory has zero layer/hidden/base/corrected error and
finite nonzero gradients over seven cached labels. The standalone artifact
auditor verifies the new config, weights, fixture and parity report together.
Source/log hashes, exact commands and receipts are in
[the evidence JSON](target-kv-export.json).
