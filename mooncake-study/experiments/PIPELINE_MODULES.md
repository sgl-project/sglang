# DSpark Pipeline Shared Modules

This dependency supplies target embedding/head TP shards to draft replicas on
all PP stages. It also makes draft graph buffers use the draft runner's own PP
size without mutating the published target configuration. Public PP speculative
serving gates remain closed.

Run the complete registered unit files:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_shared_modules.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_draft_per_runner_config.py -v -f
PYTHONPATH=python python test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py -v -f
```

Run native device transport and the existing AR PP2 regression with two GPUs:

```bash
PYTHONPATH=python python test/registered/storage/test_dspark_shared_modules_nccl.py -v -f
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_pp.py -v -f
```

Run the existing DSpark single-GPU regressions:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark.py -v -f
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_hidden.py -v -f
```

The shared-module fixture uses native vocabulary modules and actual Gloo/NCCL
collectives with deterministic weights. It substitutes group discovery and the
target model container; it does not load a target or draft checkpoint. Each
TP2/PP2, TP1/PP4 and CUDA TP1/PP2 run covers BF16, FP16 and FP32, each with tied
and untied weight values. It checks:

- Owner object identity and unchanged missing modules on other target stages.
- Replica device/dtype and non-trainable weights.
- Exact embedding rows across the TP boundary and logits cropped to vocabulary
  size 67, including padded shards.
- Coordinated failure for a missing source, an unsupported head dtype and a
  replica allocation failure, followed by a valid operation on the same groups.
- With TP2, wrong source shard identity and cross-lane metadata disagreement.

The runner fixture stops draft construction at the factory boundary and checks
PP=1/rank=0 with the target TP lane preserved. A separate case calls the real
base runner constructor and buffer allocator for draft PP1 and target PP4 while
the shared published configuration stays PP4. It does not prove complete draft
model initialization, graph capture or checkpoint compatibility in PP.

The AR PP2 P/D regression covers eager and decode CUDA graphs on a real target
model. The single-GPU suites cover target-KV drafts on D alone and on both P/D,
plus hidden-input static, cap-accept and compact verification in eager and
graph/overlap modes. They use actual Mooncake TCP transport and Store with the
Catalog test double, validate stored tensors against online sources, and exclude
missing/stale handoffs and cancellation from published snapshots.

PP module copying adds padded local embedding/head weight storage to stages
that do not own those modules. There is no automatic refresh after weight
replacement. Native unquantized modules are the supported PP binding; custom or
quantized modules need their own binding contract. Source-KV assembly over NCCL,
full PP model initialization, coordinated proposal/activation/acceptance
scheduling, distributed cancellation and production performance remain outside
this verification. No Mooncake API or training snapshot format changes here.

Exact run IDs, source/log hashes, failed setup attempts and results are retained
in `pipeline-dspark-modules.json`.
