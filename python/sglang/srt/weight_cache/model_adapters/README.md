# Weight-layout adapters

Weight-cache source and target daemons use the same adapter to describe their
final runtime allocations. The source loads the checkpoint normally. The target
uses `DummyModelLoader` and never replays `model.load_weights`. Mooncake consumes
the resulting manifests and transfers the required regions directly.

## Supported layouts

This initial implementation supports NVIDIA CUDA, unquantized FP16/BF16/FP32
parameters, immutable weights, and the existing single-node TP/PP/EP topology.
Attention DP is supported only when its attention weights are fully replicated
(`attn_dp_size == tp_size`), or when attention DP is disabled.

| Adapter | Reference checkpoints | Runtime layout |
| --- | --- | --- |
| `Qwen3WeightAdapter` | Qwen3-0.6B and Qwen3-8B BF16 | Dense Qwen3, fused QKV/gate-up, GQA, tied or untied vocabulary |
| `DeepseekV2WeightAdapter` | DeepSeek-V2-Lite BF16 | MLA with `q_lora_rank=None`, separate shared experts, standard Triton MoE |

DeepSeek requires `--moe-runner-backend triton` and
`SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED=0` on source, target, and IPC engines.
The `triton_kernels` backend is a different, transposed layout and is not
supported. Quantization, other expert packing, EPLB, redundant experts, fused
shared experts, V3/DSA/NextN, and Qwen3-MoE are not enabled by these registrations.
Unknown model types and subclasses fail explicitly.

## Adapter contract

Implement `WeightLayoutAdapter` in this directory and add an exact architecture
registration to `_ADAPTERS` in `__init__.py`. The model class is checked by
identity after import; matching a class name alone is insufficient.

- `validate(model, context)` checks model/backend capabilities. Call the base
  method to enforce CUDA and unquantized storage.
- `build_layout(model, context)` returns a `WeightLayoutPlan` without reading or
  writing weight values. Use final module metadata and actual parallel groups.
- `finalize_target(model, context)` rebuilds checkpoint-derived state after the
  transfer. The default restores parameter aliases; DeepSeek additionally calls
  its native `post_load_weights` hook. Do not rerun generic quantization packing.

`context.config` is the model's Hugging Face configuration. Helpers obtain each
module's actual TP group rather than assuming attention TP, global TP and MoE TP
are interchangeable.

Start from this template and the two concrete adapters:

```python
from .base import WeightLayoutAdapter
from .layers import LayerLayoutBuilder


class ExampleWeightAdapter(WeightLayoutAdapter):
    def build_layout(self, model, context):
        b = LayerLayoutBuilder(model)
        for index in range(model.start_layer, model.end_layer):
            layer = model.model.layers[index]
            prefix = f"model.layers.{index}"
            b.qkv_parallel(
                layer.self_attn.qkv_proj,
                tuple(f"{prefix}.self_attn.{part}_proj" for part in ("q", "k", "v")),
            )
            b.row_parallel(layer.self_attn.o_proj, prefix + ".self_attn.o_proj")
            b.mlp(layer.mlp, prefix + ".mlp")
            # Explicitly declare this model's norms, vocabulary, buffers and
            # other weights as well. build() rejects any unaccounted storage.
        return b.build()
```

## Helpers and logical identities

| Helper | Input | Output |
| --- | --- | --- |
| `replicated` | Tensor and full logical ID | One complete logical tensor |
| `replicated_linear` | ReplicatedLinear and logical prefix | Weight and optional bias |
| `column_parallel` / `row_parallel` | Exact native layer and prefix | Shard from the module's TP group |
| `merged_column` | MergedColumnParallelLinear and ordered prefixes | One region per local fused segment |
| `qkv_parallel` | QKVParallelLinear and Q/K/V prefixes | Three regions with GQA replica coordinates |
| `vocab_parallel` | Embedding/LM head and full logical ID | Valid vocabulary plus local zero padding |
| `fused_moe` | Standard FusedMoE and expert prefix | Three regions per local logical expert |
| `mlp` | Native gate_up/down MLP and prefix | Merged-column plus row-parallel declarations |

Use stable, checkpoint-like logical names even when multiple logical weights
share one runtime parameter. Linear helpers append `.weight` and `.bias`.
Layer indices are global across PP. Expert IDs identify logical experts, not
local storage slots. Tied vocabulary uses the same logical ID on every PP rank.

For a layout not covered by a helper, `b.region(...)` declares a rectangular
logical box backed by a contiguous byte range. It takes a registered tensor,
logical ID, global shape/offset, local shape and byte offset. It does not take a
raw address. Supply `partition=WeightPartition(rank=..., size=..., dim=...)`
for a sharded tensor; the default declares a fully replicated tensor. The
common layer helpers populate this declaration from the actual module groups.
`layout_fingerprint` identifies the representation and axis order;
do not include rank, topology size or addresses in it. The initial helpers use
`sglang:row-major`.

Helpers have bounded region counts per segment/expert. Do not implement
per-element provenance or generic PyTorch operation replay as a fallback. Add a
specific representation and native-loader tests for a new layout instead.

## Mooncake boundary

`WeightPartition` describes the effective partition rank, size and split
dimension. Dense ranks declaring the same partition rank are replicas. The
`expert` kind uses the daemon's EP/MoE-TP coordinates: an explicit `expert_id`
declares EP ownership, while a stacked expert tensor splits its leading expert
dimension over EP. Internal fragmentation does not imply process sharding.

Runtime manifests carry logical declarations separately from physical
fragments; worker metadata is stored once. The final conversion builds native
Mooncake descriptors and bindings once and uses
`WeightPlacementManifest.from_fragments` to assemble placements. Geometry is
validated by Mooncake rather than used to guess process ownership.

Tensors share a transfer group only when their source and target rank mappings
both match. This groups compatible GQA K/V tensors across layers without merging
different target replica arrangements. Fully replicated tensors can share the
ordinary TP placement; partial GQA replication projects onto TP and replica DP.
This DP coordinate describes weight copies, not SGLang engine DP.

The existing Mooncake split-axis contract requires a fragment on every split
rank. A vocabulary rank containing only padding is rejected before I/O; it is
not silently reclassified as replicated. Allocation registration and immutable
allocation guards retain their existing lifetime behavior.

## Storage ownership and derived state

Every registered parameter and buffer must be covered by transfer regions,
constant initializations, or an explicit `local_buffer(name)` declaration.
Only registered buffers can be declared local. For example, rotary caches are
constructor-generated and independent of checkpoint contents.

`b.initialize(tensor, byte_offset=..., nbytes=..., value=0)` describes padding
that the target initializes after all transfer groups have been planned. Source
padding never becomes network payload.

Use `b.derived("module.attribute", ...)` for plain tensor/scalar attributes that
the adapter reconstructs after transfer. Registered parameter storage must not
be replaced by finalization. Derived attributes needed by IPC must also be
handled by `derived_weight_state.py`; the existing DeepSeek attributes already
are. Merely declaring a plain attribute does not add it to IPC export.

The builder deduplicates exact aliases and rejects uncovered bytes, conflicting
regions, out-of-bounds geometry and storage changes. A changed layout with the
same shape is still a different representation: validate backend flags and
post-processing markers explicitly. Shape checks alone are insufficient.

## Tests

Run the unit and transfer contract tests from the repository root using the
environment containing the matching SGLang and Mooncake installations:

```bash
conda run -n manifest python -m pytest \
  test/registered/model_loading/test_weight_layout.py \
  test/registered/model_loading/test_mooncake_weight_adapter.py \
  test/registered/model_loading/test_weight_heterogeneous_transfer.py -q
```

The eight-GPU manual runner uses this same interpreter for all daemons and
engines. Pass local checkpoint paths, with BF16 weights for the reference
models. It uses isolated socket paths, releases comparison allocator caches
before starting IPC engines, cleans up only its own processes, and stops the
source before starting the disk-loaded reference when necessary.

```bash
conda run --no-capture-output -n manifest python \
  test/manual/test_weight_reshard_adapters.py \
  --qwen-small /path/to/Qwen3-0.6B \
  --qwen-large /path/to/Qwen3-8B \
  --deepseek /path/to/DeepSeek-V2-Lite \
  --output-dir /tmp/weight-reshard-validation
```

Use `--case` to select a case listed by `--help`. `--skip-inference` is available
for isolating state-transfer failures, but does not constitute full acceptance.
The full run checks exact exported state against independent native loading and
compares greedy token IDs/text and output log probabilities through IPC engines
at the same target topology (`rtol=atol=1e-3` for log probabilities).

Adapters duplicate some native-loader layout knowledge. Native-load comparison
is therefore required when either the loader or the adapter changes; checking
only a manifest's internal consistency cannot detect a wrong logical mapping.
