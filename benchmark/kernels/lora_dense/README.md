# Dense LoRA tuning

`tune_plans.py` searches production `DensePlan` configurations, then rechecks
one locked finalist with fresh inputs. It measures a BF16, TP-local linear
site through `DenseLoraRunner.apply`, including the base GEMM, LoRA and a fresh
route build. It does not time a model, communication, quantized base GEMMs,
embedding lookups, the Inkling windowed sink, or absorbed MLA. Those paths
retain their runtime tests; pretending they are ordinary linears gives the
wrong tuning objective. MLA currently has no configurable table selector.

## New model / GPU

1. Read **instantiated local layer** dimensions, output slice widths, adapter
   rank/pool rank and TP from the intended serving configuration. Do not infer
   every LoRA target from the model's hidden size. List distinct local sites,
   token buckets, adapter distributions and eager/graph modes in a workload file.
2. On an otherwise idle GPU, search the enumerated space. Compilation and
   correctness checks precede timing. Each candidate has paired baseline
   measurements in alternating order; eager uses synchronized wall time and
   graph uses one complete forward per captured replay. No profiler totals
   or multiple-forward graphs are substituted for that objective.
3. Inspect held-out validation and selector conflicts. Keep the incumbent on
   invalid, inconclusive or losing validation; do not retry until a winner appears.
4. Review any proposed table domain, then compare the candidate and incumbent
   with the normal SGLang serving benchmark on the actual model. Check numerics
   and target/control cells, including shapes and ranks outside the tuned set.
   Only then promote a table. Kernel gains are not model TPS gains.

Minimal workload file (local geometry, one adapter and one base-only request):

```json
[
  {
    "name": "model.attention.qkv.tp4.decode2",
    "in_features": 2048,
    "slices": [1024, 256, 256],
    "tokens": 2,
    "rank": 16,
    "pool_rank": 64,
    "slots": 4,
    "request_lengths": [1, 1],
    "request_slots": [0, -1],
    "phase": "decode",
    "mode": "graph",
    "kind": "linear",
    "tp": 4
  }
]
```

```bash
python benchmark/kernels/lora_dense/tune_plans.py \
  --workloads workloads.json --out trial.json --check
python benchmark/kernels/lora_dense/tune_plans.py \
  --workloads workloads.json --out trial.json
```

Use `--candidates plans.json` for a declared list of production plan specs.
The built-in bounded search covers families, overlap, blocks and split-K,
plus one-axis tile changes around the incumbent. It is **not an exhaustive
joint search or a claim of global optimality**. A new architecture may need
additional legal candidates. Search candidates and failures remain in the report.

The report contains exact workloads, source/device identities, every paired
sample, the locked validation, and selected plan specs. No result overwrites
an earlier file or writes packaged configs. In particular, production tables
cannot distinguish adapter occupancy, execution mode or every model detail,
and `max_tokens`/`max_rank` are upper bounds, not exact keys. The report flags
contradictory selections for the same runtime key. It deliberately does not
turn sparse samples into wide production rules; inspect lower ranks, interval
boundaries and other workloads reached by each proposed rule before promotion.

`gen_dense_adapter.py` is a synthetic adapter fixture generator for serving
validation, not another tuning entry point. Use
`python -m sglang.bench_serving --help` for model throughput and latency, with
the same model, adapter bytes, node and graph recipe for both arms.

Design references: [SGLang's MoE tuning workflow](../fused_moe_triton/README.md),
[CUDA timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing),
and [Triton autotune state restoration](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
