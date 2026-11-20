# DeepSeek-V4 row-wise prefill pipelines

This directory accompanies the experimental DeepSeek-V4 prefill pipeline implementation. It contains the [technical report](REPORT.md), six academic figures, and compact historical measurements.

The report distinguishes three mechanisms:

- **Row-wise MoE Pipelining (RMP):** divide CP-local token rows and pipeline AllGather, routed expert computation, and ReduceScatter.
- **Epilogue-Pipelined MoE (EPM):** extend preparation to include the attention output projection and mHC epilogue, so the first gather can be submitted before all row slices are prepared.
- **Cross-Layer Query Lookahead (QLA):** compute the next layer's row-wise query front only after the corresponding current-layer rows have reached their final state.

Historical experiment identifiers `C004`, `C007`, and `QAHEAD` map to RMP, EPM, and QLA, respectively. The archived figures keep their original experiment labels to preserve the evidence record.

**Validation scope:** the included GPU measurements were collected from a historical implementation on 2026-11-03. They do not benchmark the port onto upstream main. The report records the historical source hash, input protocol, scope differences, and missing measurements. The current upstream `pre-from-prev` execution path retains its PP1 restriction; this PR does not remove it. The historical CP4/EP4/PP2 result does not establish PP2 support for the port. Validate accuracy and speed on the target main revision before enabling this experimental path in a deployment.

## Experimental controls

All controls default to the normal execution path. They are feature-selection controls, not a complete model launch command.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `SGLANG_DSV4_MOE_MICROBATCHES` | `0` | Number of row partitions; zero disables the pipeline |
| `SGLANG_DSV4_MOE_MIN_ROWS` | `1024` | Minimum CP-local rows for the large-row path |
| `SGLANG_DSV4_MOE_PACKED_GATHER` | `0` | Gather locally routed, packed MXFP8 input |
| `SGLANG_DSV4_MOE_EPILOGUE_PIPELINE` | `0` | Include eligible output projection/mHC preparation in EPM |
| `SGLANG_DSV4_QUERY_LOOKAHEAD` | `0` | Precompute eligible next-layer query fronts after row finalization |
| `SGLANG_DSV4_MOE_COMM_PRIORITY` | `0` | Priority of the dedicated MoE communication stream |

The initial port is restricted to eligible CUDA SM120 MXFP4 FlashInfer CUTLASS prefill paths with CP greater than one, attention DP1 and attention TP1. Unsupported paths use normal execution. The port excludes speculative execution, current graph capture, EPLB/redundant-expert/recorder paths, waterfill, vision routing, and unsupported fusion/layout combinations. Shared experts must be absent or replicated TP1; packed-input reuse also requires a compatible MXFP8 view on the shared gate/up FlashInfer CUTLASS or CuTe-DSL backend. EPM requires packed gather and eligible BF16 O-A/O-B projection paths. Query lookahead adds next-layer layout, fusion, same-stage, Engram, and bounded-tail guards. Consult the production eligibility predicates for the complete current checks.

The gate requires `hc_stats_stream is None`. On configurations where the model creates this stream through the default multi-stream-overlap option, set `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` before model construction to allow the row-pipeline gate to match. This does not override any of the other eligibility checks or guarantee a performance gain.

Historical measured controls conceptually correspond to four partitions with packed gather, EPM and lookahead selected, and communication priority `-1`. This mapping is not a measured main-branch launch recipe. In particular, do not use it to infer support for the historical PP2 configuration on the port.

## Contents

| File | Purpose |
| --- | --- |
| [REPORT.md](REPORT.md) | Implementation mechanism, measurement method, observations, and limitations |
| [data/measurements.json](data/measurements.json) | Full precision summaries, 10 paired trial inputs per configuration, native MFU, and independent core UT results |
| [figures/data/frozen_inputs.json](figures/data/frozen_inputs.json) | Complete plotting input, including native timeline coordinates and grouping |
| [figures/build_academic_figures.py](figures/build_academic_figures.py) | Offline scientific plotting script |
| [figures/requirements.txt](figures/requirements.txt) | Plotting dependencies |

## Regenerate the figures

Use Python 3.12 and a separate plotting environment:

```bash
python3 -m venv /tmp/sglang-row-pipeline-plots
/tmp/sglang-row-pipeline-plots/bin/pip install -r figures/requirements.txt
/tmp/sglang-row-pipeline-plots/bin/python figures/build_academic_figures.py
```

The script reads only the bundled frozen inputs. It does not require a model, a GPU, a service, network access during plotting, or the original traces. It creates editable-text SVG, embedded-font PDF, and 600 dpi PNG exports at a fixed width of 166 mm. It checks text bounds, English annotations, SVG text retention, and PDF font embedding. Generated PDF, PNG, and manifests are ignored by Git; the report links to the SVGs.

Paths beginning with `evidence/` in the input bundle identify historical evidence by a public identifier and SHA-256; they are provenance references, not files that the plotting script opens. Raw service logs and machine-specific paths are not included.
