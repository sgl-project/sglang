Status: implemented

# Marlin + DeepEP

TL;DR: Run quantized MoE experts with Marlin while DeepEP moves tokens between
expert-parallel ranks. Support normal, low-latency, and auto modes using BF16
activations and the existing Marlin weight-loading and repacking paths.

## User contract

Select `--moe-runner-backend marlin --moe-a2a-backend deepep --dtype bfloat16`
with a Marlin-compatible quantized MoE checkpoint and an expert-parallel
configuration supported by the model. `--deepep-mode normal`, `low_latency`,
and `auto` retain their existing scheduling meanings; auto supports both
prefill and decode. Existing Marlin-compatible automatic runner selection
must reach the same integration when the resolved layer runner is Marlin.

The supported weight formats are those already handled by `MarlinMoeQuantInfo`
and the selected checkpoint's Marlin quantization method, including integer
GPTQ/AWQ and Marlin MXFP4/NVFP4 paths. Their existing shape, group-size,
activation, bias, and hardware restrictions still apply. This feature does
not expand the kernel's quantization contract.

DeepEP transports BF16 activations, with no activation scales. Resolve the
dispatcher dtype from the actual Marlin runner; reject an explicit incompatible
dispatcher dtype. Report unsupported combinations during initialization,
before the first dispatch, with the conflicting options in the error.

## Design and invariants

```text
router -> DeepEP dispatch -> local Marlin experts -> DeepEP combine -> output
             normal: received token rows, local expert IDs
        low latency: expert-major rows, per-expert valid counts
```

Register the `deepep` / `marlin` fused runner path and select its adapter by
dispatch format. Route Marlin layers through the common `FusedMoE` execution
path. Reuse the Marlin quantization payload and kernel; keep dispatch handles,
communication buffers, events, and combine ownership in the DeepEP dispatcher.
Normal combine must accept Marlin results independently of DeepGEMM enablement.

| Contract | Normal mode | Low-latency mode |
| --- | --- | --- |
| Marlin input | Received BF16 token rows and received local top-k IDs | Expert-major BF16 rows, respecting `masked_m` valid counts |
| Expert indexing | Use received local IDs directly; ignore `-1` routes | Derive local expert IDs from the expert-major dimension; ignore padding |
| Routing weights | Marlin applies received weights and reduces local contributions | Marlin computes unweighted expert outputs; DeepEP combine applies original weights |
| Combine input | One locally reduced result per received token, in received order | Expert-major results in exactly the layout expected by low-latency combine |

The low-latency adapter presents each valid expert row as a single-expert
Marlin invocation entry with unit routing weight, then restores the
expert-major output layout. Build IDs and masks on the GPU over fixed-capacity
buffers; valid counts must not cause host reads or dynamic allocations during
graph replay. Padding contributes zero and cannot index expert weights.

Apply each routing weight and the model's routed scaling factor exactly once.
Preserve the checkpoint's activation, gate/up ordering, clamping, bias, scales,
zero points, and activation-order metadata. NVFP4 applies its weight global
scale before adding bias, then applies routing weights. Shared-expert contributions retain
the model's existing ownership and are added once.

Weights and quantization metadata belong to the rank's local experts in the
same order used by dispatch. Never apply a global-to-local expert mapping a
second time. Empty ranks, empty experts, invalid routes, and zero-token batches
produce the required empty or zero outputs and still participate in matching
communication operations.

Inputs and communication buffers must not be overwritten while in use.
Marlin scratch storage must remain safe across layers, concurrent work, and
CUDA graph captures. Low-latency decode supports CUDA graph capture and replay;
normal mode retains the existing graph policy. The integration preserves
DeepEP event ordering and handle lifetime.

## Acceptance

- Compare distributed outputs against the same quantized weights and routing
  evaluated by standard Marlin and against a dequantized expert reference;
  use the existing format-specific kernel tolerances and record them in tests.
  Cover each supported quantization family, non-unit routing weights and
  scaling, top-k greater than one, and supported activation/bias variants.
- Exercise at least two EP ranks, multiple local experts, uneven routing,
  empty experts/ranks, invalid routes, and a supported TP+EP configuration;
  compare normal and low-latency results and test auto prefill-to-decode changes.
- Verify low-latency eager/captured parity across changing valid counts and
  repeated graph replay, plus clear initialization errors for excluded options.
  Run server prefill/decode smoke tests and standard Marlin regression tests.

Reproducible tests and component latency measurements live in
[evidence/feature-00-marlin-deepep.md](evidence/feature-00-marlin-deepep.md).
Validation uses deterministic quantized experts and two-rank H200 execution;
the serving smoke uses a local dummy AWQ model. Pretrained checkpoint loading,
language quality, and multi-node execution remain unmeasured.

## Boundaries

The supported scope is NVIDIA CUDA, BF16 activations, and the `deepep`
backend. FP16 activations, FP8/NVFP4 activation transport, `deepep_v2`, other
communication backends, `experimental_sgl_marlin`, LoRA, EPLB/elastic expert
placement, and fused shared-expert dispatch are outside this feature.
Configurations requiring down-GEMM communication overlap or single-/two-batch
overlap are also excluded initially and must fail validation when requested.
Separate shared-expert computation remains supported. No new quantization
formats, silent runner substitutions, or compatibility paths are introduced.
