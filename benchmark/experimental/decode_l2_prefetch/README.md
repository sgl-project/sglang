# Experimental decode L2 weight prefetching

This directory preserves an opt-in research patch for Qwen3.5 dense models.
It is not enabled by SGLang, and is not a production integration. The Python
startup hook only installs it when this directory is explicitly on PYTHONPATH
and SGLANG_DECODE_L2PF is nonzero.

The executable algorithm is the original 16/32 MiB, lazy-join experiment.
It does not include the smaller-budget, nearest-GEMM-only or eager-join variants.
Comments and formatting have been updated to distinguish hypotheses from evidence.

## Mechanism

1. Before CUDA graph capture, register forward pre-hooks on embedding, attention
   output projection, and non-final MLP down projection modules.
2. Each hook records the weight modules expected after the upcoming collective.
   Plans contain contiguous CUDA parameters/buffers in module order, deduplicate
   identical starting pointers, and stop at the selected byte budget.
3. The AllReduce wrapper forks a side CUDA stream from the current stream.
   That stream issues read-only weight prefetch kernels while the original
   AllReduce runs on the current stream.
4. In the default lazy mode, the next GEMM does not wait for the prefetch kernel.
   The side stream rejoins at the next AllReduce or model-forward exit.

The weight allocation remains in GPU device memory. GPU L2 holds cached lines;
this does not involve the CPU L2 cache. Normal GEMM loads already use GPU caches.
The hypothesis is that fetching some lines earlier can hide part of the later
memory access cost. It is not a guarantee of cache hits or reduced latency.

The default bulk path uses cp.async.bulk.prefetch.L2.global. It does not set
evict_last; that policy applies only to the alternative load-based kernel.
A CUDA event after the bulk kernel orders kernel execution, but does not prove
completion of all asynchronous cache-fill traffic or continued cache residency.

## Tested configuration and scope

Historical service experiments used SGLang 0.5.22 with qwen3.8-27b-nvfp4,
using the Qwen3.5 model implementation, NVFP4 weights, FP8 KV cache,
two RTX 5090 GPUs, TP=2, and DFlash.
RTX 5090 device memory is GDDR7. Do not describe these runs as HBM measurements.

The patch is model-specific and globally wraps GroupCoordinator.all_reduce and
Qwen3_5ForCausalLM.forward within each worker process. It assumes a single target
model and the expected dense projection attributes. Draft workers are not armed.
Other models, MoE, pipeline parallelism, concurrent model forwards, CPU/ROCm,
and dynamic weight replacement have not been validated. The bulk instruction
requires an NVIDIA architecture supporting it (SM90 or newer).

The row-count gate is not a forward-mode check: small prefill inputs may also
trigger prefetching. Plans retain tensor views and assume weight storage remains
stable after installation. Invalid custom budgets are not validated by this
research patch; use the defaults below. Large arbitrary budgets can exceed the
fixed sink capacity. Install only once in a fresh worker process.

End-to-end serving smoke validation also passed on upstream main commit
5007cbd6a57515bf5e008ff2fd2bdb834356b53c with the configuration above, FlashInfer
0.7.2, and sglang-kernel 0.4.10. Native and original-patch runs each completed
8 requests at concurrency 1, 4, and 8, generating exactly 128 tokens per request
(48/48 successful requests total). All 24 paired output-token sequences matched,
and all 6,144 returned output-token log probabilities were finite. Both target
workers armed 128 AllReduce sites and recorded 3,200 prefetch calls with zero
skips during graph capture; decode logs confirmed CUDA Graph execution.
This is a serving compatibility check with eight short prompts, not a repeated
performance or broad answer-quality evaluation. Historical throughput results
below remain specific to the 0.5.22 experiments.

## Reproduce

Use an isolated environment with the tested SGLang version, PyTorch, and Triton.
Keep the patch directory available to spawned worker processes. From this
repository root, add the following prefix to an otherwise identical launch:

    CUDA_VISIBLE_DEVICES=0,1 \
    PYTHONPATH="$PWD/benchmark/experimental/decode_l2_prefetch" \
    SGLANG_DECODE_L2PF=1 \
    SGLANG_L2PF_BYTES_SMALL=16777216 \
    SGLANG_L2PF_BYTES=33554432 \
    SGLANG_L2PF_JOIN=lazy \
    python -m sglang.launch_server --model-path /path/to/qwen3.8-27b-nvfp4 --tp 2 <same-serving-options>

If your environment already needs PYTHONPATH, prepend the patch directory to
its existing value. Use GPUs allocated to your experiment. Keep all model,
quantization, speculative decoding, cache, graph, seed, and serving options
identical between arms. For a native control, launch a fresh process with
SGLANG_DECODE_L2PF=0. Changing the variable after startup does not uninstall it.

Confirm both "L2PF installed" and "L2PF armed" in each target worker log.
Python sitecustomize errors can be printed without terminating startup;
a healthy endpoint alone does not establish that the patch is active.
The printed Python counters reflect calls during capture/execution, not the
number of GPU kernels replayed by a CUDA graph. Inspect a trace to verify replay.

| Variable | Default | Meaning |
| --- | --- | --- |
| SGLANG_DECODE_L2PF | 0 | Explicit installation switch |
| SGLANG_L2PF_MAX_TOKENS | 512 | Maximum AllReduce input rows |
| SGLANG_L2PF_SMALL_TOKENS | 16 | Small-budget row threshold |
| SGLANG_L2PF_BYTES_SMALL | 16777216 | Small-step byte budget |
| SGLANG_L2PF_BYTES | 33554432 | Larger-step byte budget |
| SGLANG_L2PF_KIND | bulk | Bulk hint; ld selects load-based prefetch |
| SGLANG_L2PF_JOIN | lazy | Deferred join; ar joins after AllReduce |

For kernel-only validation (no model download or service):

    CUDA_VISIBLE_DEVICES=0 SGLANG_DECODE_L2PF=0 \
    python benchmark/experimental/decode_l2_prefetch/test_kernels.py

## Performance evidence and limitations

A historical full A/B/B/A comparison used two cache-enabled chat workloads,
concurrency 4/8/16, 1,000 output tokens per request, and 3,360 successful formal
requests. A was this original patch; B used 4/8 MiB and only the nearest GEMM.
Both used lazy joining. Aggregate throughput used total generated output tokens
divided by the sum of complete formal request-group wall times, excluding startup,
cache flushes, and warmup.

| Workload | Concurrency | B throughput relative to A |
| --- | --- | --- |
| Single-turn | 4 | -2.80% |
| Single-turn | 8 | -2.11% |
| Single-turn | 16 | -1.46% |
| Multi-turn | 4 | -4.54% |
| Multi-turn | 8 | -1.98% |
| Multi-turn | 16 | -4.74% |

This compares two prefetch policies, not prefetch against native SGLang.
Single-turn paired inputs matched, but generated outputs usually differed.
Multi-turn feeds actual generated answers into subsequent requests, so subsequent
inputs also differed. Two repeats per policy do not establish significance or
answer-quality equivalence. The table is historical evidence, not an upstream
main benchmark or a generally applicable speedup claim.

Separate short traces measured 56.0% of original prefetch kernel time overlapping
AllReduce and 22.9% overlapping GEMM. These are kernel-interval overlaps, not
cache-fill completion measurements. The native-before/native-after throughput
drift in that experiment was 2.8%, larger than the measured original-patch gain.
Do not claim stable end-to-end acceleration from that small difference.

For further validation, retain failed requests and record complete wall time,
TTFT, TPOT, output lengths, cache usage, paired input/output agreement, and
answer-quality results. Include native controls and repeat runs. Profile
separately from timed runs. Device-wide bandwidth headroom or kernel overlap
alone does not establish that AllReduce is PCIe-latency-bound or that prefetching
is free: GEMM, communication, and prefetching can compete for memory bandwidth,
L2 bandwidth/capacity, and execution resources.

Raw requests, model assets, and traces remain outside the repository.
