# NPU CPU-offload policy regression

Run from the SGLang repository root:

```bash
python test/manual/npu_cpu_offload/test_cpu_offload_policy.py --source python/sglang/srt/utils/offloader.py
```

The four tests execute the actual OffloaderV1 class extracted from source with fake allocation boundaries. They check NPU pinning, unchanged non-NPU policy, layouts, parameter-budget behavior, zero budget and already-CPU parameters. They require no torch or accelerator, and do not test real forward execution, copies or graph capture.

Historical real Qwen3-ASR-0.6B validation used SGLang 0.5.20, torch/torch_npu 2.10 and CANN 9.0. Offload 0/4 each completed 20 requests with graphs disabled, and 20 requests each with prefill/decode graphs enabled and torch.compile explicitly disabled. In the latter pair all responses matched, both scored 2/216 word edits, and loading memory was 1.63/0.78 GB. The pin-only source SHA256 is 9013301d64d73f1c1a3ac04797bd3744f15792b4d9d5d5302d2b54e7380d1f7b. Original source and failure records remain preserved locally.

Do not infer torch.compile compatibility from these results. The graph-disabled YAML retained a compile setting, but graph compilation was not demonstrated in that configuration. The graph-enabled passing profile explicitly used enable_torch_compile=false. Default compile/graph problems are tracked separately from this basic offload fix.
