# DiffusionGemma serving measurements

`bench_diffusion_gemma.py` sends the same pretokenized chat prompts through each
runtime's completions API. It verifies prompt and completion lengths, rejects
empty responses, and records every trial, returned text, and prompt hash. Every
request has a different prefix; use the same run ID for every runtime.

Use the same checkpoint, physical GPUs, tensor parallel size, BF16 precision,
context limit, request limit, and memory budget. Run servers sequentially. Let
both runtimes compile and capture graphs, then discard the same number of client
warmups. Disable prefix caching for this workload, which has no reusable prefixes.

For a comparison that performs the same denoising work, use this algorithm YAML
with SGLang:

```yaml
max_denoising_steps: 48
confidence_threshold: 0.0
stability_threshold: 1
```

For vLLM 0.28.0, copy the checkpoint's `generation_config.json` into a separate
`fixed48-generation/` directory. Preserve all fields and set
`max_denoising_steps=48`, `confidence_threshold=0.0`, and `stability_threshold=1`.
Pass that directory with `--generation-config fixed48-generation`.

The DiffusionGemma sampler in this release reads the generation-config file;
`--override-generation-config` alone does not change its convergence settings.
Verify `ModelConfig.try_get_generation_config()` before accepting a fixed-work
comparison. The startup log's override dictionary is insufficient.

A zero confidence threshold disables early convergence. Keeping the normal
history size avoids adding unnecessary history processing to enforce a step
count. The temperature parameters remain `t_max=0.8` and `t_min=0.4`, with
entropy bound 0.1.
Both runtimes sample normally; they use different native random-number streams.

For the H200 TP2 comparison, SGLang uses:

```sh
python -m sglang.launch_server --model-path "$MODEL" \
  --host 127.0.0.1 --port 31400 --tp-size 2 --context-length 4096 \
  --max-running-requests 16 --mem-fraction-static 0.65 \
  --dllm-algorithm Gemma4Renoise --dllm-algorithm-config fixed48.yaml \
  --cuda-graph-config '{"decode":{"bs":[1,2,4,8,16]}}' \
  --enable-torch-compile --torch-compile-max-bs 8 --random-seed 42
```

Run both the default FDFO mode and `--no-dllm-fdfo` to expose the scheduling
tradeoff. Synchronous scheduling holds a batch until its denoising loop finishes.
It can reduce overhead for a batch that arrives together, while FDFO allows
requests to make progress independently.

The vLLM command uses:

```sh
vllm serve "$MODEL" --host 127.0.0.1 --port 31400 \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.65 --seed 42 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --diffusion-config '{"canvas_length":256,"max_denoising_steps":48}' \
  --generation-config fixed48-generation \
  --compilation-config '{"cudagraph_capture_sizes":[256,512,1024,2048,4096]}'
```

Set `CUDA_VISIBLE_DEVICES=0,1` and `OMP_NUM_THREADS=1` for both servers. Set
`VLLM_PLUGINS=''` when the environment contains unrelated plugins. Dependency
versions and the complete measured settings are recorded with the results.

For each runtime, run:

```sh
python benchmark/dllm/bench_diffusion_gemma.py \
  --url http://127.0.0.1:31400 --tokenizer "$MODEL" \
  --input-lengths 128 2048 --concurrencies 1 4 8 --output-length 256 \
  --denoising-steps 48 --warmups 3 --trials 15 \
  --run-id same-id-for-all-runtimes --result result.json
```

The `--denoising-steps` argument records the configuration; it does not configure
the server. The client disables EOS stopping to keep the work fixed. Reported
canvas tokens per second include positions after EOS and must not be described
as useful-text throughput. These synthetic, fixed-work tests do not establish
performance for adaptive stopping, continuous arrivals, or all model workloads.
Validate normal text, image, and multi-block generation separately.

## H200 TP2 results, 2026-09-09

These are median end-to-end wall times for a complete batch: three warmups,
15 measured trials per case, 256 canvas positions, and all 48 denoising steps.
The reference is [vLLM 0.28.0](https://github.com/vllm-project/vllm/releases/tag/v0.28.0),
the latest stable release checked on the measurement date. This does not compare
against unreleased vLLM main.

| Prompt tokens | Concurrency | Rebased PR | Optimized FDFO | Optimized sync | vLLM 0.28.0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 2.684 s | 0.810 s | 0.741 s | 0.569 s |
| 128 | 4 | 3.213 s | 1.521 s | 1.409 s | 1.281 s |
| 128 | 8 | 3.782 s | 2.561 s | 2.455 s | 2.226 s |
| 2048 | 1 | 2.702 s | 0.983 s | 0.915 s | 0.612 s |
| 2048 | 4 | 3.243 s | 1.843 s | 1.727 s | 1.430 s |
| 2048 | 8 | 4.056 s | 3.260 s | 3.422 s | 2.513 s |

Optimized FDFO is 1.24–3.31 times faster than the rebased PR across these cases.
vLLM has lower latency in every case, including against synchronous SGLang.
The optimization does not establish a serving performance lead.

The optimized revision is `b49ecee249`; both SGLang modes use decode graphs and
backbone compilation. The baseline includes the current request KV lifecycle fix
and support for a zero convergence threshold, but no performance changes. It can
be reconstructed from `687bc5d2fe` by cherry-picking `58046d6e4f` and `b49ecee249`.
All three use upstream base `db272201a2`.

All 1,872 requests, including warmups, have identical prompt hashes across the
four runtimes and the expected prompt and completion lengths. Servers ran
sequentially on physical GPUs 0 and 1, with unchanged power and clock limits.
Both environments use PyTorch 2.13.0+cu130 and Triton 3.7.1. Transformers differs
(5.12.1 for SGLang, 5.14.1 for vLLM), as does FlashInfer (0.6.17 versus
0.6.16.post3). Both use their native attention and communication implementations;
shared client tokenization removes tokenizer-version differences from the input.
Results come from one server session per configuration, in the order FDFO,
synchronous, baseline, and vLLM. Trial intervals do not measure variability
between server restarts.

[Machine-readable results](results/pr34061_h200_tp2.json) include launch settings,
software versions, source revisions, all measured wall times, and bootstrap
intervals for the median. The [compressed raw results](results/pr34061_h200_tp2_raw.json.gz)
retain every trial, prompt hash, usage count, and returned text. `MODEL_CHECKPOINT`,
`SGLANG_CHECKOUT`, and `ARTIFACTS` in the recorded commands denote local paths.
Exploratory runs with an incorrect graph mask or ignored vLLM generation
configuration are excluded from these results.

Validation on the optimized revision passed 264 tests and 62 subtests, including
CUDA sampling/RNG equivalence and fused QKV normalization checks. Real H200
serving smoke checks passed at TP1, TP2, and TP4: arithmetic, normal text, a
2,662-token prompt with multi-block generation, image OCR, and rejection of
unsupported request fields. TP2 also passed explicit synchronous scheduling.
These are functional smoke checks, not a model accuracy evaluation.
