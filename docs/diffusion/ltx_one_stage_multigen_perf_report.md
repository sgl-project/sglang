# LTX One-Stage Multi-Generation Performance Report

Date: 2026-05-26 UTC

## Objective

Compare LTX-2 one-stage multi-generation performance for:

- Serial baseline: four sequential OpenAI API requests with `n=1`
- Batched request: one OpenAI API request with `n=4`

The goal is to measure whether `n=4` is close to the latency of a single
generation, and to identify where the remaining time is spent.

## Benchmark Setup

Server:

- Model: `Lightricks/LTX-2`
- Entry point: `python -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve`
- GPU: one NVIDIA H200, selected through `CUDA_VISIBLE_DEVICES=6`
- Server mode: OpenAI-compatible `/v1/videos`
- `dit_cpu_offload=False`
- Default LTX layerwise offload remained enabled for `text_encoder` and `vae`
- Model loading time was excluded

Request:

- Prompt: `A small robot paints a red circle on a white wall while soft studio lights flicker.`
- Resolution: `256x256`
- Frames: `9`
- FPS: `8`
- Steps: `20`
- Seeds: `[42, 43, 44, 45]`
- CFG: default LTX-2 config, `guidance_scale=4.0`
- Output path: `/tmp/ltx_perf_compare_outputs_strict`
- Perf dump path: `/tmp/ltx_perf_compare_perf_strict`
- Raw JSON summary: `/tmp/ltx_perf_compare_report_strict.json`

Protocol:

1. Warm up once with `n=1`.
2. Warm up once with `n=4`.
3. Run three measured rounds.
4. In each round, run four sequential `n=1` requests with seeds `42,43,44,45`.
5. In the same round, run one `n=4` request with the same seed list.
6. Use API-reported `inference_time_s` as the primary timing metric.
7. Record client wall time, peak memory, stage timings, and denoising step timings.

Note: an initial attempt on GPU0 failed during warmup with CUDA OOM because that
GPU only had about 535 MB free after other processes. That failed attempt is not
included in the results below.

## Summary Results

| Metric | 4x `n=1` serial | 1x `n=4` batch | Interpretation |
| --- | ---: | ---: | --- |
| Mean inference time | 11.804 s | 5.219 s | Batch is 2.26x faster than serial |
| Min inference time | 11.720 s | 5.101 s | Stable across rounds |
| Max inference time | 11.853 s | 5.296 s | Stable across rounds |
| Stddev | 0.073 s | 0.104 s | Low variance |
| Mean client wall time | 16.012 s | 8.006 s | Includes polling overhead |
| Mean peak memory | 48187 MB | 48180 MB | Effectively unchanged |

Throughput view:

- One single request average latency inside the serial baseline:
  `11.804 / 4 = 2.951 s`
- One `n=4` request average latency:
  `5.219 s`
- Per-output latency in the `n=4` request:
  `5.219 / 4 = 1.305 s`
- Throughput speedup:
  `11.804 / 5.219 = 2.26x`

So `n=4` is not expected to match `n=1` latency. It is one request producing
four samples, and the denoising batch is larger. The useful comparison is
throughput against four serial single-sample requests, where this run shows a
2.26x improvement.

## Round-Level Results

| Round | 4x `n=1` inference | `n=4` inference | Speedup | Serial peak mem | Batch peak mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.720 s | 5.101 s | 2.30x | 48206 MB | 48180 MB |
| 1 | 11.853 s | 5.296 s | 2.24x | 48178 MB | 48180 MB |
| 2 | 11.840 s | 5.259 s | 2.25x | 48178 MB | 48180 MB |

The three rounds are consistent. The measured speedup stays in a narrow
`2.24x - 2.30x` range.

## Stage Breakdown

Mean stage durations across the three measured rounds:

| Stage | 4x `n=1` serial | 1x `n=4` batch | Serial / batch |
| --- | ---: | ---: | ---: |
| `InputValidationStage` | 0.3 ms | 0.1 ms | 3.37x |
| `TextEncodingStage` | 1697.1 ms | 415.8 ms | 4.08x |
| `LTX2TextConnectorStage` | 113.0 ms | 23.4 ms | 4.83x |
| `LTX2SigmaPreparationStage` | 0.6 ms | 0.2 ms | 3.94x |
| `TimestepPreparationStage` | 8.8 ms | 2.3 ms | 3.90x |
| `LTX2AVLatentPreparationStage` | 0.9 ms | 0.4 ms | 2.51x |
| `LTX2AVDenoisingStage` | 9691.3 ms | 4628.2 ms | 2.09x |
| `LTX2AVDecodingStage` | 229.2 ms | 134.7 ms | 1.70x |
| `LTX2ImageEncodingStage` | 0.1 ms | 0.0 ms | 4.13x |

The main bottleneck is still denoising:

- Serial denoising accounts for about `9691 / 11804 = 82.1%` of total inference time.
- Batch denoising accounts for about `4628 / 5219 = 88.7%` of total inference time.

Non-denoising stages, especially text encoding and connector work, scale close
to 4x because the batched request avoids repeating the same prompt-side work
four separate times. Denoising only improves by about 2.09x because the model
still performs a larger batched transformer pass.

## Denoising Step Breakdown

The run used 20 denoising steps.

| Metric | 4x `n=1` serial | 1x `n=4` batch | Ratio |
| --- | ---: | ---: | ---: |
| Mean total denoise step time | 9669.7 ms | 4622.0 ms | 2.09x |
| Mean time per denoise step | 483.5 ms | 231.1 ms | 2.09x |
| Mean single-request step time | 120.9 ms | N/A | N/A |

For a single `n=1` request, one denoising step is about `120.9 ms`. For one
`n=4` request, one denoising step is about `231.1 ms`. That means the batched
step is about `1.91x` slower than one single-sample step, while producing four
samples.

This is the direct reason `n=4` total latency is not close to `n=1` latency:
the per-step model call is larger. But it is still much better than doing four
separate model passes.

## Why It Is Not 4x

With CFG enabled, each denoising step uses both negative and positive branches.
The effective model batch changes like this:

- `n=1`: effective CFG batch is roughly `2`
- `n=4`: effective CFG batch is roughly `8`

The scheduler migration and CFG batch fix make this path functionally valid,
but they do not make the transformer compute free. The batched request saves
overhead and improves GPU utilization, but the dominant denoising work still
scales with batch size.

In this benchmark:

- Prompt-side work scales very well because it is reused/expanded inside one
  request instead of recomputed across four requests.
- Denoising dominates total runtime and scales to about 2.09x faster than
  serial, not 4x.
- Peak memory is effectively the same in the API metric, likely because the
  reported peak is dominated by steady model memory and allocator behavior at
  this small resolution.

## Conclusion

The current LTX-2 one-stage multi-generation path is working and provides a
clear throughput gain:

- `n=4` is about `2.26x` faster than four sequential `n=1` requests.
- `n=4` is about `1.77x` slower than one `n=1` request, but produces four
  outputs.
- Peak memory is effectively unchanged in this small 256x256, 9-frame test.
- The remaining latency is primarily denoising, not scheduler overhead or
  request-side setup.

This means the observed behavior is expected for the current implementation:
multi-generation is batched generation, not four free samples at single-sample
latency. The scheduler native migration and CFG batch fix enable the path, while
performance is now mainly bounded by batched transformer denoising.

## Limitations

- The GPU was not fully isolated. The measured rounds were still stable, but
  other processes were present on the machine.
- Only `Lightricks/LTX-2` was benchmarked here; `LTX-2.3` was not rerun in this
  performance pass.
- Only the default CFG path was measured. A separate no-CFG control run would be
  useful if we want to isolate the exact CFG overhead.
- The test uses a small request shape: `256x256`, `9` frames, `20` steps. Larger
  videos may shift the denoising and decoding balance.
