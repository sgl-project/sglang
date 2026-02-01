
## Perf baseline generation script

`python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py` starts a local diffusion server, issues requests for selected test cases, aggregates stage/denoise-step/E2E timings from the perf log, and writes the results back to the `scenarios` section of `perf_baselines.json`.

### Usage

Update a single case:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --case qwen_image_t2i
```

Select by regex:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --match 'qwen_image_.*'
```

Run all keys from the baseline file `scenarios`:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --all-from-baseline
```

Specify input/output paths and timeout:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --baseline python/sglang/multimodal_gen/test/server/perf_baselines.json --out /tmp/perf_baselines.json --timeout 600
```
