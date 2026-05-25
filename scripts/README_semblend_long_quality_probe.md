# SemBlend Long-Context Quality Probe

This probe validates SGLang `SemanticEmbedding` fuzzy KV reuse on long
contexts. It measures cold-vs-warm TTFT and quality for the same variant
prompt, which is the right way to test whether fuzzy KV reuse changes output.
The prompts are answer-shaped: every sample ends with an explicit answer task
and the assistant turn. The task requires exactly three top-level bullets with
no headings or sub-bullets, so generated output should be an answer rather than
raw document continuation.

## Datasets

Two datasets are included because they answer different questions.

`quality_clusters_exactrun_8k_16k.json`

- Hit-friendly 8K and 16K clusters.
- Shows the current upstream-friendly `|exact|fuzzy|miss|` path can recover
  nearly the full long-context suffix.
- Each prompt ends with the same three-bullet answer task, so generated output
  should be a concise answer rather than article continuation.
- Expected 16K result on Qwen2.5-7B-Instruct-AWQ: `partial_80`,
  `partial_60`, and `paraphrase` should fuzzy-fire with roughly 15.9K cached
  tokens and large TTFT speedups. `diverse` should not fuzzy-fire.

`quality_clusters_fragmented_8k_16k.json`

- Real-dataset 8K and 16K clusters with more fragmented overlap.
- Shows why future multi-segment realization matters.
- Expected result today: SemBlend may discover many reusable regions
  internally, but SGLang realizes only one contiguous block. Cached-token
  counts may be low and warm TTFT may be slower than cold. Treat that as a
  current architecture limitation, not as a regression in exact-run reuse.

## Install

From the SGLang checkout:

```bash
pip install -e "python[all]"
pip install -U "semblend[onnx-gpu]>=0.3.11" aiohttp
```

The `onnx-gpu` extra is recommended for long contexts. Without
`sentence-transformers` and `onnxruntime-gpu`, donor embedding can fail or fall
back to CPU.

## Start SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8000 \
  --enable-fuzzy-match \
  --fuzzy-match-provider SemanticEmbedding \
  --fuzzy-model-arch qwen2.5-7b \
  --fuzzy-min-match-length 1 \
  --fuzzy-semantic-threshold 0.50 \
  --fuzzy-min-reuse-ratio 0.50 \
  --fuzzy-min-cached-tokens 1024 \
  --cache-fuzzy-results \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.70 \
  2>&1 | tee sglang-server.log
```

The probe parses `sglang-server.log` for fuzzy success/failure events. Use a
build that logs request ids in fuzzy events:

```text
Fuzzy match success: rid=... cached=... prompt=... offset=...
```

## Run The Exact-Run Probe

```bash
python scripts/run_long_quality_probe.py run \
  --endpoint http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --clusters scripts/quality_clusters_exactrun_8k_16k.json \
  --lengths 16000 \
  --overlap-types exact,partial_80,partial_60,paraphrase,diverse \
  --max-tokens 192 \
  --log-file sglang-server.log \
  --require-fuzzy-log-events \
  --raw-out long_quality_exactrun_raw.json \
  --out long_quality_exactrun_scored.json
```

Expected aggregate shape:

- `fuzzy_candidates.fuzzy_fire_rate`: near `1.0`
- `fuzzy_candidates.mean_cached_tokens`: around `15900` at 16K
- `fuzzy_candidates.mean_cache_fraction`: around `0.99`
- `fuzzy_candidates.ttft_speedup_ratio_of_means`: large positive speedup
- `negative_control.fuzzy_fire_rate`: `0.0`
- `exact_control` and `negative_control` quality scores should be `1.0` or very
  close because they compare the same prompt cold vs warm.
- `paraphrase` ROUGE-L / token F1 can be lower because the same answer may be
  worded differently; inspect the saved cold/warm responses or add a judge pass
  before treating a lexical delta as a quality regression.

## Run The Fragmented Probe

```bash
python scripts/run_long_quality_probe.py run \
  --endpoint http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --clusters scripts/quality_clusters_fragmented_8k_16k.json \
  --lengths 8192 \
  --overlap-types exact,partial_80,partial_60,paraphrase,diverse \
  --max-tokens 192 \
  --log-file sglang-server.log \
  --require-fuzzy-log-events \
  --raw-out long_quality_fragmented_raw.json \
  --out long_quality_fragmented_scored.json
```

Expected aggregate shape:

- `reuse_diagnostics` may show thousands of internally reusable tokens.
- `cached_tokens_in_warm` may stay much lower because current SGLang realizes
  one contiguous fuzzy block.
- With `--fuzzy-min-cached-tokens 1024`, low-value fragmented matches should
  be rejected instead of producing misleading slow warm runs.

## Re-score An Existing Run

```bash
python scripts/run_long_quality_probe.py score \
  --raw long_quality_exactrun_raw.json \
  --log-file sglang-server.log \
  --require-fuzzy-log-events \
  --out long_quality_exactrun_scored.json
```

If `--log-file` is omitted during `run`, or if the log does not contain
parseable fuzzy events, the script warns loudly. With
`--require-fuzzy-log-events`, it fails fast instead.
