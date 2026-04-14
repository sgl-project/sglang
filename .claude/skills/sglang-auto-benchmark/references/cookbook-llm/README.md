# Cookbook LLM References

These configs are derived from `sgl-cookbook` autoregressive text-model pages and normalized into the auto-benchmark config format.

Rules used here:
- Keep the baseline as close as possible to a pure-TP launch command.
- Keep `mem_fraction_static` and `schedule_policy` at the cookbook baseline by
  default; search higher-ROI knobs first.
- Move the remaining common performance knobs into `search_space`.
- Add `ep_size` search for relevant MoE pages.
- Keep CUDA graph enabled by default.
- Prefer cookbook H200 defaults first, then H100 defaults when H200 is not available; if neither exists, fall back to the cookbook's published baseline for that model and say so in the config comments.
- Default to synthetic `random` data so every config is runnable out of the box.
- Default to `dataset.num_prompts: 80` so the reference sweep stays cheap enough
  for interactive validation.
- Default to a coarse QPS search with `benchmark.qps.max_rounds <= 5`.
- Default to `search.tier: 2` so the shipped configs stay reasonably practical to run.
- Default to `search.max_candidates: 8` so the candidate sweep stays bounded by
  default.
- Default to `search.max_duration_hours: 12` because longer searches do not fit
  the intended workflow budget.
- Treat `dataset.input_len` and `dataset.output_len` as aligned scenario lists, not a cartesian product.
- If a candidate OOMs, the result table should recommend increasing GPU count or using GPUs with larger memory.

Default random scenarios in these configs:
- `1000 -> 1000` for a chat-like shape
- `8000 -> 1000` for a summarization-like shape

Each scenario should run a full search independently, and each scenario should have its own best launch command and summary table.

Excluded from this folder because they are OCR/VL-oriented rather than text-serving benchmark configs:
- DeepSeekOCR / DeepSeekOCR2
- GLMOCR
- GLM45V / GLM46V
- Qwen2.5-VL / Qwen3-VL
- Step3-VL-10B

Configs in this folder:
- `deepseek-v3.2.yaml`
- `deepseek-math-v2.yaml`
- `deepseek-r1-0528.yaml`
- `deepseek-v3.1.yaml`
- `deepseek-v3.yaml`
- `devstral-small-2-24b-instruct-2512.yaml`
- `ernie-4.5-21b-a3b-pt.yaml`
- `glm-4.5.yaml`
- `glm-4.6.yaml`
- `glm-4.7.yaml`
- `glm-4.7-flash.yaml`
- `glm-5-fp8.yaml`
- `gpt-oss-120b.yaml`
- `glyph.yaml`
- `intern-s1.yaml`
- `kimi-k2.5.yaml`
- `kimi-k2-instruct.yaml`
- `kimi-linear-48b-a3b-instruct.yaml`
- `llada2-1-mini.yaml`
- `ling-2.5-1t.yaml`
- `llama-3.1-70b-instruct.yaml`
- `llama-3.3-70b-instruct.yaml`
- `llama-4-scout-17b-16e-instruct.yaml`
- `llama-4-maverick-17b-128e-instruct-fp8.yaml`
- `mimo-v2-flash.yaml`
- `minimax-m2.5.yaml`
- `minimax-m2.1.yaml`
- `ministral-3-8b-instruct-2512.yaml`
- `mistral-small-4-119b-2603.yaml`
- `nemotron-3-nano-30b-a3b-bf16.yaml`
- `nemotron-3-super-120b-a12b-bf16.yaml`
- `qwen35-397b-a17b-fp8.yaml`
- `qwen3-coder-480b-a35b-instruct.yaml`
- `qwen3-coder-next.yaml`
- `qwen3-235b-a22b.yaml`
- `qwen3-next-80b-a3b-instruct.yaml`
- `ring-2.5-1t.yaml`
- `step-3.5-flash.yaml`
