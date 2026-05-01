# Cookbook Configs

`configs/cookbook-llm/` is the model-specific starting set for this skill.

The files define a shared LLM serving cookbook model list. Some SGLang entries
are seeded from prior serving benchmark recipes and can preserve those launch
flags where applicable, except for sequence limits that are too small for the
shipped two-scenario synthetic workload. vLLM and TensorRT-LLM do not reuse
SGLang flag names. They use their own serving CLIs and a matching set of
batching, prefix-cache, and sequence-limit knobs.

Each config uses `source.kind: llm_serving_cookbook`; use
`source_recipe_file` only to record the recipe filename that seeded the entry.

## Translation Rules

- Keep the same model, tokenizer, GPU count, dataset scenarios, benchmark SLA,
  and bounded candidate budget across all three frameworks.
- Keep memory fractions in `base_server_flags` by default.
- Keep TensorRT-LLM pinned to `trtllm-serve serve --backend pytorch`; never
  place `backend` in its `search_space`.
- Do not copy SGLang-only parser, scheduler, or attention flags into vLLM or
  TensorRT-LLM unless the target framework exposes a matching server flag.
- Raise `context_length`, `max_model_len`, or `max_seq_len` when needed so every
  candidate can cover the largest default scenario.

The vLLM and TensorRT-LLM sections are framework-native translations, not a
claim that every upstream project publishes a model-specific recipe for every
SGLang cookbook model. vLLM maintains recipe pages for selected model families,
while TensorRT-LLM publishes model recipe commands and optional LLM API config
files for selected popular models. Treat those upstream recipes as extra input
when they exist, then run this skill's validator against the exact target CLI.

## Validation

Run the local schema and flag check:

```bash
python .claude/skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
  .claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm
```

When you have captured the target container help output, validate concrete flag
names too:

```bash
python .claude/skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
  --help-dir artifacts/help \
  .claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm
```

Expected help file names can be flexible, but they should include these words:

- SGLang server help: `sglang` and `launch`
- vLLM server help: `vllm` and `serve`
- TensorRT-LLM server help: `trtllm` and `serve`

This validation only reads YAML, checks flag names, and renders candidate server
commands. It does not start a model server.

## Config Count

The current set contains 38 text-serving configs in the shared LLM serving
cookbook.
