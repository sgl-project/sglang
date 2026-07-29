# Cookbook LLM Configs

These configs define a framework-neutral LLM serving cookbook model set and translate each model into a three-framework run plan for SGLang, vLLM, and TensorRT-LLM.

Scope:
- SGLang can preserve source-recipe `base_flags` and `search_space` where applicable; if a sequence limit is smaller than the default synthetic scenario, the config raises that limit so the shipped workload can run.
- vLLM uses framework-native `vllm serve` flags. The translation keeps the same model, tokenizer, dataset shape, GPU count, and high-impact batching/prefix-cache knobs; it does not copy SGLang-only parser or scheduler flags.
- TensorRT-LLM uses `trtllm-serve serve` with `backend: pytorch` fixed in `base_server_flags`. Backend choice is never searched.
- The two default random scenarios remain aligned pairs: `chat` uses `1000 -> 1000`, and `summarization` uses `8000 -> 1000`.

Before a real run, capture the target framework `--help` output and validate the configs:

```bash
python .claude/skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py   .claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm
```

With captured help files, add `--help-dir <artifact-help-dir>` to check the concrete flag names against that environment. This check only loads configs and renders candidate commands; it does not launch model servers.
