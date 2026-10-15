# DiffusionGemma serving measurements

`bench_diffusion_gemma.py` exercises a running server through its completions API using pretokenized chat prompts. For completed cases it records trial timings, returned text, prompt hashes, and usage counts. It rejects empty responses and unexpected token counts.

Start the server with the checkpoint and supported serving defaults:

```sh
python -m sglang.launch_server --model-path "$MODEL" \
  --dllm-algorithm Gemma4Renoise
```

Point the client at that server and the same tokenizer:

```sh
python benchmark/dllm/bench_diffusion_gemma.py \
  --url http://127.0.0.1:30000 --tokenizer "$MODEL" \
  --input-lengths 128 2048 --concurrencies 1 4 8 --output-length 256 \
  --run-id serving-check --result result.json
```

## Interpretation

The client disables EOS stopping to request a fixed output length. Canvas positions after EOS count toward the reported token rate, which is not useful-text throughput. `--denoising-steps` records metadata and does not configure or verify the server's convergence settings. Its default is unset. Set it only when the actual server work has been verified. If a run fails, keep both server and client logs, since the result file contains only completed cases.

For comparisons, freeze the protocol before collecting results. Match checkpoint identity, physical hardware, precision, prompts, resource limits, generation settings, and stopping policy. Use the same run ID to generate identical prompts. Give each runtime equivalent access to its normal compilation and graph features, preserve every trial and failed request, and disclose dependency and implementation differences. Validate normal text, images, mixed requests, and multi-block generation separately.

Performance for the merged revision remains unmeasured. Earlier measurements and test counts do not validate this revision. A fixed-work experiment does not establish adaptive-generation performance, model accuracy, or a serving performance lead.
