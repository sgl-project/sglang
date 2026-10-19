# DiffusionGemma decision prefix regression

**Three frozen TweetEval requests return HTTP 400. This layer adds a reproducer;
the decoding fix is still pending.**

This follows the output logprobs PR (#41232) and decision API PR (#43752).
The manual script is not registered in CI while it demonstrates an unfixed bug.

## What fails

The decision adapter expects the first four canvas tokens to be
`[100, 45518, 107, 101]`: `<|channel>thought\n<channel|>`.
It reads candidate scores at position 4, immediately after that header.

For three requests, the fourth token is `**` (1018) or `<eos>` (1).
The adapter cannot trust the answer position, so its prefix check raises an
error. The API maps that error to HTTP 400. The request itself is valid.

The same bad output occurs on upstream commit
`20c49f12c9a1261ea0ba9b5a2b206613661494f0` without either PR.
This rules out a regression introduced by these two PRs for these cases.
It does not establish whether the cause is the checkpoint or SGLang's decoder.

[Raw outputs, upstream comparison, and stopping/step experiments](https://github.com/Hangzhi/sglang/tree/273838caccd07a03f17eb6065fca3877c23f6692/validation/diffusiongemma-2026-10-12/tweeteval-diagnosis).

## Reproduce

Use this stack's head and the prepared data from
[Hangzhi/diffusion-jev-sglang](https://github.com/Hangzhi/diffusion-jev-sglang/tree/a6dce88a2fc9c11b1df44d03033f1ebafd75c504).
The script checks hashes for both the question manifest and the 1,000-row test
file. It sends the three known failing requests once each, with no retries.

Start an eager server with the pinned checkpoint and sampler seed:

```bash
printf '%s\n' '{"seed": 42}' > /tmp/diffusiongemma-prefix-config.json
python -m sglang.launch_server \
  --model-path google/diffusiongemma-26B-A4B-it \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --served-model-name google/diffusiongemma-26B-A4B-it \
  --dllm-algorithm Gemma4Renoise \
  --dllm-algorithm-config /tmp/diffusiongemma-prefix-config.json \
  --trust-remote-code --dtype bfloat16 --attention-backend triton \
  --context-length 8192 --max-running-requests 1 --mem-fraction-static 0.85 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --no-dllm-fdfo --host 127.0.0.1 --port 30000
```

In another terminal, run from the SGLang repository root:

```bash
python test/manual/dllm/reproduce_diffusiongemma_prefix.py \
  --data-dir /path/to/diffusion-jev-sglang/data/emoji-benchmark/tweeteval \
  --output /tmp/diffusiongemma-prefix-results.json
```

**Current result: three HTTP 400 responses; exit code 1.** After the fix, require
three HTTP 200 responses with finite, normalized candidate probabilities.
This checks valid decisions, not whether the chosen emoji is correct.
The report records each request, response, and validation error.

The original reproduction used one RTX PRO 6000 Blackwell 96GB GPU, BF16, TP1,
and the default full 256-token canvas. Other hardware may produce different
outputs. A passing replay alone does not establish that the bug is fixed.

## Proposed work

- [ ] Compare the same checkpoint and prompts with Google's reference path.
      Check the chat template, special tokens, and denoising behavior.
- [ ] Fix any SGLang behavior that differs from the reference first.
- [ ] If unconstrained decoding still breaks the header, prototype a fixed
      prefix for decision scoring. Keep the four header positions fixed through
      each denoising step and self-conditioning update. Leave label tokens free.
      This is a proposal, not a validated fix.
- [ ] Keep the prefix check. Do not rewrite bad tokens after decoding, skip
      `<eos>`, or read a different score row to make a request pass.
- [ ] Verify score replay using the same prefix mode for raw generation and
      decisions. Add focused regression tests for the chosen implementation.
- [ ] Test eager execution, CUDA graphs, batches, streaming, and retraction.
      Confirm ordinary generation and other dLLM algorithms retain their behavior.
- [ ] Rerun all 1,381 frozen cases. Report valid responses, accuracy, and latency.
      Count failures as incorrect. Require all three prefix cases to pass without
      retries before accepting a decoder fix.
