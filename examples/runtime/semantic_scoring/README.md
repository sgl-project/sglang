# Semantic decision scoring

Use `/v1/score` or `Engine.score` for runtime-defined categorical decisions without
generating an answer. Any supported causal LM can supply the next-token scores;
the caller owns the prompt, candidate meanings, and decision policy. This is the
single-token decision pattern used by SemIf, not support for Jev's undisclosed model.

```python
result = engine.score(
    query=[],
    items=prompt_token_ids,             # one complete prompt per decision
    label_token_ids=[[32, 33], [34, 32, 33]],
    apply_softmax=True,
    temperature=1.5,
    return_token_logprobs=True,
)
```

The HTTP request uses the same fields. `Engine.async_score` is also supported.
Use `query=""` with string prompts, or `query=[]` with tokenized prompts. A flat
`label_token_ids` list applies to every item; nested lists specify each item's
candidates. Output rows preserve both item order and candidate order.

## Score semantics

- `apply_softmax=True` returns `softmax(candidate_logits / temperature)` per item.
  These are probabilities conditional on the supplied candidates, not calibrated
  confidence that the decision is correct. Fit temperature on held-out workload data.
- `token_logprobs`, when requested, contains the original full-vocabulary logprobs
  for those candidates, **not raw logits** and not temperature-scaled values.
  `softmax(token_logprobs / temperature)` gives the same candidate probabilities
  because the vocabulary normalizer cancels.
- The default `apply_softmax=False` preserves vocabulary probabilities. Non-unit
  temperature requires `apply_softmax=True` to avoid ambiguous semantics.
- Sequence-classification models can temperature-scale class softmax too, but do
  not support `return_token_logprobs`; their default scores remain class logits.
- Candidates must be distinct, valid single token IDs. Multi-token answer strings
  need sequence scoring, which this interface does not provide. Apply a chat
  template once, disable thinking when appropriate, and verify answer boundaries
  with the model's tokenizer. No template or letter-to-token mapping is hardcoded
  in the server.

Scoring uses `max_new_tokens=0`. Requests retain the normal scheduler and prefix
cache path. Submitting identical prefixes in one cold batch does not guarantee
one shared prefill. Hybrid models also need recurrent-state checkpoints at reusable
boundaries; do not equate their cache behavior with attention-only KV caching.
`--enable-mis` is a separate packed-scoring mode with backend restrictions, not a
generic switch for shared-state branching.

## SemIf client

See the [SemIf cookbook](../../../docs/cookbook/semantic-scoring/semif.mdx) for
client setup, JSONL inputs, result handling, and direct HTTP usage. This is a
direct-scoring adapter, not a backend registered in SemIf's CLI.

Run a current SGLang server on NVIDIA CUDA/Linux, using the same model revision
and dtype as the reference. This example uses one GPU and BF16:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --dtype bfloat16 --port 30000
```

In a separate client environment, install [SemIf](https://github.com/TheoLeeCJ/SemIf)
and `requests`. Keep its pinned tokenizer dependencies separate from the server.
From this SGLang checkout, run:

```bash
python examples/runtime/semantic_scoring/semif.py \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input /path/to/SemIf/examples/decisions.jsonl \
  --output decisions-sglang.jsonl
```

The client imports SemIf's prompt encoder, preserving its prompt hashes and token
boundary checks. Its output retains option IDs and probabilities but deliberately
uses `token_logprobs`, not SemIf's `option_logits`. `--temperature` applies a
previously fitted calibration value; it does not fit calibration automatically.
The client cannot enforce the server's checkpoint revision, so launch both with
the same revision. Compare decision agreement and probability error against the
reference; different kernels and batching can change BF16 results near a tie.
