# FlashLoop for Ouro (draft integration)

FlashLoop treats **recurrence as an additional redundancy axis** in looped
Transformers. Reusing parameters across loops does not by itself remove the
computation or KV traffic of repeated traversals. FlashLoop uses *lazy updates*
to reuse stable state and selectively refresh the state that changes.

- Paper: [FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates](https://arxiv.org/abs/2609.29812)
- Reference project: [Superone77/FlashLoop](https://github.com/Superone77/FlashLoop)

## Three independent components

1. **Token-sparse updates.** During prefill, rank tokens by relative hidden-state
   change across recurrences. Later loops process nested active-token sets;
   inactive hidden states and KV rows reuse the preceding loop's state. The last
   token of each request is retained, and attention uses the original causal
   positions.
2. **Sparse attention.** During decode, the second loop computes the source
   attention output, selects keys, and retains their probability mass and value
   contribution. Later loops replace that selected contribution with a newly
   computed contribution, preserving the cached complement.
3. **KV-residual quantization.** Store a packed INT4 anchor and recurrent
   residual streams instead of full BF16 caches for each loop. K is grouped
   across tokens and V across channels. Fused readers reconstruct requested
   tiles; a rolling BF16 tail handles newly generated tokens. Inactive prefill
   rows have explicit zero-residual flags.

Weights are shared across recurrent traversals; logical KV layer IDs are
separate. This implementation uses SGLang layers, scheduling, paged allocation
and attention rather than invoking a Hugging Face model forward.

## Try the draft

Use a source installation of this branch on Linux with an NVIDIA CUDA GPU and
an official local Ouro checkpoint. The current path supports four loops,
BF16 compute, equal Q/KV head counts and TP=PP=DP=DCP=1.

```bash
MODEL_PATH=/path/to/Ouro-1.4B
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --model-config-parser flashloop \
  --model-impl sglang \
  --dtype bfloat16 \
  --attention-backend triton \
  --tp-size 1 --max-running-requests 4 \
  --context-length 4096 \
  --json-model-override-args '{"flashloop_components":{"token_sparse_prefill":true,"sparse_decode":true,"kv_residual_quantization":true},"flashloop_prefill_fractions":[0.25,0.1],"flashloop_decode_fraction":0.1}'
```

Each boolean can be set to `false` independently. With no component override,
the explicit `flashloop` parser selects the dense Ouro path. Fractions denote
**retained** tokens or keys. Other models and the default `auto` parser are
unchanged.

The model-specific configuration disables prefix reuse, chunked prefill,
overlap scheduling and prefill CUDA graphs. INT4 KV selects 64-token pages and
requires a bounded request count. Speculative verification, weight quantization,
offload, unified KV arenas and distributed execution are outside this draft.
Packed capacity includes metadata, zero-residual flags and per-request BF16 tails.
Late-loop residual pages are not yet compacted by active-token count.

## Validation scope

The numerical tests cover paged attention, cached-mass correction, nested
selection, INT4 packing against an independent Torch oracle, inactive-state
reuse, request reuse, tail flushes and CUDA graph replay.

```bash
python test/registered/unit/configs/test_flashloop_config.py
python -m pytest -q test/manual/flashloop
```

This PR is an upstream integration draft. The standalone prototype was exercised
on SGLang 0.5.14; current-main server/model integration, broad accuracy validation
and performance profiling remain review gates. Sparse updates and quantization
can change outputs. This document makes no performance or accuracy claims.

Code adapted from FlashLoop retains its MIT notice in
[`LICENSE`](../../python/sglang/srt/layers/attention/flashloop/LICENSE).
