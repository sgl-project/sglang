# Model-native trie beam search

A compatible model bundle may contain `trie_output_head.safetensors`, a
three-codebook semantic-ID trie. The artifact is static model state, not a
second learned language-model head. The implementation reuses the original
LM-head weights.

The artifact contains `root_token_ids`, `level1_token_ids`, `level1_offsets`,
`level2_token_ids`, and `level2_offsets`, together with the schema metadata
validated by `trie_config.py` and `trie_constraint.py`.

Full-vocabulary probability normalization is preserved by default.
`SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB=true` instead normalizes over retained legal
candidates. That is an explicit approximation and can change rankings; it must
not be used to claim full-vocabulary score equivalence.

## Optional fast paths

`SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION=true` enables shared-prompt
attention for supported FlashInfer MHA FP16/BF16 KV caches. Unsupported
backend/cache configurations retain ordinary attention. Attention logit caps
are not supported by the shared-context kernel.

Requests retain their own beam widths. Mixed-width batches use the eager
width-bucketed shared-context path. Uniform batches can additionally replay
CUDA graphs when their width and exact row count match the capture configuration:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `SGLANG_BEAM_SHARED_CONTEXT_WIDTH` | 2000 | Beam width of graph-compatible groups |
| `SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT` | 32768 | Context-slot capacity per captured group |
| `SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE` | 8 | Private decode-slot capacity per beam |

Configure decode capture buckets to include the exact total beam-row counts
needed by the workload. A width-1000 workload, for example, needs buckets such
as 1000, 2000, and 3000 rather than padding a shared-context batch to an
incompatible row layout. Different per-request decode lengths are supported
within a uniform-width batch: private slot arrays are padded and actual lengths
are refreshed for each group on every replay.

`SGLANG_ENABLE_BEAM_TRIE_SELECTION_GRAPH=true` separately replays the existing
compact-trie selection tensor operations. It is off by default. Requests with
other widths, stop-token handling, or unsupported layouts retain the native
selection path. Every replay restages its inputs and snapshots active tokens,
parents, scores, and prefixes into request-owned storage.

Normal and speculative decoding retain their ordinary graph keys and capture
preparation interfaces. Trie graph inputs and output keys are separate, and
changing a captured scoring/shared-attention configuration rejects stale
replay. Changing startup-only controls requires restarting or recapturing the
engine; flags must not be changed to reinterpret an existing graph.

## Correctness and measurement

Verify the original full trie and full-vocabulary score policy independently
of any compatible synthetic-catalog throughput control. Check ordered
semantic IDs, sequence scores, membership, uniqueness, output counts, and
output lengths. Report set overlap or Jaccard similarity separately from
ordered agreement.

CUDA attention and matrix-multiplication reductions can produce different
finite-precision trajectories when batch geometry changes. Controlled replay
equality is not a promise of bitwise equality across arbitrary asynchronous
batches.

`max_running_requests` is sequence-row capacity, not client concurrency.
A beam-width-K request uses approximately K decode rows. Report this setting,
actual graph-path coverage, and whether measured latency includes queueing.
