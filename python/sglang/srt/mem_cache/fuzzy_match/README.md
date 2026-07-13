# Fuzzy KV Cache Reuse (`--radix-cache-backend fuzzy_match`)

Semantic KV cache reuse for prompts that share meaning but not tokens.
When exact prefix matching leaves part of a prompt uncovered, a pluggable
`FuzzyMatchProvider` may nominate donor KV from a previously finished
request; the donor KV is position-corrected (RoPE) into recipient-owned
slots before the forward pass. Reuse follows the `|exact|fuzzy|miss|`
prompt decomposition: one contiguous fuzzy span anchored at the exact
prefix boundary.

## Enabling

```bash
pip install "sglang[fuzzy-semantic]"   # installs the semblend provider dependency
                                       # https://github.com/WorldFlowAI/semblend

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --radix-cache-backend fuzzy_match \
  --fuzzy-model-arch qwen2.5-7b
```

Selecting the backend enables the feature; the default provider is
`SemanticEmbedding`. Reuse markers in the server log: `fuzzy match success`
(match accepted), `[FUZZY] Realized N fuzzy tokens` (KV copied +
RoPE-corrected), and `#fuzzy-token: N` in the prefill batch line.

## End-to-end call chain

One request, from arrival to donor registration. Ownership of locks,
allocation, and RoPE is called out at each step.

```
Scheduler admits request
  └─ Req.init_next_round_input                    (managers/schedule_batch.py)
       └─ FuzzyRadixCache.match_prefix(params)    (fuzzy_radix_cache.py)
            1. RadixCache.match_prefix (super)  -> exact device_indices, last_node
            2. Gate: provider configured AND params.req is not None AND
               exact < total AND the miss suffix is at least
               fuzzy_min_suffix_tokens (short suffixes cannot amortize the
               semantic lookup, so they never pay for one). Internal
               re-matches (cache_unfinished_req) pass req=None and stay
               exact-only.
            3. provider.match_on_prefix_miss(...) -> FuzzyMatchResult | None
            4. Validate: donor_last_node_id resolvable in _node_registry
               (stale donors from resets/evictions are dropped);
               len(kv_cache_indices) == cached_token_count.
            5. ALLOC (owner: recipient request): if the donor span is not
               position-aligned, pre-allocate cached_token_count slots ->
               req.fuzzy_realized_locs. Allocation failure = clean
               exact-only fallback; no request state was mutated.
            6. LOCK (owner: FuzzyRadixCache): inc_lock_ref(donor node) ->
               req.fuzzy_donor_node. Pinned donors cannot be LRU-evicted
               while any recipient is in flight.
            7. Return MatchResult with device_indices = exact ++ donor
               indices, fuzzy_matched_len, and cache_protected_len =
               exact+fuzzy (the freeing floor for cache_*_req).
  └─ req.cache_fuzzy_matched_len = fuzzy_matched_len

ForwardBatch.init_new                              (model_executor/forward_batch_info.py)
  └─ fuzzy_reqs = [reqs with cache_fuzzy_matched_len > 0]   (extend only)

ModelRunner._forward_raw                           (model_executor/model_runner.py)
  └─ FuzzyKVRealizer.realize(fuzzy_reqs)           (fuzzy_match/realizer.py)
       Runs on the forward stream after the decode-CUDA-graph early
       return and before the extend dispatch — never inside graph
       capture/replay (same placement as the deferred mamba COW hook).
       ROPE (owner: FuzzyKVRealizer): V copied; K = apply_rope(new_pos,
       reverse_rope(donor_pos, K_donor)). Layers flagged by
       layer_recompute_mask are zeroed instead (drop, not recompute).
       req_to_token[fuzzy span] repointed to the realized slots.
       Per-request state cleared in a finally block so chunked-prefill
       re-entry, decode, and retract-resume never re-trigger.

decode rounds                                       (unchanged)

FuzzyRadixCache.cache_finished_req                 (fuzzy_radix_cache.py)
  1. FREE (owner: recipient): reclaim req.fuzzy_realized_locs if the
     forward pass never consumed them (e.g. aborted request).
  2. RadixCache.cache_finished_req (super):
       insert(): realized slots are adopted by the recipient's tree path.
       _on_finished_insert hook (between insert and duplicate-freeing,
       while the request's slots are still live):
         provider.cache_on_request_finished(...)   -> register as donor
         provider.on_donor_inserted(node.id)       -> donor addressable by
                                                      TreeNode id
       free duplicates above req.cache_protected_len; dec_lock_ref(last_node).
  3. UNLOCK: dec_lock_ref(req.fuzzy_donor_node).
```

`_node_registry` (TreeNode.id -> node) is maintained by `FuzzyRadixCache`
overrides of `_insert_helper` / `_split_node` / `_delete_leaf` / `reset`,
so a donor reference is always either resolvable-and-pinnable or
detectably stale — never dangling.

## Configuration

| Flag | Default | Why it exists |
|---|---|---|
| `--radix-cache-backend fuzzy_match` | off | The enable switch; registers nothing and costs nothing when unset. |
| `--fuzzy-match-provider` | `SemanticEmbedding` | Provider selection; the interface admits out-of-tree providers. |
| `--fuzzy-semantic-threshold` | `0.60` | Precision knob: cosine floor for accepting a donor. |
| `--fuzzy-min-reuse-ratio` | `0.50` | Hit gate: donors covering less of the prompt are rejected. |
| `--fuzzy-min-match-length` | `16` | Skips fuzzy lookup behind weak partial exact anchors. |
| `--fuzzy-model-arch` | none | Provider preset (RoPE layout / alignment) for the served model. |

Provider-internal tuning (donor store size, chunking, embedding model,
and the minimum-suffix lookup gate `fuzzy_min_suffix_tokens=256` that
bounds no-hit overhead) lives in `FuzzyMatchConfig` defaults, not CLI
flags.

## Scope and guarantees

- Exact prefix matching is untouched and always wins; fuzzy runs only on
  the missed suffix. The default backend path is byte-identical when the
  backend is not selected (two seams: one `MatchResult` field, one no-op
  hook in `cache_finished_req`).
- Reuse changes model outputs by construction: donor K/V attended to the
  donor's context. The provider's quality gates plus the per-layer
  zero-out mask bound the drift; accuracy methodology and results are in
  the PR description.
- Not yet supported: MLA-style KV pools, EAGLE speculative decoding,
  multi-region (`|exact|miss|fuzzy|miss|...`) reuse, hierarchical (host)
  cache interaction. Each is rejected explicitly rather than silently.
