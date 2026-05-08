# Speculative Decoding — Naming Conventions

Apply this rule when adding, renaming, or reviewing identifiers in speculative decoding code (anything under `python/sglang/srt/speculative/`, related attention backends, scheduler accumulators, IPC fields, observability metrics, or CLI flags).

## Rule 1 — Verb form, drop `-ed`

Use the verb form `accept` everywhere. Don't use the past-participle form `accepted`.

| Don't | Do |
|---|---|
| `num_accepted_tokens` | `num_accept_tokens` |
| `accepted_indices` | `accept_indices` |
| `accepted_token_ids` | `accept_tokens` (also see Rule 3) |

## Rule 2 — The extra/bonus token is `bonus_token`

The "+1" token that the target model always emits in addition to verifying drafts is the **bonus token**. Always call it `bonus_token`.

| Don't | Do |
|---|---|
| `verified_id` | `bonus_token` |
| `output_id` (when referring to the bonus) | `bonus_token` |

`req.output_ids` (the full output history of a request) is unrelated and stays as is.

## Rule 3 — `accept` includes bonus; `correct` excludes bonus

The semantic distinction lives in the **verb**, not the noun. Don't enumerate noun pairs.

| Verb | Meaning |
|---|---|
| **`accept_*`** | Includes the bonus token |
| **`correct_*`** | Drafts only, no bonus |

Pair with whatever noun fits the data (`tokens`, `drafts`, `indices`, …). No required pairing, but **preferred default nouns**: `accept_tokens` and `correct_drafts` — `correct` semantically describes drafts (what got verified), `accept` describes the resulting token sequence (incl. bonus).

| Form | Meaning |
|---|---|
| `accept_tokens` / `accept_indices` | Include bonus |
| `correct_drafts` | Drafts only, no bonus |
| `num_accept_tokens` | Count incl. bonus |
| `num_correct_drafts` | Count excl. bonus |

### Exception: `accept_rate` / `accept_length` follow paper convention

These two metric names are entrenched in the spec-decoding literature and in external-facing fields (`meta_info`, Prometheus). Their semantics are paper-defined, not Rule-3-defined:

| Name | Paper term | Bonus? | Definition |
|---|---|---|---|
| `accept_rate` | $\alpha$ (Leviathan 2023) | **No** | per-draft-token acceptance probability = `correct_drafts / proposed_drafts` |
| `accept_length` | $\tau$ (EAGLE) | **Yes** | avg tokens per verify step = `completion_tokens / verify_ct` |

Internal counters still follow Rule 3 strict semantics: `num_correct_drafts` (no bonus), `num_accept_tokens` (with bonus).

## Rule 4 — `num_` for counts; `_ct` for counters; `_rate` for rates; no prefix for IDs

Each form has its own marker. **Never mix** (no `num_X_ct`, no `num_accept_rate`).

| Form | Pattern | Meaning | Examples |
|---|---|---|---|
| **Count** | `num_X` | Snapshot quantity at one point in time (often a tensor or scalar) | `num_accept_tokens`, `num_correct_drafts`, `num_proposed_drafts` |
| **Counter** | `X_ct` | Monotonically incrementing accumulator over time | `spec_verify_ct`, `forward_ct` |
| **Rate / ratio** | `X_rate` | Fractional value in `[0, 1]` | `accept_rate` |
| **Tokens / content array** | no prefix | The actual token data, not a count | `accept_tokens`, `correct_drafts`, `bonus_token` |

## Rule 5 — Avoid `length` / `lens` internally

Internal code uses `num_X` for counts (Rule 4), not `length` / `lens`.

The only allowed `_length` name is `accept_length` (paper-aligned external metric, see Rule 3 exception). Outside kernel signatures, never write `accept_lens` / `accepted_length` / `commit_lens`.

### Exception: Triton kernel signatures

Triton kernel parameters (and their inner scalar loads) may use `_lens` / `_len` to align with the broader PyTorch ecosystem (`seq_lens`, `cu_seqlens_q`). Example: `accept_lens` (kernel param, `[bs]`) and `accept_len = tl.load(accept_lens + pid)` (scalar inside kernel) are OK. The wrapping Python code that calls the kernel still uses `num_accept_tokens`.

## Rule 6 — Drop redundant `_token_id` / `_token_ids` suffix in spec scope

`_id` / `_ids` and `_token` / `_tokens` are both fine. But don't combine — `_token_id` / `_token_ids` is redundant **inside spec decoding**, because spec code only ever deals with vocab integers.

The semantic differs by scope:

| Scope | Example | What `_token_id` means |
|---|---|---|
| **Framework / multimodal / tokenizer** | `image_token_id`, `pad_token_id`, `eos_token_id`, `mask_token_id`, `bos_token_id` | A specific named/role token's vocab ID. The prefix names the role; `_token_id` says it's the integer ID for that role. Both halves carry information. |
| **Spec decoding** | `accepted_token_ids`, `curr_token_id`, `out_token_ids` | Redundant. Spec only deals with vocab integers; `_id` adds nothing beyond `_token`. |

### Renames

| Don't | Do |
|---|---|
| `accepted_token_ids` | `accept_tokens` (Rule 1 + 3) |
| `curr_token_id` | `current_token` |
| `out_token_ids` | `out_tokens` |
| `_resolve_spec_overlap_token_ids` | `_resolve_spec_overlap_tokens` |

## Rule 7 — Singular vs plural

Plural for any non-scalar tensor (`[bs]`-shaped, flat, or multi-dim); singular only for scalars (kernel `tl.load` results, single-int locals). Applies to all spec-decoding tensors (tokens, indices, etc.). The fixed name `bonus_token` (Rule 2) is the one singular exception even though its shape is `[bs]`.

```python
accept_tokens: torch.Tensor     # [total_accepted] flat - plural
accept_indices: torch.Tensor    # [bs, num_draft_tokens] - plural
draft_tokens: torch.Tensor      # [bs * num_draft_tokens] flat - plural
accept_token = tl.load(...)     # int32 scalar in a kernel iteration - singular
```

## Out of scope (these names stay as is)

These rules apply to **spec-decoding-specific** identifiers. Pre-existing or framework-level names are kept.

- **PyTorch / ecosystem**: `seq_lens`, `extend_seq_lens`, `cu_seqlens_q`
- **Framework / multimodal vocab**: `image_token_id`, `pad_token_id`, `eos_token_id`, `mask_token_id`, `hot_token_id`, `bos_token_id`, `topk_id`
- **Request-level state**: `req.input_ids`, `req.output_ids`, `req.origin_input_ids`, `next_token_ids` (`model_runner.sample` output)
- **Frozen C++ kwargs**: `accept_token_num` (sgl-kernel)
- **Non-token IDs**: `req_id`, `gpu_id`, `layer_id`, `program_id`
