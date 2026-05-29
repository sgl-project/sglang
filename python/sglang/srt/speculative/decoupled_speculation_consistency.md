# Decoupled Speculation Prefix Consistency

Decoupled speculation lets the drafter run ahead of the verifier. For this to
produce useful speedup, the drafter must generate draft tokens from a prefix that
matches the verifier's committed prefix.

If the drafter runs from a wrong prefix, its draft tokens are much less likely to
be accepted. Even if the verifier rejects them correctly, the system has already
spent draft compute, transport bandwidth, and scheduler capacity on tokens that
cannot accelerate decoding.

> The consistency protocol is designed around one goal: keep the drafter's
> committed prefix aligned with the verifier before accepting future draft tail
> tokens.

## Core Invariant

On the drafter side, each decoupled draft request tracks
`verifier_committed_prefix_len`. This is the number of output tokens that are
known to be aligned with the verifier-committed output prefix.

```text
drafter.output_ids[:verifier_committed_prefix_len]
==
verifier.output_ids[:verifier_committed_prefix_len]
```

The drafter may have more speculative tokens after this prefix, but those tokens
are not committed. Future verifier commits decide whether they can become part
of the aligned prefix.

On the verifier side, `DraftTailBuffer.committed_len` tracks the same concept:
how many verifier-prefix output tokens have been confirmed by the drafter. The
verifier may also hold `pending_expected_tokens` for verifier-committed tokens
that the drafter has not confirmed yet.

## Protocol Messages

The protocol messages are defined in
`python/sglang/srt/speculative/decoupled_spec_io.py`.

| Message | Direction | Role |
| --- | --- | --- |
| `DraftSync` | Verifier -> drafter | Opens a drafter-side request from a verifier-owned prompt and committed output prefix. |
| `DraftTailStreamOutput` | Drafter -> verifier | Streams one drafter token with `base_committed_len`, `new_token_pos`, and `new_token_id`. |
| `VerifyCommit` | Verifier -> drafter | Publishes a non-empty contiguous verifier-committed output segment. |
| `DraftClose` | Verifier -> drafter | Releases drafter-side state when the request finishes or aborts. |

### DraftSync

`DraftSync` establishes the first aligned prefix for a drafter request. It
carries `prompt_token_ids` and `committed_output_ids`, both owned by the
verifier.

### DraftTailStreamOutput

`DraftTailStreamOutput` carries one generated drafter token:

| Field | Meaning |
| --- | --- |
| `base_committed_len` | Verifier-committed prefix length used as the base when the drafter emitted this token. |
| `new_token_pos` | 0-based output-token position of `new_token_id`. |
| `new_token_id` | Token id emitted by the drafter. |

The verifier uses these fields to decide whether the token is contiguous, stale
relative to a known mismatch, or eligible to confirm a pending verifier token.

### VerifyCommit

`VerifyCommit` publishes this verifier-committed segment:

```text
verifier.output_ids[
    pre_verify_committed_len:
    pre_verify_committed_len + len(committed_token_ids)
]
```

The drafter applies `VerifyCommit` messages in prefix order. Matching tokens
advance `verifier_committed_prefix_len`. A mismatch means the drafter's local
speculative suffix was generated from a different prefix, so the drafter must
truncate or reprefill from the mismatch position before continuing.

## Core Data Structures

| Structure | Owner | Purpose |
| --- | --- | --- |
| `DraftTailBuffer` | Verifier entry rank | Turns asynchronous drafter stream outputs into a verifier-consumable draft tail while preserving prefix consistency. |
| `VerifierCommitSegment` | Drafter-side control inbox | Coalesces contiguous verifier commits for one drafter request. |
| `DraftControlInbox` | Drafter-side token-sync thread and scheduler | Temporarily stores control messages, then exposes ready controls at scheduler-safe points. |

## DraftTailBuffer

`DraftTailBuffer` is the verifier-side state machine for one synced request.

It tracks:

| Field | Meaning |
| --- | --- |
| `committed_len` | Prefix length already confirmed between verifier and drafter on the verifier side. |
| `tail_tokens` | Contiguous drafter tokens received after `committed_len`. |
| `pending_expected_tokens` | Verifier-committed tokens that had no comparable draft-tail token yet. |
| `can_accept_prefix_len` | Stale-base boundary used to reject stream outputs generated before a known mismatch. |

> `can_accept_prefix_len` is a stale-base boundary, not a pending-token cursor.
> It should only advance after verifier and drafter tokens are compared at the
> same `token_pos` and found to mismatch.

### Applying VerifyCommit

When a `VerifyCommit` reaches `DraftTailBuffer`, the verifier first checks the
commit prefix:

```text
pre_verify_committed_len == committed_len + len(pending_expected_tokens)
```

Then it compares `committed_token_ids` with the buffered draft tail.

#### Full match

The committed segment fully matches the buffered tail prefix. `DraftTailBuffer`
removes the matched tail tokens and advances `committed_len`. No stale-base
boundary changes, because no divergence was observed.

#### Tail too short

The verifier committed more tokens than the drafter has yielded. The missing
committed tokens are appended to `pending_expected_tokens`, and the verifier
waits for later drafter-side confirmation.

This case must not advance `can_accept_prefix_len`: a short tail means there was
no comparable drafter token yet, not that the drafter was using a wrong prefix.

#### Actual mismatch

Both verifier and drafter have a token at the same `token_pos`, but the token ids
differ. The prefix has diverged at that position.

In this case, `DraftTailBuffer`:

1. Drops the unmatched draft tail.
2. Appends the remaining verifier-committed tokens to `pending_expected_tokens`.
3. Advances `can_accept_prefix_len` to the new `committed_len`.

After that, any stream output with
`base_committed_len < can_accept_prefix_len` is rejected as `stale_base`.

### Applying DraftTailStreamOutput

When a drafter stream output arrives, `DraftTailBuffer` validates both its base
prefix and its token position.

#### Pending expected prefix

If `pending_expected_tokens` is non-empty, `tail_tokens` must be empty and the
next acceptable output must satisfy:

```text
base_committed_len == committed_len
new_token_pos == committed_len
new_token_id == pending_expected_tokens[0]
```

If `new_token_pos > committed_len`, the output is treated as
`pending_expected_gap`. A future-position token cannot consume the current
pending expected token.

If `new_token_pos == committed_len` but `new_token_id` differs, this is an actual
verifier/drafter mismatch and `can_accept_prefix_len` advances.

#### Normal tail append

If there is no pending expected prefix, the stream output can append to
`tail_tokens` only when it is contiguous with the current buffer:

```text
new_token_pos == committed_len + len(tail_tokens)
```

If `new_token_pos` points inside the existing tail, the token is a duplicate and
must match the already buffered token. If it points beyond the buffer end while
using the current base, a token was skipped, so `DraftTailBuffer` raises instead
of silently accepting a gap.

#### Stale base

Before accepting normal tail data or pending-prefix confirmation, the verifier
checks:

```text
base_committed_len >= can_accept_prefix_len
```

Outputs below that boundary are rejected as `stale_base`.

## VerifierCommitSegment

`VerifierCommitSegment` protects the drafter from applying verifier commits out
of order. When receiving contiguous `VerifyCommit` messages for the same drafter
request, the token-sync thread coalesces them into one segment.

Each appended message must satisfy:

```text
message.pre_verify_committed_len == segment.end_committed_len
```

When the drafter scheduler is ready to consume work, `DraftControlInbox` extracts
a prefix of each segment that is safe to apply. The scheduler then aligns the
drafter request to that verifier-committed segment before emitting more draft
tail tokens.

## Summary

The consistency protocol keeps this invariant true:

```text
drafter.output_ids[:verifier_committed_prefix_len]
==
verifier.output_ids[:verifier_committed_prefix_len]
```

`DraftSync` establishes the invariant, `VerifyCommit` advances it,
`DraftTailStreamOutput` supplies speculative tokens, and `DraftTailBuffer`
rejects stale or position-inconsistent tokens. This lets the drafter run ahead
while still drafting from a prefix that is known to match the verifier.
