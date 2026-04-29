# RelayKV SGLang MVP-0 Completion Checklist

## Date

2026-04-29

## Repository

- repo: `~/work/sglang-relaykv`
- branch: `relaykv-memory-mvp0`
- remote: `mine/relaykv-memory-mvp0`

## MVP-0 Goal

RelayKV MVP-0 is a shadow-only integration for SGLang.

It adds server arguments, a RelayKV skeleton module, resident/cold planning, logging, no-op behavior, and shallow model/attention profile metadata.

MVP-0 explicitly does not move KV tensors or alter attention behavior.

## Confirmed Behavior

### 1. Server args

Confirmed via `launch_server --help`.

Expected args:

```text
--enable-relaykv
--relaykv-mode {off,shadow}
--relaykv-resident-budget-tokens
--relaykv-recent-window
--relaykv-anchor-pages
--relaykv-log-interval
```

### 2. RelayKV skeleton imports

Confirmed import and compile.

Files:

```text
python/sglang/srt/relaykv/__init__.py
python/sglang/srt/relaykv/config.py
python/sglang/srt/relaykv/planner.py
python/sglang/srt/relaykv/metrics.py
python/sglang/srt/relaykv/profile.py
```

### 3. Short input shadow planning

Short input produced a shadow plan with all tokens resident.

Observed:

```text
seq_len: 36
resident_budget_tokens: 1024
planned_resident_tokens: 36
planned_cold_tokens: 0
estimated_resident_ratio: 1.0
```

Interpretation:

```text
seq_len <= resident_budget_tokens
all tokens are planned resident
no cold tokens
```

### 4. Long input shadow planning

Long input produced a shadow plan with cold tokens.

Observed:

```text
seq_len: 2535
resident_budget_tokens: 1024
planned_resident_tokens: 1024
planned_cold_tokens: 1511
estimated_resident_ratio: 0.4039447731755424
recent_page_range: [1767, 2535]
anchor_pages: [0, 1, 2, 3]
```

Interpretation:

```text
logical context: 2535 tokens
resident budget: 1024 tokens
planned cold tokens: 1511
resident ratio: about 40.4%
```

### 5. No-op when RelayKV is disabled

Confirmed:

```text
--enable-relaykv not provided
/v1/chat/completions returns 200 OK
relaykv_shadow_plan_prefill is not emitted
```

### 6. No-op when RelayKV mode is off

Confirmed:

```text
--enable-relaykv --relaykv-mode off
/v1/chat/completions returns 200 OK
relaykv_shadow_plan_prefill is not emitted
```

### 7. Shadow log with model profile metadata

Confirmed Qwen2.5-style GQA profile metadata in shadow log.

Observed:

```json
{
  "attention_type": "gqa",
  "model_arch": "Qwen2ForCausalLM",
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "reason": "Qwen2.5-style standard full attention is supported for shadow planning",
  "relaykv_profile_supported": true
}
```

Full observed log excerpt:

```text
relaykv_shadow_plan_prefill={
  "anchor_pages": [0, 1, 2, 3],
  "attention_type": "gqa",
  "estimated_resident_ratio": 0.4039447731755424,
  "mode": "shadow",
  "model_arch": "Qwen2ForCausalLM",
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "page_size": 1,
  "planned_cold_tokens": 1511,
  "planned_resident_tokens": 1024,
  "reason": "Qwen2.5-style standard full attention is supported for shadow planning",
  "recent_page_range": [1767, 2535],
  "relaykv_enabled": true,
  "relaykv_profile_supported": true,
  "request_id": "bd967363a8ba45739328d0e15742fd3e",
  "resident_budget_tokens": 1024,
  "seq_len": 2535
}
```

### 8. OpenAI-compatible API still works

Confirmed:

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
```

## Safety Boundaries Still Preserved

The current MVP-0 integration should continue to preserve:

```text
No KV tensor movement
No KV cache eviction
No host/device KV swap
No attention kernel changes
No scheduler behavior changes except logging
No generation output changes expected
No .github/workflows changes
```

## Current MVP-0 Status

MVP-0 can be considered complete if the latest profile metadata commit is pushed and working tree is clean.

Completed:

```text
[x] Design note added
[x] Devlogs added
[x] RelayKV server args added
[x] RelayKV skeleton module added
[x] Shadow planner added
[x] Shadow plan log hook added
[x] Duplicate log suppression added
[x] Disabled no-op confirmed
[x] Mode off no-op confirmed
[x] Long input cold-token planning confirmed
[x] Qwen2.5 GQA profile metadata confirmed
```

## Recommended Commit / Push Check

```bash
cd ~/work/sglang-relaykv

git status --short
git log --oneline --decorate --max-count=10
git push
```

## Next Phase Recommendation

Next phase should not yet be full apply-budget.

Recommended next step is MVP-1 planning:

```text
Host backup shadow
```

MVP-1 should still avoid changing generation behavior.

MVP-1 scope:

```text
- observe which KV blocks would be resident/cold
- estimate host backup memory
- optionally copy metadata only first
- if tensor copy is introduced, keep GPU tensors untouched
- do not free GPU KV
- do not alter attention
- do not swap-in/swap-out during generation
```

## Suggested MVP-1 Phases

### MVP-1a: Metadata-only host backup plan

```text
- no tensor copy
- compute cold ranges
- estimate bytes for K/V by layer/head/dtype if accessible
- log per-request resident/cold bytes estimate
```

### MVP-1b: Host backup dry-copy behind flag

```text
- copy cold KV to CPU only when explicit flag is enabled
- do not free GPU KV
- do not use copied KV for attention
- log copy time and host memory
```

### MVP-1c: Resident mapping design

```text
- design logical token index -> resident index map
- no kernel change yet
- validate mapping with unit tests or offline planning
```

## Next Codex Task Candidate

```text
Add RelayKV MVP-1a metadata-only memory estimate logging.

Do not move KV tensors.
Do not copy KV tensors.
Do not change attention behavior.
Estimate resident/cold KV bytes from model config and shadow plan.
Log bytes only in shadow mode.
```
