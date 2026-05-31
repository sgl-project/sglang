# AC-8 — 64K servability at the lifted DS operating point (PASS)

A ~70K-token raw `/generate` is now **ADMITTED (HTTP 200)** at the lifted DS int8 / `mem_fraction_static=0.7`
/ radix-on operating point — no longer the Loop-5 mem-0.6 `HTTP 400 "Input length (69970) exceeds the
maximum allowed (53050)"`. This is the footprint→admission spine's servability payoff for long context.
**This is a servability/admission test, NOT a recall test** (DS long-context needle recall is bounded by
the kernel-locked `top_k=2048` and is Tier-2/AC-10; see `ac12_topk_sweep/` and the Loop-5 NIAH 75/5/0).

## Operating point (proven from `get_server_info_{before,after}.json` + boot log)
- `enable_double_sparsity=True`, `double_sparsity_config.signature_dtype=int8`, int8 `token_label_table`
  **6.48 GB/rank on all 8 ranks** (`dtype=torch.int8 scales=float16`, L=61 T=396160 H=16 D=16 page=64).
- `mem_fraction_static=0.7`, `disable_radix_cache=False`, radix authorized by the config-bound fixture
  artifact `ds_radix_fixture_state_int8.json` (sha256 `f3b67943…`, both M3-B fixtures PASSED).
- KV cache allocated: `#tokens = max_total_num_tokens = 396096` (fp8_e4m3, 17.73 GB), `context_len=163840`,
  `chunked_prefill_size=8192`. The identical lifted point verified in AC-4/AC-5/AC-7.

## The probe (`development/loop6/probe_64k.json` — the named AC-8 deliverable)
- Deterministic varied prose (seed 20260531) + a final one-line question, raw `/generate`,
  `max_new_tokens=16`, `temperature=0`. `text_sha256=652e4f51c00dd77e…`.
- Local tokenizer estimate **70759 tokens**; the server reported the **same** `prompt_tokens=70759`
  (exact provenance match). `70759 > 53056` (the Loop-5 mem-0.6 pool that rejected 69970 with HTTP 400)
  and `>= 69970` (the Loop-5 64K reference) and `< 396096` (the lifted pool).

## Result (`ac8_probe_response.json`)
| Field | Value |
|---|---|
| HTTP status | **200** (admitted) |
| `prompt_tokens` (server) | **70759** (== local estimate) |
| served `max_total_num_tokens` | **396096** |
| `completion_tokens` / `finish_reason` | 16 / `length` (generated to the 16-token cap) |
| latency | 11.95 s (chunked prefill + 16 decode) |
| server alive before / after | **yes / yes** (`/get_server_info` → 200 both) |
| OOM / CUDA-error lines in the whole boot+serve log | **0** |

The output text is a degenerate continuation (`" The passage is a list of thethe passage…"`) — expected
for **raw** `/generate` on an instruction-tuned model with no chat template (`BL-20260529-dsv32-quality-smoke-needs-chat-template`)
and irrelevant to servability: AC-8 measures **admission**, not answer quality.

## Server-log evidence (`server_log_excerpt.txt`)
The 70759-token prompt chunk-prefilled in **8×8192 + 5248 = 70759** tokens (matching `prompt_tokens`
exactly), `#queue-req: 0` (admitted immediately, no admission queue), `#running-req: 0` (single request),
token usage rose only **0.02 → 0.18** of the 396096 pool, `cuda graph: False` (eager prefill). **0 OOM lines**
anywhere in the boot+serve log; the server answered `/get_server_info` 200 after the probe (alive, stable).

## Loop-5 → lifted contrast (the AC-8 negative test: this is a lifted-mem RETRY, not a silent re-record)
| | Loop-5 DS mem-0.6 | This round — lifted DS int8 mem-0.7 |
|---|---|---|
| `max_total_num_tokens` (KV pool) | 53056 | **396096** |
| ~70K `/generate` | **HTTP 400** "Input length (69970) exceeds the maximum allowed (53050)" | **HTTP 200**, `prompt_tokens=70759` served |
| OOM during serve | n/a (rejected at admission) | **none** |

The 70759-token prompt is larger than the Loop-5 prompt that 400'd at mem-0.6, so the same admission
length-check that failed at mem-0.6 is exercised here and now passes — a genuine lifted-mem retry, not a
re-record of the old 400.

## Verdict
**AC-8 PASS (servability).** At the lifted DS int8 / mem-0.7 / radix-on point, a ~70K-token `/generate`
is admitted (HTTP 200), served without OOM or instability, with `max_total_num_tokens=396096` recorded —
the Loop-5 64K HTTP-400 admission ceiling is removed. No characterized ceiling is needed (the prompt fits
with large margin: 70759 of a 396096 pool, 18% token usage). Recall accuracy at 64K remains a separate
Tier-2/AC-10 concern, unchanged by this servability result.

## Artifacts
- `development/loop6/probe_64k.json` — the named ~70K-token probe payload (text + sha256 + token estimate).
- `ac8_probe.py` — reproducible probe driver (reads the payload, asserts its sha, captures before/after
  `/get_server_info`, sends raw `/generate` catching rejections as recordable results, writes the response).
- `ac8_probe_response.json` — HTTP status, prompt_tokens, served max_total, finish_reason, latency, snippet.
- `get_server_info_before.json` / `get_server_info_after.json` — operating point + server-alive proof.
- `server_log_excerpt.txt` — chunked-prefill window (admit, 0 queue, token usage) + the 0-OOM scan.
