---
name: compute-mamba-ratio
description: Compute the optimal --mamba-full-memory-ratio (or --max-mamba-cache-size pin) for a hybrid attention + linear-attention (Mamba / GDN / KDA) model's two serving memory pools, from the workload and serving config. Use when a user asks what ratio to set, why concurrency is clamped, or how to size the state vs KV pools for a hybrid model.
---

# Optimal hybrid dual-pool ratio (`--mamba-full-memory-ratio`)

A hybrid model (attention layers + linear-attention layers — the recurrent-state family: Mamba/SSM, GDN, KDA, etc.) splits serving memory into two independently-budgeted pools, fixed once at startup:

- **state pool** (the linear-attention recurrent state) → caps **concurrency** (hard: whole slots, worst-case reserved, fail-loud)
- **full-KV pool** (attention KV) → caps **context × concurrency** (soft: paged, over-committable via retraction)

`--mamba-full-memory-ratio r` splits the post-weight budget: `mamba_budget = rest · r/(1+r)`, i.e. `mamba_budget : kv_budget = r`. This skill picks the `r` (or the pin-`--max-mamba-cache-size` alternative) at which **neither pool bottlenecks first** for the user's workload.

## The formula

```
r*  =  (S + D) · token_equiv · dcp_size / L
token_equiv  =  state_bytes_per_slot / kv_bytes_per_token
```

- `L` = average context length per request (input + output tokens)
- `token_equiv` = full-KV token-equivalent of one state slot
- `S` = state slots per running request (cache-strategy dependent, table below)
- `D` = `--speculative-num-draft-tokens` (0 if NOSPEC); each running req carries `D` extra intermediate states
- `dcp_size` = `--dcp-size` (1 without DCP). DCP shards the per-rank KV by `dcp_size`, so KV gets ~`dcp_size×` cheaper per request → the balance shifts that much toward the state pool.

`r` is dimensionless (just the split). To also predict the actual concurrency you need `rest` (below).

## Inputs to collect from the user

1. **`L`** — average context (input + output) in tokens.
2. **The two per-GPU byte constants** — one of:
   - **(a) measured (preferred, exact)** — from one boot log at *any* ratio:
     - `Mamba Cache is allocated. ... ssm_state size X GB` with `max_mamba_cache_size: N` → `state_bytes_per_slot = X / N`
     - `KV Cache is allocated. #tokens: M, KV size: Y GB` → `kv_bytes_per_token = Y / M`
   - **(b) derived** — model arch (linear-layer count + state dims `d_state/d_conv/heads/head_dim`; attention type + KV dims: MLA latent dim, or GQA `kv_heads·head_dim·layers`) **× the dtypes** below.
3. **`S`** — from `--mamba-radix-cache-strategy`, the overlap scheduler, and `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` (table below).
4. **`D`** — `--speculative-num-draft-tokens` (0 if NOSPEC; **also 0 when ReplaySSM spec-verify is enabled** — see caveats).
5. **`dcp_size`** — `--dcp-size` (1 if no DCP). If DCP **and** spec with a replicated draft KV, apply the draft caveat (below).
6. **KV dtype** (bf16 / fp8) and **ssm dtype** (fp32 / bf16) — they set the two byte constants (see the `token_equiv` 2×2).
7. *(only to also predict the clamp, not just `r`)* **`rest`** = per-GPU memory − weights at the chosen `--mem-fraction-static`. Read `avail mem` after `Load weight end`, or `Memory pool end. avail mem` + pool sizes, from the boot log.

### `S` (state slots per running request), set by `--mamba-radix-cache-strategy`

All strategies keep prefix caching on; they differ in the track buffer that snapshots chunk-boundary state under the overlap scheduler. `S = base + ping-pong`, where base is 3 (live state + radix retention/COW headroom), and ping-pong = 2 (overlap on, non-lazy) / 1 (lazy, **or** overlap off) / 0 (no track buffer at all). This mirrors `kv_cache_configurator._calculate_mamba_ratio`; read it there if a release moves the constants.

`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` takes the base to 2, since the decode-time skip frees one resident slot per running request. `no_buffer` is the exception and stays at an effective 3: it adds that slot straight back, because its binding limit is the prefill-to-decode peak, which a decode-time change does not shrink.

| strategy | S | S with decode-lock skip | notes |
|---|---|---|---|
| `--disable-radix-cache` | 1 | 1 | prefix cache off entirely; most concurrency, no reuse |
| `no_buffer` | 3 | 3 | no track buffer; requires overlap **off** (+ `page_size 1`) → lower decode throughput |
| `extra_buffer` (default), overlap on | 5 | 4 | track buffer reserved per running request |
| `extra_buffer`, overlap off | 4 | 3 | same buffer, but ping-pong costs 1 slot instead of 2 |
| `extra_buffer_lazy` (recommended) | 4 | 3 | track buffer allocated lazily at the boundary → 1 fewer slot/req; requires overlap **on** |

The skip is a real concurrency lever, not a rounding detail: at the default strategy it takes `S` from 5 to 4, so the same state budget admits 25% more requests and `r*` drops by the same factor.

Overlap counts as off when `--disable-overlap-schedule` is passed **or** `--pp-size > 1` (pipeline parallelism turns it off for you). So a PP deployment left on the default strategy sits on the `S = 4` row, not `S = 5` — using 5 there overstates the state cost and starves KV.

### `token_equiv` — compute it from the two byte constants

`token_equiv = state_bytes_per_slot / kv_bytes_per_token`. Both bytes are **per-model, per-GPU** — read them from a boot log (or derive from the arch + dtypes) and divide. That's the whole thing; the dtype effects below are just for a quick mental estimate.

The recurrent state has two parts: `state_bytes = SSM_bytes + conv_bytes`. The **SSM** part scales with `--mamba-ssm-dtype`; the **conv** part is always bf16 (`SGLANG_MAMBA_CONV_DTYPE`, fixed). How the dtype knobs move `token_equiv`:

- **fp8 KV → ×2** (universal, exact): fp8 is 1 byte vs bf16's 2, so `kv_bytes_per_token` halves.
- **fp32 ssm vs bf16 ssm → ×≈2**: only the SSM tensor doubles, conv stays fixed, so the exact factor is `(2·SSM + conv)/(SSM + conv)` — just under 2. **conv is usually small relative to SSM, so estimate ≈2** (e.g. one measured model: conv ≈7% of state → factor ≈1.9). Only bother with the exact split if a model's conv is unusually large.
- **fp32 ssm + fp8 KV → ×≈4**.

For an actual number, don't apply factors — read `state_bytes_per_slot` and `kv_bytes_per_token` for **your** model+dtype from the boot log and divide.

The `GB` the boot log prints for both pools is **GiB** (`1024³`). Convert both the same way and `token_equiv` is unaffected either way, since it is a ratio — but get it right before comparing an absolute byte figure against your own log.

Worked example (one measured model, TP8, fp32 ssm + fp8 KV), read off one boot log:

```
Mamba Cache is allocated. max_mamba_cache_size: 257, conv_state size: 0.46GB, ssm_state size: 13.04GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 1167552, KV size: 15.03 GB
```

`state_bytes_per_slot = (0.46 + 13.04) GiB / 257 ≈ 56.4 MB`, `kv_bytes_per_token = 15.03 GiB / 1167552 ≈ 13.8 KB` → `token_equiv ≈ **4080**`.

## Compute

```python
def optimal_ratio(L, state_bytes_per_slot, kv_bytes_per_token, S, D=0, dcp_size=1):
    token_equiv = state_bytes_per_slot / kv_bytes_per_token
    r = (S + D) * token_equiv * dcp_size / L
    return r  # value of --mamba-full-memory-ratio (>1 is legal)

def predict_clamp(rest_bytes, r, state_bytes_per_slot, S, D=0):
    mamba_budget = rest_bytes * r / (1 + r)
    slots = mamba_budget / state_bytes_per_slot
    # spec: each running req reserves (S+D) worth; non-spec just S
    return int(slots // S)  # NOSPEC; with spec the budget joint-solves for (S+D)·per_req per req
```

Then state the result three ways: the **`r` value**, the **predicted clamp** (if `rest` given), and **which pool binds** (`min(mamba_clamp, KV_cap)`, where `KV_cap = kv_tokens · dcp_size / L`).

## Procedure

1. **Free lever first**: if the boot log shows large idle in `Memory pool end. avail mem` (e.g. 30–40 GB at mem-frac 0.85), raise `--mem-fraction-static` (→0.92) before touching the split — it grows `rest` for both pools at no cost. Validate graph-capture headroom once.
2. Collect the inputs. Prefer a real boot log for the two byte constants.
3. Compute `r*`. If `r*` would drive the state pool below one request's worth (`mamba_budget < S · per_req`, happens at very long `L`), **switch to pinning `--max-mamba-cache-size = target_concurrency · S`** and let the rest go to KV — a sub-0.15 `r` is fragile.
4. Report `r`, predicted clamp, binding pool, and any dtype accuracy gate that applies.

## Worked examples (TP8, B300, validated against measured clamps)

Constants (from boot logs): `state_bytes_per_slot ≈ 56.4 MB` (fp32 ssm), `kv_bytes_per_token ≈ 13.8 KB` (fp8) → `token_equiv ≈ 4080`. Cache strategy `extra_buffer_lazy` → `S = 4`. Workload `L = 9216` (8192 in + 1024 out). NOSPEC → `D = 0`.

- **TP (dcp_size=1)**: `r = 4 · 4080 · 1 / 9216 ≈ 1.8`. Measured: clamp 88, KV cap 87 → balanced. ✅
- **DCP8 (dcp_size=8)**: `r = 4 · 4080 · 8 / 9216 ≈ 14`. Measured at r=14: clamp 125, KV cap 129 → balanced. ✅ (At the naive r=1.8 the DCP KV pool is ~8× over-provisioned — cap 683 vs clamp 86 — wasting budget that should go to the state pool.)

`predict_clamp` on the same model at the **default** `extra_buffer`, overlap on and the decode-lock skip off (`S = 5`), three boot logs:

| config | `max_mamba_cache_size` | `mmcs // S` | measured `max_running_requests` |
|---|---|---|---|
| TP8, no DCP | 257 | 51 | 51 |
| DCP8, `r = 5.97` | 451 | 90 | 90 |
| DCP8, `r = 9` | 474 | 94 | 94 |

Reference `r` for **this example model** (`token_equiv ≈ 4080`, i.e. fp32 ssm + fp8 KV; S=5 default; multiply by `(S+D)/S` for spec, by `dcp_size` for DCP) — recompute with your own `token_equiv` for a different model:

| L | 2K | 4K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| r | 10.0 | 5.0 | 2.5 | 0.62 | 0.31 | 0.16 |

## Caveats

- **DCP + spec**: a replicated (non-DCP-sharded) draft KV does **not** shard by `dcp_size`; at long `L` it dominates per-token cost, so the clean `×dcp_size` overstates DCP's advantage — fall back to a per-token direct-solve (KV term `/dcp` + an un-sharded draft-KV term) when spec is on. (NOSPEC → clean `×dcp_size` holds.)
- **Spec + ReplaySSM → use `D=0`, not the draft-token count.** When ReplaySSM spec-verify is enabled, the `D` intermediate SSM states move off the per-request slot budget onto a **fixed ring** (a one-time deduction from `rest`, not a per-req term). So the mamba-slot cost per running request drops back to `S` (clamp = `mmcs / S`, not `/(S+D)`), and the balance ratio returns to the NOSPEC value (`r* ≈ S·token_equiv·dcp/L`). Measured example (TP8, D=8, L≈9K): applying `D=8` computes `r*≈2.6` but the true optimum is `r≈1.0` — `D=8` lands KV-bound at ~45% below the achievable peak concurrency. Plain spec (no replayssm) keeps `D` = the draft-token count.
- **Under DCP the balance point sits beyond any realistic `L`** (the state pool binds first for almost everything), so the practical recommendation is to **pin `--max-mamba-cache-size = target_concurrency · S` directly** rather than dial a large `r`.
- **`--max-mamba-cache-size` overrides `r`**; bytes beyond the `r` budget come out of the KV pool one-for-one.
- **Precision changes are behind accuracy gates**: `--kv-cache-dtype fp8_e4m3` (doubles `token_equiv` → doubles `r`) and `--mamba-ssm-dtype bfloat16` (~halves `token_equiv`; also silently switches the linear-attention decode backend on SM100+ — pin `--linear-attn-decode-backend triton`) shift outputs; validate accuracy for the workload before production.
- **Asymmetry**: the state pool is worst-case-reserved and fail-loud; KV degrades gracefully (retraction). When `L` is spiky/uncertain, bias `r` **up** rather than starve the state pool.
