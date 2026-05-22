#!/usr/bin/env python3
"""Validate remaining performance levers for MLX hybrid decode.

Measures each proposed optimization against Qwen3.5-0.8B on real hardware.
Run from worktree root:
    PYTHONPATH="$PWD/python:$PYTHONPATH" .venv/bin/python bench_next_levers.py

Levers tested:
  L1  mx.compile on _decode_with_hybrid_batching
  L2  Batched pool KV sync (flush_all_decode_kv)
  L3  Deferred auxiliary-state snapshots (every-N vs every-step)
  L4  Layout alignment (transpose elimination)
  L5  mx.clear_cache() interval tuning
  L6  Batched DeltaNet: stack cache states across requests, one call per layer
"""

import argparse
import copy
import os
import time

os.environ.setdefault("SGLANG_USE_MLX", "1")

import mlx.core as mx
import mlx.nn as nn


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _banner(label):
    print(f"\n{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")


# ─────────────────────────────────────────────────────────────
# Setup: load model + build caches
# ─────────────────────────────────────────────────────────────


def _load_runner(model_path):
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    runner = MlxModelRunner(
        model_path=model_path,
        disable_radix_cache=True,
    )
    return runner


def _build_caches(runner, batch_size, prefill_len=128):
    """Prefill B requests and return their caches + token state."""
    prompt = list(range(1, prefill_len + 1))
    req_ids = []
    for i in range(batch_size):
        rid = f"bench-{i}"
        runner.prefill(
            req_id=rid,
            new_token_ids=prompt,
            full_token_ids=prompt,
            prefix_slot_ids=[],
            new_slot_ids=[],
            req_pool_idx=i,
        )
        req_ids.append(rid)
    return req_ids


def _decode_one_step(runner, req_ids):
    """Run one decode step (build graph + eval + finalize)."""
    pending = runner.decode_batch_start(req_ids)
    cache_arrays = runner._cache_state_arrays(pending.caches)
    mx.eval(pending.lazy_tokens, *cache_arrays)
    runner.decode_batch_finalize(pending)


# ─────────────────────────────────────────────────────────────
# L1: mx.compile on hybrid forward
# ─────────────────────────────────────────────────────────────


def bench_l1_compile(runner, batch_size=8, prefill_len=128, warmup=5, trials=30):
    _banner("L1: mx.compile on _decode_with_hybrid_batching")

    if not runner._cache_layout.has_auxiliary_state:
        print("  SKIP: model has no auxiliary state (not hybrid)")
        return

    req_ids = _build_caches(runner, batch_size, prefill_len)

    # Warmup baseline
    for _ in range(warmup):
        _decode_one_step(runner, req_ids)

    # Baseline: uncompiled
    times_baseline = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _decode_one_step(runner, req_ids)
        times_baseline.append((time.perf_counter() - t0) * 1000)

    # Now compile the method
    original_fn = runner._decode_with_hybrid_batching.__func__

    def compiled_hybrid_batching(self, caches, batched_input):
        return mx.compile(lambda c, b: original_fn(self, c, b))(caches, batched_input)

    import types

    runner._decode_with_hybrid_batching = types.MethodType(
        compiled_hybrid_batching, runner
    )

    # Warmup compiled version (compilation happens on first call)
    for _ in range(warmup):
        _decode_one_step(runner, req_ids)

    # Compiled
    times_compiled = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _decode_one_step(runner, req_ids)
        times_compiled.append((time.perf_counter() - t0) * 1000)

    # Restore
    runner._decode_with_hybrid_batching = types.MethodType(original_fn, runner)

    baseline_med = _median(times_baseline)
    compiled_med = _median(times_compiled)
    speedup = baseline_med / compiled_med if compiled_med > 0 else float("inf")
    print(f"  B={batch_size}  prefill={prefill_len}")
    print(f"  baseline:  {baseline_med:7.2f}ms/step")
    print(f"  compiled:  {compiled_med:7.2f}ms/step")
    print(f"  speedup:   {speedup:.2f}x")

    # Cleanup
    for rid in req_ids:
        runner.remove_request(rid)


def bench_l1_compile_manual(runner, batch_size=8, prefill_len=128, warmup=5, trials=30):
    """Test mx.compile by compiling the inner compute (embed→norm→lm_head)."""
    _banner("L1b: mx.compile on inner compute graph")

    if not runner._cache_layout.has_auxiliary_state:
        print("  SKIP: model has no auxiliary state (not hybrid)")
        return

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        AttentionOffsetCache,
        BatchedDecodeContext,
        clear_context,
        set_context,
    )

    req_ids = _build_caches(runner, batch_size, prefill_len)

    # Measure per-component costs
    caches = [runner._req_caches[rid] for rid in req_ids]
    last_tokens = [runner._req_token_ids[rid][-1] for rid in req_ids]
    batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

    # Time embed
    times_embed = []
    for _ in range(trials):
        t0 = time.perf_counter()
        h = runner._model_embed(batched_input)
        mx.eval(h)
        times_embed.append((time.perf_counter() - t0) * 1000)

    # Time attention layers (batched)
    h = runner._model_embed(batched_input)
    mx.eval(h)
    seq_lens = [
        runner._first_attention_cache(caches[i]).offset for i in range(batch_size)
    ]
    attention_layer_caches = runner._cache_layout.attention_layer_caches(caches)
    ctx = BatchedDecodeContext(
        batch_size=batch_size,
        seq_lens=seq_lens,
        attention_layer_caches=attention_layer_caches,
        attention_pool_index_by_layer=runner._cache_layout.attention_pool_index_by_layer,
    )
    max_offset = max(seq_lens)

    times_attn = []
    for _ in range(trials):
        h_test = mx.array(h)
        t0 = time.perf_counter()
        for layer_idx in runner._cache_layout.attention_layer_indices:
            layer = runner._cache_layout.layers[layer_idx]
            set_context(ctx)
            try:
                shim = AttentionOffsetCache(offset=max_offset)
                h_test = layer(h_test, mask=None, cache=shim)
            finally:
                clear_context()
        mx.eval(h_test)
        times_attn.append((time.perf_counter() - t0) * 1000)

    # Time DeltaNet layers (serialized)
    times_delta = []
    for _ in range(trials):
        h_test = mx.array(h)
        t0 = time.perf_counter()
        for layer_idx in runner._cache_layout.auxiliary_layer_indices:
            layer = runner._cache_layout.layers[layer_idx]
            per_req = [h_test[i : i + 1] for i in range(batch_size)]
            results = []
            for i in range(batch_size):
                results.append(layer(per_req[i], mask=None, cache=caches[i][layer_idx]))
            h_test = mx.concatenate(results, axis=0)
        mx.eval(h_test)
        times_delta.append((time.perf_counter() - t0) * 1000)

    # Time norm + lm_head
    times_head = []
    for _ in range(trials):
        h_test = mx.array(h)
        t0 = time.perf_counter()
        h_test = runner._model_norm(h_test)
        logits = runner._extract_logits(runner._model_lm_head(h_test))
        tokens = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tokens)
        times_head.append((time.perf_counter() - t0) * 1000)

    print(f"  B={batch_size}  prefill={prefill_len}")
    print(f"  Component breakdown (median ms):")
    print(f"    embed:       {_median(times_embed):7.3f}ms")
    print(
        f"    attention:   {_median(times_attn):7.3f}ms  "
        f"({len(runner._cache_layout.attention_layer_indices)} layers, batched)"
    )
    print(
        f"    deltanet:    {_median(times_delta):7.3f}ms  "
        f"({len(runner._cache_layout.auxiliary_layer_indices)} layers, "
        f"{len(runner._cache_layout.auxiliary_layer_indices) * batch_size} calls)"
    )
    print(f"    norm+head:   {_median(times_head):7.3f}ms")
    total = (
        _median(times_embed)
        + _median(times_attn)
        + _median(times_delta)
        + _median(times_head)
    )
    print(f"    sum:         {total:7.3f}ms")
    pct_delta = _median(times_delta) / total * 100 if total > 0 else 0
    print(f"    deltanet %%:  {pct_delta:5.1f}%")

    for rid in req_ids:
        runner.remove_request(rid)


# ─────────────────────────────────────────────────────────────
# L2: Batched pool KV sync
# ─────────────────────────────────────────────────────────────


def bench_l2_batched_sync(
    runner, batch_size=8, prefill_len=128, decode_steps=10, trials=10
):
    _banner("L2: Batched pool KV sync (flush_all_decode_kv)")

    if runner.disable_radix_cache:
        print("  Re-initializing with radix cache enabled...")
        runner.disable_radix_cache = False
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        pool_size = runner._pool_size
        runner.init_cache_pools(None)
        runner._attention_kv_pool = None

    import torch

    from sglang.srt.hardware_backend.mlx.kv_cache import MlxAttentionKVPool

    n_kv_heads, head_dim, dtype = runner._get_attn_config()
    num_attn_layers = runner._cache_layout.num_attention_layers
    pool_size = 8192

    pool = MlxAttentionKVPool(
        pool_size=pool_size + 1,
        num_layers=num_attn_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    mx.eval(*pool.all_buffers())

    unflushed_tokens = 5

    # Build fake caches representing unflushed decode KV
    from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache

    all_caches = {}
    for i in range(batch_size):
        cache = []
        for _ in range(runner._cache_layout.num_layers):
            c = ContiguousAttentionKVCache(
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                max_seq_len=1024,
                dtype=dtype,
            )
            c.keys = mx.random.normal(c.keys.shape).astype(dtype)
            c.values = mx.random.normal(c.values.shape).astype(dtype)
            c.offset = prefill_len + unflushed_tokens
            mx.eval(c.keys, c.values)
            cache.append(c)
        all_caches[f"r{i}"] = cache

    # Approach A: per-request sync (current)
    times_serial = []
    for _ in range(trials):
        for i in range(batch_size):
            rid = f"r{i}"
            cache = all_caches[rid]
            for c in cache:
                c.offset = prefill_len + unflushed_tokens

        t0 = time.perf_counter()
        for i in range(batch_size):
            rid = f"r{i}"
            cache = all_caches[rid]
            for layer_pool_idx, layer_idx in enumerate(
                runner._cache_layout.attention_layer_indices
            ):
                c = cache[layer_idx]
                slot_ids_mx = mx.arange(
                    prefill_len + 1, prefill_len + unflushed_tokens + 1, dtype=mx.int32
                )
                k_sync = c.keys[
                    0, :, prefill_len : prefill_len + unflushed_tokens, :
                ].transpose(1, 0, 2)
                v_sync = c.values[
                    0, :, prefill_len : prefill_len + unflushed_tokens, :
                ].transpose(1, 0, 2)
                pool.k_buffer[layer_pool_idx][slot_ids_mx] = k_sync
                pool.v_buffer[layer_pool_idx][slot_ids_mx] = v_sync
        mx.eval(*pool.all_buffers())
        times_serial.append((time.perf_counter() - t0) * 1000)

    # Approach B: batched sync — stack all requests, one scatter per layer
    times_batched = []
    for _ in range(trials):
        for i in range(batch_size):
            rid = f"r{i}"
            cache = all_caches[rid]
            for c in cache:
                c.offset = prefill_len + unflushed_tokens

        t0 = time.perf_counter()
        for layer_pool_idx, layer_idx in enumerate(
            runner._cache_layout.attention_layer_indices
        ):
            all_k = []
            all_v = []
            all_slots = []
            for i in range(batch_size):
                c = all_caches[f"r{i}"][layer_idx]
                slot_ids_mx = mx.arange(
                    i * 100 + prefill_len + 1,
                    i * 100 + prefill_len + unflushed_tokens + 1,
                    dtype=mx.int32,
                )
                k_sync = c.keys[
                    0, :, prefill_len : prefill_len + unflushed_tokens, :
                ].transpose(1, 0, 2)
                v_sync = c.values[
                    0, :, prefill_len : prefill_len + unflushed_tokens, :
                ].transpose(1, 0, 2)
                all_k.append(k_sync)
                all_v.append(v_sync)
                all_slots.append(slot_ids_mx)
            k_batch = mx.concatenate(all_k, axis=0)
            v_batch = mx.concatenate(all_v, axis=0)
            slots_batch = mx.concatenate(all_slots, axis=0)
            pool.k_buffer[layer_pool_idx][slots_batch] = k_batch
            pool.v_buffer[layer_pool_idx][slots_batch] = v_batch
        mx.eval(*pool.all_buffers())
        times_batched.append((time.perf_counter() - t0) * 1000)

    serial_med = _median(times_serial)
    batched_med = _median(times_batched)
    speedup = serial_med / batched_med if batched_med > 0 else float("inf")
    print(f"  B={batch_size}  unflushed_tokens={unflushed_tokens}")
    print(f"  serial:   {serial_med:7.3f}ms")
    print(f"  batched:  {batched_med:7.3f}ms")
    print(f"  speedup:  {speedup:.2f}x")


# ─────────────────────────────────────────────────────────────
# L3: Deferred auxiliary-state snapshots
# ─────────────────────────────────────────────────────────────


def bench_l3_deferred_snapshots(
    runner, batch_size=8, prefill_len=128, warmup=3, trials=20
):
    _banner("L3: Deferred auxiliary-state snapshots")

    if not runner._cache_layout.has_auxiliary_state:
        print("  SKIP: model has no auxiliary state")
        return

    from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
        _arrays_in_tree,
        _snapshot_cache,
    )

    req_ids = _build_caches(runner, batch_size, prefill_len)
    caches = [runner._req_caches[rid] for rid in req_ids]
    aux_indices = runner._cache_layout.auxiliary_layer_indices

    # Warmup
    for _ in range(warmup):
        _decode_one_step(runner, req_ids)

    # Measure snapshot cost per decode step
    times_snapshot = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for i in range(batch_size):
            for layer_idx in aux_indices:
                _snapshot_cache(caches[i][layer_idx])
        times_snapshot.append((time.perf_counter() - t0) * 1000)

    # Measure full decode step WITH snapshots (baseline)
    times_with = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _decode_one_step(runner, req_ids)
        times_with.append((time.perf_counter() - t0) * 1000)

    # Measure decode step WITHOUT snapshots (monkey-patch store away)
    original_store = runner._store_auxiliary_state
    runner._store_auxiliary_state = lambda *a, **kw: None

    for _ in range(warmup):
        # Still need to do decode steps to advance state
        pending = runner.decode_batch_start(req_ids)
        cache_arrays = runner._cache_state_arrays(pending.caches)
        mx.eval(pending.lazy_tokens, *cache_arrays)
        raw = pending.lazy_tokens.tolist()
        if not isinstance(raw, list):
            raw = [raw]
        next_tokens = [int(t) for t in raw]
        for i, rid in enumerate(pending.req_ids):
            runner._req_token_ids[rid].append(next_tokens[i])
        runner._decode_step_ct += 1

    times_without = []
    for _ in range(trials):
        t0 = time.perf_counter()
        pending = runner.decode_batch_start(req_ids)
        cache_arrays = runner._cache_state_arrays(pending.caches)
        mx.eval(pending.lazy_tokens, *cache_arrays)
        raw = pending.lazy_tokens.tolist()
        if not isinstance(raw, list):
            raw = [raw]
        next_tokens = [int(t) for t in raw]
        for i, rid in enumerate(pending.req_ids):
            runner._req_token_ids[rid].append(next_tokens[i])
        runner._decode_step_ct += 1
        times_without.append((time.perf_counter() - t0) * 1000)

    runner._store_auxiliary_state = original_store

    snapshot_med = _median(times_snapshot)
    with_med = _median(times_with)
    without_med = _median(times_without)
    saved = with_med - without_med
    pct = saved / with_med * 100 if with_med > 0 else 0

    print(f"  B={batch_size}  aux_layers={len(aux_indices)}")
    print(
        f"  snapshot cost (isolated):     {snapshot_med:7.3f}ms "
        f"({batch_size} reqs x {len(aux_indices)} layers)"
    )
    print(f"  decode step WITH snapshots:   {with_med:7.3f}ms")
    print(f"  decode step WITHOUT:          {without_med:7.3f}ms")
    print(f"  savings from deferring:       {saved:7.3f}ms ({pct:.1f}%)")

    for rid in req_ids:
        runner.remove_request(rid)


# ─────────────────────────────────────────────────────────────
# L4: Layout alignment (transpose elimination)
# ─────────────────────────────────────────────────────────────


def bench_l4_transpose(
    n_kv_heads=8, head_dim=128, token_counts=(1, 5, 10, 50), trials=50
):
    _banner("L4: Transpose cost in KV sync path")
    print(f"  n_kv_heads={n_kv_heads}  head_dim={head_dim}")

    for num_tokens in token_counts:
        # (1, heads, tokens, dim) → (tokens, heads, dim)
        cache_shaped = mx.random.normal((1, n_kv_heads, num_tokens, head_dim)).astype(
            mx.float16
        )
        mx.eval(cache_shaped)

        # With transpose (current path)
        times_transpose = []
        for _ in range(trials):
            t0 = time.perf_counter()
            result = cache_shaped[0, :, :, :].transpose(1, 0, 2)
            mx.eval(result)
            times_transpose.append((time.perf_counter() - t0) * 1000)

        # Without transpose (if layout matched)
        pool_shaped = mx.random.normal((num_tokens, n_kv_heads, head_dim)).astype(
            mx.float16
        )
        mx.eval(pool_shaped)

        times_direct = []
        for _ in range(trials):
            t0 = time.perf_counter()
            result = pool_shaped  # no-op, just the view
            mx.eval(result)
            times_direct.append((time.perf_counter() - t0) * 1000)

        trans_med = _median(times_transpose)
        direct_med = _median(times_direct)
        overhead = trans_med - direct_med
        print(
            f"  tokens={num_tokens:>3d}  transpose={trans_med:7.4f}ms  "
            f"direct={direct_med:7.4f}ms  overhead={overhead:7.4f}ms"
        )


# ─────────────────────────────────────────────────────────────
# L5: mx.clear_cache() interval
# ─────────────────────────────────────────────────────────────


def bench_l5_clear_interval(
    runner,
    batch_size=4,
    prefill_len=128,
    intervals=(64, 128, 256, 512, 1024, 0),
    steps=100,
    warmup=10,
):
    _banner("L5: mx.clear_cache() interval tuning")

    if not runner._cache_layout.has_auxiliary_state:
        print("  Using dense model path")

    req_ids = _build_caches(runner, batch_size, prefill_len)

    for _ in range(warmup):
        _decode_one_step(runner, req_ids)

    for interval in intervals:
        label = f"every {interval}" if interval > 0 else "never"
        runner._decode_step_ct = 0
        mx.clear_cache()

        times = []
        for step in range(steps):
            t0 = time.perf_counter()
            pending = runner.decode_batch_start(req_ids)
            cache_arrays = runner._cache_state_arrays(pending.caches)
            mx.eval(pending.lazy_tokens, *cache_arrays)
            raw = pending.lazy_tokens.tolist()
            if not isinstance(raw, list):
                raw = [raw]
            next_tokens = [int(t) for t in raw]
            for i, rid in enumerate(pending.req_ids):
                runner._req_token_ids[rid].append(next_tokens[i])
            if interval > 0 and (step + 1) % interval == 0:
                mx.clear_cache()
            times.append((time.perf_counter() - t0) * 1000)

        med = _median(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        print(
            f"  interval={label:>10s}  median={med:7.3f}ms  "
            f"p95={p95:7.3f}ms  p99={p99:7.3f}ms"
        )

    for rid in req_ids:
        runner.remove_request(rid)


# ─────────────────────────────────────────────────────────────
# E2E: Full decode throughput comparison
# ─────────────────────────────────────────────────────────────


def bench_e2e_decode(
    runner, batch_sizes=(1, 4, 8), prefill_len=128, decode_steps=50, warmup=5
):
    _banner("E2E: Decode throughput baseline")

    for bs in batch_sizes:
        req_ids = _build_caches(runner, bs, prefill_len)

        for _ in range(warmup):
            _decode_one_step(runner, req_ids)

        t0 = time.perf_counter()
        for _ in range(decode_steps):
            _decode_one_step(runner, req_ids)
        elapsed = time.perf_counter() - t0

        total_tokens = bs * decode_steps
        throughput = total_tokens / elapsed
        ms_per_step = elapsed * 1000 / decode_steps
        print(
            f"  B={bs:>2d}  steps={decode_steps}  "
            f"{ms_per_step:7.2f}ms/step  {throughput:7.1f} tok/s"
        )

        for rid in req_ids:
            runner.remove_request(rid)


# ─────────────────────────────────────────────────────────────
# L6: Batched DeltaNet — stack cache states across requests
# ─────────────────────────────────────────────────────────────


def bench_l6_batched_deltanet(
    runner, batch_size=8, prefill_len=128, warmup=5, trials=30
):
    """Compare serial per-request DeltaNet with batched (B>1) execution.

    The mlx-lm GatedDeltaNet Metal kernel grid is (32, Dv, B*Hv), so it
    naturally supports B>1. We stack per-request ArraysCache states along
    the batch dimension, run the layer once, then split back.
    """
    _banner("L6: Batched DeltaNet (stack caches, one call per layer)")

    if not runner._cache_layout.has_auxiliary_state:
        print("  SKIP: model has no auxiliary state")
        return

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        AttentionOffsetCache,
        BatchedDecodeContext,
        clear_context,
        set_context,
    )

    req_ids = _build_caches(runner, batch_size, prefill_len)
    caches = [runner._req_caches[rid] for rid in req_ids]

    for _ in range(warmup):
        _decode_one_step(runner, req_ids)
        caches = [runner._req_caches[rid] for rid in req_ids]

    aux_indices = list(runner._cache_layout.auxiliary_layer_indices)
    layout = runner._cache_layout

    # --- Approach A: Serial per-request (current implementation) ---
    times_serial = []
    for _ in range(trials):
        last_tokens = [runner._req_token_ids[rid][-1] for rid in req_ids]
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
        hidden_states = runner._model_embed(batched_input)

        t0 = time.perf_counter()
        for layer_idx in aux_indices:
            layer = layout.layers[layer_idx]
            per_req = [hidden_states[i : i + 1] for i in range(batch_size)]
            results = []
            for i in range(batch_size):
                results.append(layer(per_req[i], mask=None, cache=caches[i][layer_idx]))
            hidden_states = mx.concatenate(results, axis=0)
        mx.eval(hidden_states)
        times_serial.append((time.perf_counter() - t0) * 1000)

    # --- Approach B: Batched DeltaNet (stack cache states) ---
    # Reload caches fresh
    for rid in req_ids:
        runner.remove_request(rid)
    req_ids = _build_caches(runner, batch_size, prefill_len)
    caches = [runner._req_caches[rid] for rid in req_ids]
    for _ in range(warmup):
        _decode_one_step(runner, req_ids)
        caches = [runner._req_caches[rid] for rid in req_ids]

    times_batched = []
    for _ in range(trials):
        last_tokens = [runner._req_token_ids[rid][-1] for rid in req_ids]
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
        hidden_states = runner._model_embed(batched_input)

        t0 = time.perf_counter()
        for layer_idx in aux_indices:
            layer = layout.layers[layer_idx]

            # Stack cache states: per-request (1, ...) → batched (B, ...)
            req_caches = [caches[i][layer_idx] for i in range(batch_size)]
            cache_size = len(req_caches[0].cache)
            batched_cache_data = []
            for slot in range(cache_size):
                arrays = [rc.cache[slot] for rc in req_caches]
                if arrays[0] is not None:
                    batched_cache_data.append(mx.concatenate(arrays, axis=0))
                else:
                    batched_cache_data.append(None)

            # Build a temporary batched ArraysCache
            from mlx_lm.models.cache import ArraysCache

            batched_ac = ArraysCache(cache_size)
            batched_ac.cache = batched_cache_data

            hidden_states = layer(hidden_states, mask=None, cache=batched_ac)

            # Split back to per-request
            for slot in range(cache_size):
                if batched_ac.cache[slot] is not None:
                    parts = [
                        batched_ac.cache[slot][i : i + 1] for i in range(batch_size)
                    ]
                    for i in range(batch_size):
                        req_caches[i].cache[slot] = parts[i]

        mx.eval(hidden_states)
        times_batched.append((time.perf_counter() - t0) * 1000)

    serial_med = _median(times_serial)
    batched_med = _median(times_batched)
    speedup = serial_med / batched_med if batched_med > 0 else float("inf")
    print(f"  B={batch_size}  aux_layers={len(aux_indices)}")
    print(
        f"  serial  (current):  {serial_med:7.2f}ms  "
        f"({len(aux_indices) * batch_size} layer calls)"
    )
    print(
        f"  batched (proposed): {batched_med:7.2f}ms  "
        f"({len(aux_indices)} layer calls)"
    )
    print(f"  speedup:            {speedup:.2f}x")

    for rid in req_ids:
        runner.remove_request(rid)


def bench_l6_batched_e2e(
    runner, batch_sizes=(1, 4, 8), prefill_len=128, decode_steps=50, warmup=5
):
    """E2E decode throughput with batched DeltaNet."""
    _banner("L6 E2E: Decode with batched DeltaNet")

    if not runner._cache_layout.has_auxiliary_state:
        print("  SKIP: model has no auxiliary state")
        return

    from mlx_lm.models.cache import ArraysCache

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        AttentionOffsetCache,
        BatchedDecodeContext,
        clear_context,
        set_context,
    )

    layout = runner._cache_layout
    aux_indices = list(layout.auxiliary_layer_indices)

    def _decode_hybrid_batched_deltanet(runner, caches, batched_input):
        batch_size = len(caches)
        hidden_states = runner._model_embed(batched_input)

        seq_lens = [
            runner._first_attention_cache(caches[i]).offset for i in range(batch_size)
        ]
        attention_layer_caches = layout.attention_layer_caches(caches)
        ctx = BatchedDecodeContext(
            batch_size=batch_size,
            seq_lens=seq_lens,
            attention_layer_caches=attention_layer_caches,
            attention_pool_index_by_layer=layout.attention_pool_index_by_layer,
        )
        max_offset = max(seq_lens)

        for layer_idx in range(layout.num_layers):
            layer = layout.layers[layer_idx]

            if layout.attention_attrs[layer_idx] is not None:
                set_context(ctx)
                try:
                    shim = AttentionOffsetCache(offset=max_offset)
                    hidden_states = layer(hidden_states, mask=None, cache=shim)
                finally:
                    clear_context()
            else:
                if batch_size == 1:
                    hidden_states = layer(
                        hidden_states, mask=None, cache=caches[0][layer_idx]
                    )
                else:
                    req_caches = [caches[i][layer_idx] for i in range(batch_size)]
                    cache_size = len(req_caches[0].cache)
                    batched_cache_data = []
                    for slot in range(cache_size):
                        arrays = [rc.cache[slot] for rc in req_caches]
                        if arrays[0] is not None:
                            batched_cache_data.append(mx.concatenate(arrays, axis=0))
                        else:
                            batched_cache_data.append(None)

                    batched_ac = ArraysCache(cache_size)
                    batched_ac.cache = batched_cache_data

                    hidden_states = layer(hidden_states, mask=None, cache=batched_ac)

                    for slot in range(cache_size):
                        if batched_ac.cache[slot] is not None:
                            for i in range(batch_size):
                                req_caches[i].cache[slot] = batched_ac.cache[slot][
                                    i : i + 1
                                ]

        hidden_states = runner._model_norm(hidden_states)
        logits = runner._extract_logits(runner._model_lm_head(hidden_states))
        return mx.argmax(logits[:, -1, :], axis=-1)

    for bs in batch_sizes:
        req_ids = _build_caches(runner, bs, prefill_len)

        for _ in range(warmup):
            _decode_one_step(runner, req_ids)

        # Baseline (current hybrid batching)
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            _decode_one_step(runner, req_ids)
        baseline_elapsed = time.perf_counter() - t0
        baseline_tps = bs * decode_steps / baseline_elapsed

        # Now test with batched DeltaNet
        for rid in req_ids:
            runner.remove_request(rid)
        req_ids = _build_caches(runner, bs, prefill_len)
        for _ in range(warmup):
            _decode_one_step(runner, req_ids)

        t0 = time.perf_counter()
        for _ in range(decode_steps):
            caches_list = [runner._req_caches[rid] for rid in req_ids]
            last_tokens = [runner._req_token_ids[rid][-1] for rid in req_ids]
            batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
            lazy_tokens = _decode_hybrid_batched_deltanet(
                runner, caches_list, batched_input
            )
            cache_arrays = runner._cache_state_arrays(caches_list)
            mx.eval(lazy_tokens, *cache_arrays)
            raw = lazy_tokens.tolist()
            if not isinstance(raw, list):
                raw = [raw]
            next_tokens = [int(t) for t in raw]
            for i, rid in enumerate(req_ids):
                runner._req_token_ids[rid].append(next_tokens[i])
        batched_elapsed = time.perf_counter() - t0
        batched_tps = bs * decode_steps / batched_elapsed

        speedup = batched_tps / baseline_tps if baseline_tps > 0 else float("inf")
        print(
            f"  B={bs:>2d}  baseline={baseline_tps:7.1f} tok/s  "
            f"batched={batched_tps:7.1f} tok/s  speedup={speedup:.2f}x"
        )

        for rid in req_ids:
            runner.remove_request(rid)


def main():
    parser = argparse.ArgumentParser(description="Validate remaining MLX perf levers")
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-0.8B", help="Hybrid model to test"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated: l1,l1b,l2,l3,l4,l5,l6,l6e2e,e2e",
    )
    args = parser.parse_args()

    selected = set(args.only.lower().split(",")) if args.only else None

    print(f"Loading model: {args.model}")
    runner = _load_runner(args.model)
    print(
        f"Model loaded. Layers: {runner._cache_layout.num_layers} "
        f"(attn: {runner._cache_layout.num_attention_layers}, "
        f"aux: {len(runner._cache_layout.auxiliary_layer_indices)})"
    )

    if selected is None or "e2e" in selected:
        bench_e2e_decode(runner, batch_sizes=(1, 4, 8), prefill_len=args.prefill_len)

    if selected is None or "l1" in selected:
        bench_l1_compile(
            runner, batch_size=args.batch_size, prefill_len=args.prefill_len
        )

    if selected is None or "l1b" in selected:
        bench_l1_compile_manual(
            runner, batch_size=args.batch_size, prefill_len=args.prefill_len
        )

    if selected is None or "l2" in selected:
        bench_l2_batched_sync(
            runner, batch_size=args.batch_size, prefill_len=args.prefill_len
        )

    if selected is None or "l3" in selected:
        bench_l3_deferred_snapshots(
            runner, batch_size=args.batch_size, prefill_len=args.prefill_len
        )

    if selected is None or "l4" in selected:
        bench_l4_transpose()

    if selected is None or "l5" in selected:
        bench_l5_clear_interval(
            runner, batch_size=min(args.batch_size, 4), prefill_len=args.prefill_len
        )

    if selected is None or "l6" in selected:
        bench_l6_batched_deltanet(
            runner, batch_size=args.batch_size, prefill_len=args.prefill_len
        )

    if selected is None or "l6e2e" in selected:
        bench_l6_batched_e2e(
            runner, batch_sizes=(1, 4, 8), prefill_len=args.prefill_len
        )


if __name__ == "__main__":
    main()
